#
# Copyright (C) 2015  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
from __future__ import division

from enum import Enum
import itertools
import math

from pyclick.click_models.ClickModel import ClickModel
from pyclick.click_models.Inference import EMInference
from pyclick.click_models.Param import ParamEM, ParamStatic
from pyclick.click_models.ParamContainer import QueryDocumentParamContainer, SingleParamContainer

__author__ = 'Ilya Markov, Luka Stout, Aleksandr Chuklin'


class CCM(ClickModel):
    """
    The click chain model (CCM) according to the following paper:
    Guo, Fan and Liu, Chao and Kannan, Anitha and Minka, Tom and Taylor, Michael and Wang, Yi-Min and Faloutsos, Christos.
    Click chain model in web search.
    Proceedings of WWW, pages 11-20, 2009.

    CCM contains a set of attractiveness parameters,
    which depends on a query and a document.
    It also contains three continuation (persistence) parameters.
    """

    param_names = Enum('CCMParamNames',
                       'attr cont_noclick cont_click_nonrel cont_click_rel exam car')
    """
    The names of the DBN parameters.

    :attr: the attractiveness parameter.
    Determines whether a user clicks on a search results after examining it.
    :cont_noclick: the examination parameter (tau_1).
    Determines whether a user continues examining search results after
        not clicking the current one.
    :cont_click_nonrel: the examination parameter (tau_2).
    Determines whether a user continues examining search results after
        clicking a non-relevant result.
    :cont_click_rel: the examination parameter (tau_3).
    Determines whether a user continues examining search results after clicking
        a relevant result.

    :exam: the examination probability.
    Not defined explicitly in the CCM model, but needs to be calculated during inference.
    Determines whether a user examines a particular search result.
    :car: the probability of click on or after rank $r$ given examination at rank $r$.
    Not defined explicitly in the CCM model, but needs to be calculated during inference.
    Determines whether a user clicks on the current result or any result below the current one.
    """

    def __init__(self, inference=EMInference()):
        self.params = {self.param_names.attr: QueryDocumentParamContainer(CCMAttrEM),
            self.param_names.cont_noclick: SingleParamContainer(CCMContNoclickEM),
            self.param_names.cont_click_nonrel: SingleParamContainer(CCMContClickNonrelEM),
            self.param_names.cont_click_rel: SingleParamContainer(CCMContClickRelEM)}
        self._inference = inference

    def get_session_params(self, search_session):
        session_params = super(CCM, self).get_session_params(search_session)

        session_exam = self._get_session_exam(session_params)
        session_clickafterrank = self._get_session_clickafterrank(
                len(search_session.web_results),
                session_params)

        for rank, session_param in enumerate(session_params):
            session_param[self.param_names.exam] = ParamStatic(session_exam[rank])
            session_param[self.param_names.car] = ParamStatic(session_clickafterrank[rank])

        return session_params

    def get_full_click_probs(self, search_session):
        session_params = self.get_session_params(search_session)
        click_probs = []

        for rank, session_param in enumerate(session_params):
            attr = session_param[self.param_names.attr].value()
            exam = session_param[self.param_names.exam].value()

            click_probs.append(attr * exam)

        return click_probs

    def get_conditional_click_probs(self, search_session):
        session_params = self.get_session_params(search_session)
        return self._get_tail_clicks(search_session, 0, session_params)[0]

    def predict_relevance(self, query, search_result):
        attr = self.params[self.param_names.attr].get(query, search_result).value()
        return attr**2

    @classmethod
    def _get_session_exam(cls, session_params):
        """
        Calculates the examination probability P(E_r=1) for each search result in a given search session.

        :param session_params: The current values of parameters for a given search session.

        :returns: The list of examination probabilities for a given search session.
        """
        session_exam = [1]

        for rank, session_param in enumerate(session_params):
            attr = session_param[cls.param_names.attr].value()
            tau_1 = session_param[cls.param_names.cont_noclick].value()
            tau_2 = session_param[cls.param_names.cont_click_nonrel].value()
            tau_3 = session_param[cls.param_names.cont_click_rel].value()
            exam = session_exam[rank]

            exam *= (1 - attr) * tau_1 + attr * ((1 - attr) * tau_2 + attr * tau_3)
            session_exam.append(exam)

        return session_exam

    @classmethod
    def _get_tail_clicks(cls, search_session, start_rank, session_params):
        """
        Calculate P(C_r | C_{r-1}, ..., C_l, E_l = 1), P(E_r = 1 | C_{r-1}, ..., C_l, E_l = 1)
        for each r in [l, n) where l is start_rank.
        """
        exam = 1.0
        click_probs = []
        exam_probs = [exam]
        for rank, result in enumerate(search_session.web_results[start_rank:]):
            attr = session_params[rank][cls.param_names.attr].value()
            tau_1 = session_params[rank][cls.param_names.cont_noclick].value()
            tau_2 = session_params[rank][cls.param_names.cont_click_nonrel].value()
            tau_3 = session_params[rank][cls.param_names.cont_click_rel].value()

            if result.click:
                click_prob = attr * exam
                exam = tau_2 * (1 - attr) + tau_3 * attr
            else:
                click_prob = 1 - attr * exam
                exam *= tau_1 * (1 - attr) / click_prob
            click_probs.append(click_prob)
            exam_probs.append(exam)
        return click_probs, exam_probs

    @classmethod
    def _get_continuation_factor(cls, search_session, rank, session_params):
        """Calculate P(E_r = x, S_r = y, E_{r+1} = z, \mathbf{C}) up to a constant."""
        click = search_session.web_results[rank].click
        attr = session_params[rank][cls.param_names.attr].value()
        tau_1 = session_params[rank][cls.param_names.cont_noclick].value()
        tau_2 = session_params[rank][cls.param_names.cont_click_nonrel].value()
        tau_3 = session_params[rank][cls.param_names.cont_click_rel].value()

        def factor(x, y, z):
            log_prob = 0.0
            # We use the chain rule for probabilities and drop the first term P(\mathbf{C}_{<r})
            # right away.
            #
            # First, compute the middle part P(C_r, S_r, E_{r+1} | E_r)
            if not click:
                if y:
                    # no click -> no satisfaction
                    return 0.0
                log_prob += math.log(1 - attr)
                if x:
                    log_prob += math.log(tau_1 if z else (1 - tau_1))
                elif z:
                    # no examination at r -> no examination at r+1
                    return 0.0
            else:
                if not x:
                    return 0.0
                log_prob += math.log(attr)
                if not y:
                    log_prob += math.log(1 - attr)
                    log_prob += math.log(tau_2 if z else (1 - tau_2))
                else:
                    log_prob += math.log(attr)
                    log_prob += math.log(tau_3 if z else (1 - tau_3))
            # Then we compute P(\mathbf{C}_{>r} | E_{r+1} = z)
            if not z:
                if search_session.get_last_click_rank() >= rank + 1:
                    # no examination -> no clicks
                    return 0.0
            elif rank + 1 < len(search_session.web_results):
                log_prob += sum(
                    math.log(p) for p in cls._get_tail_clicks(search_session,
                                                              rank + 1,
                                                              session_params)[0])
            # Finally, we compute P(E_r = 1 | \mathbf{C}_{<r})
            exam = cls._get_tail_clicks(search_session, 0, session_params)[1][rank]
            log_prob += math.log(exam if x else (1 - exam))
            return math.exp(log_prob)
        return factor

    @classmethod
    def _get_session_clickafterrank(cls, session_size, session_params):
        """
        For each search result in a given search session,
        calculates the probability of a click on the current result
        or any result below the current one given examination at the current rank,
        i.e., P(C_{>=r} = 1 | E_r = 1), where r is the rank of the current search result.

        :param session_size: The size of the observed search session.
        :param session_params: The current values of parameters for a given search session.

        :returns: The list of P(C_{>=r} = 1 | E_r = 1) for a given search session.
        """
        session_clickafterrank = [0] * (session_size + 1)

        for rank in reversed(range(session_size)):
            attr = session_params[rank][cls.param_names.attr].value()
            tau_1 = session_params[rank][cls.param_names.cont_noclick].value()
            car = session_clickafterrank[rank + 1]

            car = attr + (1 - attr) * tau_1 * car
            session_clickafterrank[rank] = car

        return session_clickafterrank


class CCMAttrEM(ParamEM):
    """
    The attractiveness parameter of the CCM model.
    The value of the parameter is inferred using the EM algorithm.
    """
    @classmethod
    def _get_numerator_update(cls, search_session, rank, session_params):
        click = search_session.web_results[rank].click
        last_click_rank = search_session.get_last_click_rank()

        numerator_update = 0

        # 1. The attractiveness part (analogy with DBN):
        if click:
            numerator_update += 1
        elif rank >= last_click_rank:
            attr = session_params[rank][CCM.param_names.attr].value()
            exam = session_params[rank][CCM.param_names.exam].value()
            car = session_params[rank][CCM.param_names.car].value()

            numerator_update += (1 - exam) * attr / (1 - exam * car)

        # 2. The satisfaction part (analogy with DBN):
        if click and rank == last_click_rank:
            attr = session_params[rank][CCM.param_names.attr].value()
            tau_2 = session_params[rank][CCM.param_names.cont_click_nonrel].value()
            tau_3 = session_params[rank][CCM.param_names.cont_click_rel].value()
            car = session_params[rank + 1][CCM.param_names.car].value() \
                if rank < len(search_session.web_results) - 1 \
                else 0

            numerator_update += attr / (1 - (tau_2 * (1 - attr) + tau_3 * attr) * car)

        return numerator_update

    @classmethod
    def _get_denominator_update(cls, search_session, rank, session_params):
        denominator_update = 1

        if search_session.web_results[rank].click:
            denominator_update += 1

        return denominator_update

    # def update(self, search_session, rank, session_params):
    #     click = search_session.web_results[rank].click
    #     last_click_rank = search_session.get_last_click_rank()
    #
    #     # First, compute the denominator:
    #     self._denominator += 1
    #     if click:
    #         self._denominator += 1
    #
    #     # Now, the numerator.
    #     #
    #     # 1. The attractiveness part (analogy with DBN):
    #     if click:
    #         self._numerator += 1
    #     elif rank >= last_click_rank:
    #         attr = session_params[rank][CCM.param_names.attr].value()
    #         exam = session_params[rank][CCM.param_names.exam].value()
    #         car = session_params[rank][CCM.param_names.car].value()
    #
    #         self._numerator += (1 - exam) * attr / (1 - exam * car)
    #     # 2. The satisfaction part (analogy with DBN):
    #     if click and rank == last_click_rank:
    #         attr = session_params[rank][CCM.param_names.attr].value()
    #         tau_2 = session_params[rank][CCM.param_names.cont_click_nonrel].value()
    #         tau_3 = session_params[rank][CCM.param_names.cont_click_rel].value()
    #         car = session_params[rank + 1][CCM.param_names.car].value() \
    #             if rank < len(search_session.web_results) - 1 \
    #             else 0
    #
    #         self._numerator += attr / (1 - (tau_2 * (1 - attr) + tau_3 * attr) * car)


class CCMContEM(ParamEM):
    """
    The abstract continuation parameter of the CCM model.
    The value of the parameter is inferred using the EM algorithm.
    """
    @classmethod
    def _get_numerator_update(cls, search_session, rank, session_params):
        return cls._get_exam_prob(search_session, rank, session_params, 1)

    @classmethod
    def _get_denominator_update(cls, search_session, rank, session_params):
        return sum(cls._get_exam_prob(search_session, rank, session_params, x) for x in [0, 1])

    @classmethod
    def _get_exam_prob(cls, search_session, rank, session_params, value):
        pass


class CCMContNoclickEM(CCMContEM):
    """
    The continuation parameter in case of no click.
    """
    @classmethod
    def _is_update_needed(cls, search_session, rank):
        return not search_session.web_results[rank].click

    @classmethod
    def _get_exam_prob(cls, search_session, rank, session_params, value):
        factor = CCM._get_continuation_factor(search_session, rank, session_params)
        # P(E_r = 1, E_{r+1} = z | C)
        return (factor(1, 0, value) + factor(1, 1, value)) / sum(
            factor(*p) for p in itertools.product([0, 1], repeat=3))

    # def update(self, search_session, rank, session_params):
    #     if not search_session.web_results[rank].click:
    #         factor = CCM._get_continuation_factor(search_session, rank, session_params)
    #         # P(E_r = 1, E_{r+1} = z | C)
    #         exam_prob = lambda z: (factor(1, 0, z) + factor(1, 1, z)) / sum(
    #                 factor(*p) for p in itertools.product([0, 1], repeat=3))
    #         self._numerator += exam_prob(1)
    #         self._denominator += sum(exam_prob(x) for x in [0, 1])


class CCMContClickNonrelEM(CCMContEM):
    """
    The continuation parameter in case of a click on a non-relevant document.
    """
    @classmethod
    def _is_update_needed(cls, search_session, rank):
        return search_session.web_results[rank].click

    @classmethod
    def _get_exam_prob(cls, search_session, rank, session_params, value):
        factor = CCM._get_continuation_factor(search_session, rank, session_params)
        # P(E_r = 1, S_r = 0, E_{r+1} = z | C)
        return factor(1, 0, value) / sum(
            factor(*p) for p in itertools.product([0, 1], repeat=3))

    # def update(self, search_session, rank, session_params):
    #     if search_session.web_results[rank].click:
    #         factor = CCM._get_continuation_factor(search_session, rank, session_params)
    #         # P(E_r = 1, S_r = 0, E_{r+1} = z | C)
    #         exam_prob = lambda z: factor(1, 0, z) / sum(
    #                 factor(*p) for p in itertools.product([0, 1], repeat=3))
    #         self._numerator += exam_prob(1)
    #         self._denominator += sum(exam_prob(x) for x in [0, 1])


class CCMContClickRelEM(CCMContEM):
    """
    The continuation parameter in case of a click on a relevant document.
    """
    @classmethod
    def _is_update_needed(cls, search_session, rank):
        return search_session.web_results[rank].click

    @classmethod
    def _get_exam_prob(cls, search_session, rank, session_params, value):
        factor = CCM._get_continuation_factor(search_session, rank, session_params)
        # P(E_r = 1, S_r = 1, E_{r+1} = z | C)
        return factor(1, 1, value) / sum(
            factor(*p) for p in itertools.product([0, 1], repeat=3))

    # def update(self, search_session, rank, session_params):
    #     if search_session.web_results[rank].click:
    #         factor = CCM._get_continuation_factor(search_session, rank, session_params)
    #         # P(E_r = 1, S_r = 1, E_{r+1} = z | C)
    #         exam_prob = lambda z: factor(1, 1, z) / sum(
    #                 factor(*p) for p in itertools.product([0, 1], repeat=3))
    #         self._numerator += exam_prob(1)
    #         self._denominator += sum(exam_prob(x) for x in [0, 1])
