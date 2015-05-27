#
# Copyright (C) 2015  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
from enum import Enum

from pyclick.click_models.ClickModel import ClickModel
from pyclick.click_models.Inference import EMInference
from pyclick.click_models.Param import ParamEM, ParamStatic
from pyclick.click_models.ParamContainer import QueryDocumentParamContainer, SingleParamContainer


__author__ = 'Ilya Markov'


class DBN(ClickModel):
    """
    The dynamic Bayesian network click model (DBN) according to the following paper:
    Chapelle, Olivier and Zhang, Ya.
    A dynamic bayesian network click model for web search ranking.
    Proceedings of WWW, pages 1-10, 2009.

    DBN contains the set of attractiveness and satisfactoriness parameters,
    which both depend on a query and a document.
    It also contains a single continuation (persistence) parameter.
    """

    param_names = Enum('DBNParamNames', 'attr sat cont exam car')
    """
    The names of the DBN parameters.

    :attr: the attractiveness parameter.
    Determines whether a user clicks on a search results after examining it.
    :sat: the statisfactoriness parameter.
    Determines whether a user is satisfied with a search result (and abandons the corresponding search session)
    after clicking and reading the result.
    :cont: the continuation parameter.
    Determines whether a user continues examining search results after examining the current one.

    :exam: the examination probability.
    Not defined explicitly in the DBN model, but needs to be calculated during inference.
    Determines whether a user examines a particular search result.
    :car: the probability of click on or after rank $r$ given examination at rank $r$.
    Not defined explicitly in the DBN model, but needs to be calculated during inference.
    Determines whether a user clicks on the current result or any result below the current one.
    """

    def __init__(self):
        self.params = {self.param_names.attr: QueryDocumentParamContainer(DBNAttrEM),
                       self.param_names.sat: QueryDocumentParamContainer(DBNSatEM),
                       self.param_names.cont: SingleParamContainer(DBNContEM)}
        self._inference = EMInference()

    def get_session_params(self, search_session):
        session_params = []

        for rank, result in enumerate(search_session.web_results):
            attr = self.params[self.param_names.attr].get(search_session.query, result.id)
            sat = self.params[self.param_names.sat].get(search_session.query, result.id)
            cont = self.params[self.param_names.cont].get()

            param_dict = {self.param_names.attr: attr, self.param_names.sat: sat, self.param_names.cont: cont}
            session_params.append(param_dict)

        session_exam = self._get_session_exam(search_session, session_params)
        session_clickafterrank = self._get_session_clickafterrank(search_session, session_params)

        for rank, session_param in enumerate(session_params):
            session_param[self.param_names.exam] = ParamStatic(session_exam[rank])
            session_param[self.param_names.car] = ParamStatic(session_clickafterrank[rank])

        return session_params

    def predict_click_probs(self, search_session):
        session_params = self.get_session_params(search_session)
        click_probs = []

        for rank, session_param in enumerate(session_params):
            attr = session_param[self.param_names.attr].value()
            exam = session_param[self.param_names.attr].value()

            click_probs.append(attr * exam)

        return click_probs

    def get_conditional_click_probs(self, search_session):
        session_params = self.get_session_params(search_session)
        exam = 1
        click_probs = []

        for rank, result in enumerate(search_session.web_results):
            attr = session_params[rank][self.param_names.attr].value()
            sat = session_params[rank][self.param_names.sat].value()
            cont = session_params[rank][self.param_names.cont].value()

            if result.click:
                click_prob = attr * exam
                exam = cont * (1 - sat)
            else:
                click_prob = 1 - attr * exam
                exam *= cont * (1 - attr) / click_prob

            click_probs.append(click_prob)

        return click_probs

    def predict_relevance(self, query, search_result):
        attr = self.params[self.param_names.attr].get(query, search_result).value()
        sat = self.params[self.param_names.sat].get(query, search_result).value()
        return attr * sat

    def _get_session_exam(self, search_session, session_params):
        """
        Calculates the examination probability for each search result in a given search session.

        :param search_session: The observed search session.
        :param session_params: The current values of parameters for a given search session.

        :returns: The list of examination probabilities for a given search session.
        """
        session_exam = [1]

        for rank, session_param in enumerate(session_params):
            attr = session_param[self.param_names.attr].value()
            sat = session_param[self.param_names.sat].value()
            cont = session_param[self.param_names.cont].value()
            exam = session_exam[rank]

            exam *= cont * ((1 - sat) * attr + (1 - attr))
            session_exam.append(exam)

        return session_exam

    def _get_session_clickafterrank(self, search_session, session_params):
        """
        For each search result in a given search session,
        calculates the probability of a click on the current result
        or any result below the current one given examination at the current rank,
        i.e., P(C_{>=r} = 1 | E_r = 1), where r is the rank of the current search result.

        :param search_session: The observed search session.
        :param session_params: The current values of parameters for a given search session.

        :returns: The list of P(C_{>=r} = 1 | E_r = 1) for a given search session.
        """
        session_clickafterrank = [0] * (len(search_session.web_results) + 1)

        for rank in range(len(search_session.web_results) - 1, -1, -1):
            attr = session_params[rank][self.param_names.attr].value()
            cont = session_params[rank][self.param_names.cont].value()
            car = session_clickafterrank[rank + 1]

            car = attr + (1 - attr) * cont * car
            session_clickafterrank[rank] = car

        return session_clickafterrank


class DBNAttrEM(ParamEM):
    """
    The attractiveness parameter of the DBN model.
    The value of the parameter is inferred using the EM algorithm.
    """

    def update(self, search_session, rank, session_params):
        if search_session.web_results[rank].click:
            self._numerator += 1
        elif rank >= search_session.get_last_click_rank():
            attr = session_params[rank][DBN.param_names.attr].value()
            exam = session_params[rank][DBN.param_names.exam].value()
            car = session_params[rank][DBN.param_names.car].value()

            num = (1 - exam) * attr
            denom = 1 - exam * car
            self._numerator += num / denom

        self._denominator += 1


class DBNSatEM(ParamEM):
    """
    The satisfactoriness parameter of the DBN model.
    The value of the parameter is inferred using the EM algorithm.
    """

    def update(self, search_session, rank, session_params):
        if search_session.web_results[rank].click:
            if rank == search_session.get_last_click_rank():
                sat = session_params[rank][DBN.param_names.sat].value()
                cont = session_params[rank][DBN.param_names.cont].value()
                car = session_params[rank + 1][DBN.param_names.car].value() \
                    if rank < len(search_session.web_results) - 1 \
                    else 0

                self._numerator += sat / (1 - (1 - sat) * cont * car)

            self._denominator += 1


class DBNContEM(ParamEM):
    """
    The continuation (persistence) parameter of the DBN model.
    The value of the parameter is inferred using the EM algorithm.
    """

    def update(self, search_session, rank, session_params):
        if rank < search_session.get_last_click_rank():
            self._numerator += 1
        else:
            cont = session_params[rank][DBN.param_names.cont].value()
            car = session_params[rank + 1][DBN.param_names.car].value() \
                if rank < len(search_session.web_results) - 1 \
                else 0

            num = cont * (1 - car)
            denom = 1 - cont * car
            self._numerator += num / denom

        self._denominator += 1
