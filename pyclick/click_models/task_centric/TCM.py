#
# Copyright (C) 2015  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
from abc import abstractmethod
from enum import Enum

from pyclick.click_models.ClickModel import ClickModel
from pyclick.click_models.Param import ParamEM
from pyclick.click_models.ParamContainer import QueryDocumentParamContainer, RankParamContainer, SingleParamContainer
from pyclick.click_models.task_centric.TaskCentricInferenceEM import TaskCentricEMInference

__author__ = 'Aleksandr Chuklin, Ilya Markov'


class TCM(ClickModel):
    """
    The task-centric click model (TCM) according to the following paper:
    Zhang, Yuchen and Chen, Weizhu and Wang, Dong and Yang, Qiang.
    User-click modeling for understanding and predicting search-behavior.
    Proceedings of SIGKDD, pages 1388-1396, 2011.

    TCM takes into account search tasks consisting of multiple query sessions.
    This implementation is built on top of the position-based click model (PBM).
    """

    param_names = Enum('TCMParamNames', 'attr exam match new fresh')
    """
    The names of the TCM parameters.

    :attr: the attractiveness parameter.
    Determines whether a user clicks on a search results after examining it.
    :exam: the examination probability.
    Determines whether a user examines a particular search result.
    :match: the match parameter.
    Determines whether a query matches the user's information need.
    :new: the "new query" parameter.
    Determines whether a user submits a new query after a matching query.
    :fresh: the freshness parameter.
    Determines whether a search result is still interesting to a user,
    despite that it has already been presented to the user within the same search task.
    """

    def __init__(self):
        self.params = {self.param_names.attr: QueryDocumentParamContainer(TCMAttrEM),
                       self.param_names.exam: RankParamContainer.default(TCMExamEM),
                       self.param_names.match: SingleParamContainer(TCMMatchEM),
                       self.param_names.new: SingleParamContainer(TCMNewEM),
                       self.param_names.fresh: SingleParamContainer(TCMFreshEM)}
        self._inference = TaskCentricEMInference()

    def get_conditional_click_probs(self, search_session):
        click_probs = self.get_full_click_probs(search_session)

        for rank, result in enumerate(search_session.web_results):
            if not result.click:
                click_probs[rank] = 1 - click_probs[rank]

        return click_probs

    def get_full_click_probs(self, search_session):
        session_params = self.get_session_params(search_session)
        click_probs = []

        for rank, session_param in enumerate(session_params):
            attr = session_param[self.param_names.attr].value()
            exam = session_param[self.param_names.exam].value()

            click_prob = attr * exam
            click_probs.append(click_prob)

        return click_probs

    def predict_relevance(self, query, search_result):
        return self.params[self.param_names.attr].get(query, search_result).value()


class TCMParamEM(ParamEM):
    """A parameter that uses multiple sessions in the same task and is trained using the EM algorithm."""

    def update(self, search_task, search_session, rank, session_params):
        session_index = search_task.search_sessions.index(search_session)
        is_last_session = session_index == len(search_task.search_sessions) - 1
        previous_results = self._get_previous_results(search_task, session_index)

        self._update(search_session, rank, session_params, previous_results, is_last_session)

    @abstractmethod
    def _update(self, search_session, rank, session_params, previous_results, is_last_session):
        """
        Updates the parameter using history (previously shown results) and the last_session flag
        (set to true if this is the last session in the task).

        :param search_session: The current search session.
        :param rank: The rank of the current search result.
        :param previous_results: The set of search results previously shown within the current search task.
        :param is_last_session: The indicator whether this is the last session in the task.
        """
        pass

    @staticmethod
    def _get_previous_results(search_task, session_index):
        """
        Returns the list of search results shown to a user within the given search task
        prior to the given search session.

        :param search_task: The search task.
        :param session_index: The index of the given session withing the given search task.

        :returns: The list of previously shown search results.
        """

        # TODO(chuklin): in the original paper, history is not just the occurrence of the document,
        # but rather examination of it.
        # Ideally, one should sum over position of the first examined occurrence: equation (27).

        previous_results = set()
        for i in range(session_index):
            for result in search_task.search_sessions[i].web_results:
                previous_results.add(result.id)
        return previous_results


class TCMAttrEM(TCMParamEM):
    """
    The attractiveness parameter of the TCM model.
    """
    def _update(self, search_session, rank, session_params, previous_results, is_last_session):
        result = search_session.web_results[rank]

        attr = session_params[rank][TCM.param_names.attr].value()
        exam = session_params[rank][TCM.param_names.exam].value()
        fresh = session_params[rank][TCM.param_names.fresh].value() \
            if (result.id in previous_results) else 1.0
        match = session_params[rank][TCM.param_names.match].value()

        if result.click:
            self._numerator += 1
        else:
            self._numerator += ((1 - exam * fresh * match) * attr /
                                (1 - exam * attr * fresh * match))

        self._denominator += 1


class TCMExamEM(TCMParamEM):
    """
    The examination parameter of the TCM model.
    """
    def _update(self, search_session, rank, session_params, previous_results, is_last_session):
        result = search_session.web_results[rank]

        attr = session_params[rank][TCM.param_names.attr].value()
        exam = session_params[rank][TCM.param_names.exam].value()
        fresh = session_params[rank][TCM.param_names.fresh].value() \
            if (result.id in previous_results) else 1.0
        match = session_params[rank][TCM.param_names.match].value()

        if result.click:
            self._numerator += 1
        else:
            self._numerator += ((1 - attr * fresh * match) * exam /
                                (1 - exam * attr * fresh * match))

        self._denominator += 1


class TCMMatchEM(TCMParamEM):
    """
    The parameter controlling if the query matches the user's information need or not.
    """
    def _update(self, search_session, rank, session_params, previous_results, is_last_session):
        self._numerator += self.get_match_given_session_prob(search_session, session_params,
                                                             previous_results, is_last_session)
        self._denominator += 1

    @staticmethod
    def get_no_clicks_given_match_prob(search_session, session_params, previous_results):
        """
        Computes and returns the probability of not having clicks in the given search session
        given the task match.

        :param search_session: The current search session.
        :param session_params: The click model parameters corresponding to the current session.
        :param previous_results: The list of search results shown previously (in the same search task).
        :param is_last_session: The indicator whether this is the last search session in the current search task.

        :returns: The probability of not having clicks in the given search session.
        """
        no_clicks_prob = 1.0

        for rank, result in enumerate(search_session.web_results):
            attr = session_params[rank][TCM.param_names.attr].value()
            exam = session_params[rank][TCM.param_names.exam].value()
            fresh = session_params[rank][TCM.param_names.fresh].value() \
                if (result.id in previous_results) else 1.0

            no_clicks_prob *= (1 - attr * exam * fresh)

        return no_clicks_prob

    @staticmethod
    def get_match_given_session_prob(search_session, session_params, previous_results, is_last_session):
        """
        Computes and returns the probability that the given search session (query)
        matches the user's information need.

        :param search_session: The current search session.
        :param session_params: The click model parameters corresponding to the current session.
        :param previous_results: The list of search results shown previously (in the same search task).
        :param is_last_session: The indicator whether this is the last search session in the current search task.

        :returns: The probability that the given search session matches the user's information need.
        """
        if any(search_session.get_clicks()) or is_last_session:
            return 1.0
        else:
            match = session_params[0][TCM.param_names.match].value()
            new = session_params[0][TCM.param_names.new].value()
            p = TCMMatchEM.get_no_clicks_given_match_prob(search_session, session_params,
                                                          previous_results)
            # Here we simplify the computation assuming that the next sessions' clicks
            # do not depend on the current session given the value of N parameter.
            return 1.0 / (1 + (1.0 / match - 1) / (p * new))


class TCMNewEM(TCMParamEM):
    """
    The parameter controlling whether there will be further search sessions (queries).
    """
    def _update(self, search_session, rank, session_params, previous_results, is_last_session):
        match_prob = TCMMatchEM.get_match_given_session_prob(search_session, session_params,
                                                             previous_results, is_last_session)
        if not is_last_session:
            self._numerator += match_prob

        self._denominator += match_prob


class TCMFreshEM(TCMParamEM):
    """
    The parameter controlling whether a search result is still interesting to a user,
    despite that it has already been presented to the user within the same search task.
    """
    def _update(self, search_session, rank, session_params, previous_results, is_last_session):
        result = search_session.web_results[rank]
        if result.id in previous_results:
            attr = session_params[rank][TCM.param_names.attr].value()
            exam = session_params[rank][TCM.param_names.exam].value()
            fresh = session_params[rank][TCM.param_names.fresh].value()
            match = session_params[rank][TCM.param_names.match].value()

            if result.click:
                self._numerator += 1
            else:
                self._numerator += (1 - exam * attr * match) * fresh / (1 - exam * attr * fresh * match)

            self._denominator += 1
