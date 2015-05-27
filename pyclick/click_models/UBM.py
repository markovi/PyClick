#
# Copyright (C) 2015  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
from enum import Enum

from pyclick.click_models.ClickModel import ClickModel
from pyclick.click_models.Inference import EMInference
from pyclick.click_models.Param import ParamEM
from pyclick.click_models.ParamContainer import QueryDocumentParamContainer, RankSquaredParamContainer


__author__ = 'Ilya Markov'


class UBM(ClickModel):
    """
    The user browsing model (UBM) according to the following paper:
    Dupret, Georges E. and Piwowarski, Benjamin.
    A user browsing model to predict search engine click data from past observations.
    Proceedings of SIGIR, pages 331-338, 2008.

    UBM contains a set of attractiveness and examination parameters.
    An attractiveness parameter depends on a query and a document.
    An examination parameter depends on the document rank and on the rank of the previously clicked document.

    Note, that ranks start from 0.
    If there is no click before the current document,
    the rank of the previously clicked document is considered to be M-1,
    where M is the length of a result list.
    """

    param_names = Enum('UBMParamNames', 'attr exam')
    """The names of the UBM parameters."""

    def __init__(self):
        self.params = {self.param_names.attr: QueryDocumentParamContainer(UBMAttrEM),
                       self.param_names.exam: RankSquaredParamContainer.default(UBMExamEM)}
        self._inference = EMInference()

    def get_session_params(self, search_session):
        session_params = []

        for rank, result in enumerate(search_session.web_results):
            attr = self.params[self.param_names.attr].get(search_session.query, result.id)
            exam = self.params[self.param_names.exam].get(rank, self._get_prev_clicked_rank(search_session, rank))

            param_dict = {self.param_names.attr: attr, self.param_names.exam: exam}
            session_params.append(param_dict)

        return session_params

    def get_conditional_click_probs(self, search_session):
        session_params = self.get_session_params(search_session)
        click_probs = []

        for rank, result in enumerate(search_session.web_results):
            attr = session_params[rank][self.param_names.attr].value()
            exam = session_params[rank][self.param_names.exam].value()

            if result.click:
                click_prob = attr * exam
            else:
                click_prob = 1 - attr * exam

            click_probs.append(click_prob)

        return click_probs

    def predict_click_probs(self, search_session):
        click_probs = []

        for rank, result in enumerate(search_session.web_results):
            click_prob = 0

            for rank_prev_click in range(-1, rank):
                # Compute probability that there is no clicks between rank_prev_click and rank
                no_click_between = 1
                for rank_between in range(rank_prev_click + 1, rank):
                    no_click_between *= 1 - self._get_click_prob(search_session, rank_between, rank_prev_click)

                click_prob += (click_probs[rank_prev_click] if rank_prev_click >= 0 else 1) * \
                              no_click_between * self._get_click_prob(search_session, rank, rank_prev_click)

            click_probs.append(click_prob)

        return click_probs

    def predict_relevance(self, query, search_result):
        return self.params[self.param_names.attr].get(query, search_result).value()

    def _get_click_prob(self, search_session, rank, rank_prev_click):
        """
        Returns the click probability for a search result at the given rank in the given search session,
        where the previously clicked result was at prev_clicked_rank.

        :param search_session: The current search session.
        :param rank: The rank of a search result.
        :param rank_prev_click: The rank of the previously clicked search result.

        :returns: The click probability for a given search result.
        """
        attr = self.params[self.param_names.attr].get(search_session.query, search_session.web_results[rank].id).value()
        exam = self.params[self.param_names.exam].get(rank, rank_prev_click).value()
        return attr * exam

    def _get_prev_clicked_rank(self, search_session, rank):
        """
        Given the rank, returns the rank of the previously clicked search result.
        If none of the above results was clicked,
        returns M-1, where M is the number of results in a given search session.

        :param search_session: The current search session.
        :param rank: The rank of a search result.

        :returns: The rank of the previously clicked search result.
        """
        prev_clicks = [rank_click for rank_click, click in enumerate(
                search_session.get_clicks()[:rank]) if click]
        prev_click_rank = prev_clicks[-1] if len(prev_clicks) else len(search_session.web_results) - 1
        return prev_click_rank


class UBMAttrEM(ParamEM):
    """
    The attractiveness parameter of the UBM model.
    The value of the parameter is inferred using the EM algorithm.
    """
    def update(self, search_session, rank, session_params):
        attr = session_params[rank][UBM.param_names.attr].value()
        exam = session_params[rank][UBM.param_names.exam].value()

        if search_session.web_results[rank].click:
            self._numerator += 1
        else:
            self._numerator += (1 - exam) * attr / (1 - exam * attr)

        self._denominator += 1


class UBMExamEM(ParamEM):
    """
    The examination parameter of the UBM model.
    The value of the parameter is inferred using the EM algorithm.
    """
    def update(self, search_session, rank, session_params):
        attr = session_params[rank][UBM.param_names.attr].value()
        exam = session_params[rank][UBM.param_names.exam].value()

        if search_session.web_results[rank].click:
            self._numerator += 1
        else:
            self._numerator += (1 - attr) * exam / (1 - exam * attr)

        self._denominator += 1
