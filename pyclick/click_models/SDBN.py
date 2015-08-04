#
# Copyright (C) 2015  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
from enum import Enum

from pyclick.click_models.ClickModel import ClickModel
from pyclick.click_models.Inference import MLEInference
from pyclick.click_models.Param import ParamMLE
from pyclick.click_models.ParamContainer import QueryDocumentParamContainer


__author__ = 'Ilya Markov'


class SDBN(ClickModel):
    """
    The simplified dynamic bayesian network model (SDBN) according to the following paper:
    Chapelle, Olivier and Zhang, Ya.
    A dynamic bayesian network click model for web search ranking.
    Proceedings of WWW, pages 1-10, 2009.

    SDBN contains the set of attractiveness and satisfactoriness parameters,
    which both depend on a query and a document.

    SDBN differs from the standard DBN by setting the continuation parameter to 1.
    It assumes that all search results up to the last clicked result are examined,
    while others (below) are not.
    """

    param_names = Enum('SDBNParamNames', 'attr sat')
    """The names of the SDBN parameters."""

    def __init__(self):
        self.params = {self.param_names.attr: QueryDocumentParamContainer(SDBNAttrMLE),
                       self.param_names.sat: QueryDocumentParamContainer(SDBNSatMLE)}
        self._inference = MLEInference()

    def get_conditional_click_probs(self, search_session):
        session_params = self.get_session_params(search_session)
        exam = 1
        click_probs = []

        for rank, result in enumerate(search_session.web_results):
            attr = session_params[rank][self.param_names.attr].value()
            sat = session_params[rank][self.param_names.sat].value()

            if result.click:
                click_prob = attr * exam
                exam = 1 - sat
            else:
                click_prob = 1 - attr * exam
                exam *= (1 - attr) / click_prob

            click_probs.append(click_prob)

        return click_probs

    def get_full_click_probs(self, search_session):
        session_params = self.get_session_params(search_session)
        exam = 1
        click_probs = []

        for rank, session_param in enumerate(session_params):
            attr = session_param[self.param_names.attr].value()
            sat = session_param[self.param_names.sat].value()

            click_probs.append(attr * exam)
            exam *= (1 - sat) * attr + (1 - attr)

        return click_probs

    def predict_relevance(self, query, search_result):
        attr = self.params[self.param_names.attr].get(query, search_result).value()
        sat = self.params[self.param_names.sat].get(query, search_result).value()
        return attr * sat


class SDBNAttrMLE(ParamMLE):
    """
    The attractiveness parameter of the DBN model.
    The value of the parameter is inferred using the MLE algorithm.
    """

    def update(self, search_session, rank):
        if rank <= search_session.get_last_click_rank():
            if search_session.web_results[rank].click:
                self._numerator += 1
            self._denominator += 1


class SDBNSatMLE(ParamMLE):
    """
    The satisfactoriness parameter of the DBN model.
    The value of the parameter is inferred using the MLE algorithm.
    """

    def update(self, search_session, rank):
        if search_session.web_results[rank].click:
            if rank == search_session.get_last_click_rank():
                self._numerator += 1
            self._denominator += 1
