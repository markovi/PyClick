#
# Copyright (C) 2015  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
from enum import Enum

from pyclick.click_models.ClickModel import ClickModel
from pyclick.click_models.Inference import MLEInference
from pyclick.click_models.Param import ParamMLE
from pyclick.click_models.ParamContainer import QueryDocumentParamContainer, RankParamContainer


__author__ = 'Ilya Markov'


class DCM(ClickModel):
    """
    The dependent click model (DCM) according to the following paper:
    Guo, Fan and Liu, Chao and Wang, Yi Min
    Efficient multiple-click models in web search.
    Proceedings of WSDM, pages 124-131, 2009.

    DCM contains the set of attractiveness and continuation parameters.
    An attractiveness parameter depends on a query and a document.
    A continuation parameter (1 - satisfaction) depends on the document rank.

    The original paper makes an additional assumption that a user is satisfied with the last clicked document,
    while all documents before it are examined.
    In this case, the MLE inference can be used instead of EM.
    """

    param_names = Enum('DCMParamNames', 'attr cont')
    """The names of the DCM parameters."""

    def __init__(self):
        self.params = {self.param_names.attr: QueryDocumentParamContainer(DCMAttrMLE),
                       self.param_names.cont: RankParamContainer.default(DCMContMLE)}
        self._inference = MLEInference()

    def get_full_click_probs(self, search_session):
        session_params = self.get_session_params(search_session)
        exam = 1
        click_probs = []

        for rank, session_param in enumerate(session_params):
            attr = session_param[self.param_names.attr].value()
            cont = session_param[self.param_names.cont].value()

            click_probs.append(attr * exam)
            exam *= cont * attr + (1 - attr)

        return click_probs

    def get_conditional_click_probs(self, search_session):
        session_params = self.get_session_params(search_session)
        exam = 1
        click_probs = []

        for rank, result in enumerate(search_session.web_results):
            attr = session_params[rank][self.param_names.attr].value()
            cont = session_params[rank][self.param_names.cont].value()

            if result.click:
                click_prob = attr * exam
                exam = cont
            else:
                click_prob = 1 - attr * exam
                exam *= (1 - attr) / click_prob

            click_probs.append(click_prob)

        return click_probs

    def predict_relevance(self, query, search_result):
        return self.params[self.param_names.attr].get(query, search_result).value()


class DCMAttrMLE(ParamMLE):
    """
    The attractiveness parameter of the DCM model.
    The value of the parameter is inferred using the MLE algorithm.
    """

    def update(self, search_session, rank):
        if rank <= search_session.get_last_click_rank():
            if search_session.web_results[rank].click:
                self._numerator += 1
            self._denominator += 1


class DCMContMLE(ParamMLE):
    """
    The continuation parameter of the DCM model (1 - satisfaction).
    The value of the parameter is inferred using the MLE algorithm.
    """

    def update(self, search_session, rank):
        if search_session.web_results[rank].click:
            if rank != search_session.get_last_click_rank():
                self._numerator += 1
            self._denominator += 1
