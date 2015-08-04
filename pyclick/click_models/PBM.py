#
# Copyright (C) 2015  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
from enum import Enum
from pyclick.click_models.ClickModel import ClickModel
from pyclick.click_models.Inference import EMInference
from pyclick.click_models.Param import ParamEM
from pyclick.click_models.ParamContainer import QueryDocumentParamContainer, RankParamContainer

__author__ = 'Luka Stout, Ilya Markov, Aleksandr Chuklin'


class PBM(ClickModel):
    """
    The position-based click model (PBM).

    PBM contains the set of attractiveness and examination parameters.
    An attractiveness parameter depends on a query and a document.
    """

    param_names = Enum('PBMParamNames', 'attr exam')
    """The names of the PBM parameters."""

    def __init__(self):
        self.params = {self.param_names.attr: QueryDocumentParamContainer(PBMAttrEM),
                       self.param_names.exam: RankParamContainer.default(PBMExamEM)}
        self._inference = EMInference()

    def get_conditional_click_probs(self, search_session):
        click_probs = self.get_full_click_probs(search_session)

        for rank, result in enumerate(search_session.web_results):
            if not result.click:
                click_probs[rank] = 1 - click_probs[rank]

        return click_probs

    def get_full_click_probs(self, search_session):
        session_params = self.get_session_params(search_session)
        click_probs = []

        for session_param in session_params:
            attr = session_param[self.param_names.attr].value()
            exam = session_param[self.param_names.exam].value()

            click_prob = attr * exam
            click_probs.append(click_prob)

        return click_probs

    def predict_relevance(self, query, search_result):
        return self.params[self.param_names.attr].get(query, search_result).value()


class PBMAttrEM(ParamEM):
    """
    The attractiveness parameter of the PBM model.
    The value of the parameter is inferred using the EM algorithm.
    """
    def update(self, search_session, rank, session_params):
        attr = session_params[rank][PBM.param_names.attr].value()
        exam = session_params[rank][PBM.param_names.exam].value()

        if search_session.web_results[rank].click:
            self._numerator += 1
        else:
            self._numerator += (1 - exam) * attr / (1 - exam * attr)

        self._denominator += 1


class PBMExamEM(ParamEM):
    """
    The examination parameter of the PBM model
    """
    def update(self, search_session, rank, session_params):
        attr = session_params[rank][PBM.param_names.attr].value()
        exam = session_params[rank][PBM.param_names.exam].value()

        if search_session.web_results[rank].click:
            self._numerator += 1
        else:
            self._numerator += (1 - attr) * exam / (1 - exam * attr)

        self._denominator += 1

