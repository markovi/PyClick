#
# Copyright (C) 2015  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
from __future__ import division

from enum import Enum
from pyclick.click_models.ClickModel import ClickModel
from pyclick.click_models.Inference import MLEInference
from pyclick.click_models.Param import ParamMLE
from pyclick.click_models.ParamContainer import QueryDocumentParamContainer

__author__ = 'Luka Stout, Ilya Markov'


class CM(ClickModel):
    """
    The cascade click model (CM) according to the following paper:
    Craswell, Nick and Zoeter, Onno and Taylor, Michael and Ramsey, Bill.
    An experimental comparison of click position-bias models.
    Proceedings of WSDM, pages 87-94, 2008.

    CM contains the set of attractiveness parameters,
    which depend on a query and a document.
    """

    PROB_MIN = 0.000001
    """The minimum probability for the cases, where the CM model cannot compute any probability."""

    param_names = Enum('CMParamNames', 'attr')
    """The names of the CM parameters."""

    def __init__(self):
        self.params = {self.param_names.attr: QueryDocumentParamContainer(CMAttrMLE)}
        self._inference = MLEInference()

    def get_conditional_click_probs(self, search_session):
        click_ranks = [rank for rank, click in enumerate(search_session.get_clicks()) if click]
        first_click_rank = click_ranks[0] if len(click_ranks) else len(search_session.web_results)
        click_probs = self.get_full_click_probs(search_session)

        for rank, result in enumerate(search_session.web_results):
            if rank <= first_click_rank:
                if not result.click:
                    click_probs[rank] = 1 - click_probs[rank]
            else:
                click_probs[rank] = self.PROB_MIN

        return click_probs

    def get_full_click_probs(self, search_session):
        session_params = self.get_session_params(search_session)
        click_probs = []

        exam = 1
        for rank, result in enumerate(search_session.web_results):
            attr = session_params[rank][self.param_names.attr].value()

            click_prob = attr * exam
            click_probs.append(click_prob)

            exam *= 1 - attr

        return click_probs

    def predict_relevance(self, query, search_result):
        return self.params[self.param_names.ctr].get(query, search_result).value()


class CMAttrMLE(ParamMLE):
    """
    The attractiveness parameter of the CM model.
    The value of the parameter is inferred using the MLE algorithm.
    """

    def update(self, search_session, rank):
        if not any(search_session.get_clicks()[:rank]):
            self._numerator += search_session.web_results[rank].click
            self._denominator += 1
