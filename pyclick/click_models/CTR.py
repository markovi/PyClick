#
# Copyright (C) 2015  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
from abc import abstractmethod
from enum import Enum
from pyclick.click_models.ClickModel import ClickModel
from pyclick.click_models.Inference import MLEInference
from pyclick.click_models.Param import ParamMLE
from pyclick.click_models.ParamContainer import QueryDocumentParamContainer, RankParamContainer, SingleParamContainer

__author__ = 'Ilya Markov, Luka Stout'


class CTR(ClickModel):
    """
    A click model based on the click-through rate (CTR).

    CTR contains the set of attractiveness (or relevance, or CTR) parameters.
    """

    param_names = Enum('CTRParamNames', 'ctr')
    """The names of the CTR parameters."""

    def __init__(self):
        self.params = {self.param_names.ctr: self._init_ctr_params()}
        self._inference = MLEInference()

    def get_conditional_click_probs(self, search_session):
        click_probs = self.get_full_click_probs(search_session)

        for rank, result in enumerate(search_session.web_results):
            if not result.click:
                click_probs[rank] = 1 - click_probs[rank]

        return click_probs

    def get_full_click_probs(self, search_session):
        session_params = self.get_session_params(search_session)
        click_probs = [session_param[self.param_names.ctr].value() for session_param in session_params]
        return click_probs

    @abstractmethod
    def _init_ctr_params(self):
        """
        Initializes and returs a container of the CTR parameters.

        :returns: A container of the CTR parameters.
        """
        pass

    @abstractmethod
    def _get_ctr_param(self, search_session, rank):
        """
        Returns the CTR parameter that corresponds to the given search session and rank.

        :returns: The CTR parameter that corresponds to the given search session and rank.
        """
        pass


class DCTR(CTR):
    """
    The document-based CTR click model (DCTR),
    where each attractiveness parameter depends on a query and a search result.
    """

    def predict_relevance(self, query, search_result):
        return self.params[self.param_names.ctr].get(query, search_result).value()

    def _init_ctr_params(self):
        return QueryDocumentParamContainer(CTRParamMLE)

    def _get_ctr_param(self, search_session, rank):
        return self.params[self.param_names.ctr].get(search_session.query, search_session.web_results[rank].id)


class RCTR(CTR):
    """
    The rank-based CTR click model (DCTR),
    where each attractiveness parameter depends on rank.
    """

    def predict_relevance(self, query, search_result):
        """
        RCTR cannot predict relevance of a search result given a query,
        so always returns 0.
        """
        return 0

    def _init_ctr_params(self):
        return RankParamContainer.default(CTRParamMLE)

    def _get_ctr_param(self, search_session, rank):
        return self.params[self.param_names.ctr].get(rank)


class GCTR(CTR):
    """
    The global CTR click model (GCTR) also known as random click model (RCM).
    GCTR has only one single parameter, which defines whether a user clicks on a search result or not.
    """

    def predict_relevance(self, query, search_result):
        return self.params[self.param_names.ctr].get().value()

    def _init_ctr_params(self):
        return SingleParamContainer(CTRParamMLE)

    def _get_ctr_param(self, search_session, rank):
        return self.params[self.param_names.ctr].get()


class CTRParamMLE(ParamMLE):
    """
    The CTR parameter of the CTR model.
    The value of the parameter is inferred using the MLE algorithm.
    """

    def update(self, search_session, rank):
        self._numerator += search_session.web_results[rank].click
        self._denominator += 1

