#
# Copyright (C) 2015  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
from enum import Enum

from pyclick.click_models.ClickModel import ClickModel
from pyclick.click_models.Inference import EMInference
from pyclick.click_models.Param import ParamEM, TaskBasedParamEM
from pyclick.click_models.ParamContainer import QueryDocumentParamContainer, RankParamContainer, SingleParamContainer

__author__ = 'Ilya Markov, Aleksandr Chuklin'


class TCM(ClickModel):
    """
    The task-centric click model.
    It takes entire search tasks of a user into account. Taken from:
    Zhang, Yuchen and Chen, Weizhu and Wang, Dong and Yang, Qiang
    User-click modeling for understanding and predicting search-behavior.
    2011 in Proceedings of the 17th ACM SIGKDD - KDD '11
    """

    param_names = Enum('TCMParamNames', 'attr exam match new fresh')
    """The names of the TCM parameters."""

    def __init__(self):
        self.params = {self.param_names.attr: QueryDocumentParamContainer(TCMAttrEM),
                       self.param_names.exam: RankParamContainer.default(TCMExamEM),
                       self.param_names.match: SingleParamContainer(TCMMatchEM),
                       self.param_names.new: SingleParamContainer(TCMNewEM),
                       self.param_names.fresh: SingleParamContainer(TCMFreshEM)}
        self._inference = EMInference()

    def get_session_params(self, search_session):
        session_params = []

        fresh = self.params[self.param_names.fresh].get()
        for rank, result in enumerate(search_session.web_results):
            attr = self.params[self.param_names.attr].get(search_session.query, result.id)
            exam = self.params[self.param_names.exam].get(rank)

            param_dict = {self.param_names.attr: attr,
                          self.param_names.exam: exam,
                          self.param_names.fresh: fresh}
            if rank == 0:
                # Attach the session-wide params to the first document for uniformity.
                match = self.params[self.param_names.match].get()
                new = self.params[self.param_names.new].get()
                param_dict.update({self.param_names.match: match, self.param_names.new: new})

            session_params.append(param_dict)

        return session_params

    def get_conditional_click_probs(self, search_session, result_history):
        """
        result_history is a list of boolean values for each document in the session,
        True corresponds to the cases where the document appeared before for the same task
        """
        click_probs = self.predict_click_probs(search_session, result_history)

        for rank, result in enumerate(search_session.web_results):
            if not result.click:
                click_probs[rank] = 1 - click_probs[rank]

        return click_probs

    def predict_click_probs(self, search_session, result_history):
        assert len(search_session.web_results) == len(result_history)

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


class TCMAttrEM(TaskBasedParamEM):
    """
    The attractiveness parameter of the TCM model.
    The value of the parameter is inferred using the EM algorithm.
    """
    def _update(self, search_session, rank, session_params, history, last_session):
        doc = search_session.web_results[rank]

        attr = session_params[rank][TCM.param_names.attr].value()
        exam = session_params[rank][TCM.param_names.exam].value()
        fresh = session_params[0][TCM.param_names.fresh].value() \
                if (doc.id in history) else 1.0
        match = session_params[0][TCM.param_names.match].value()

        if doc.click:
            self._numerator += 1
        else:
            self._numerator += ((1 - exam) * attr * fresh * match /
                                (1 - exam * attr * fresh * match))

        self._denominator += 1


class TCMExamEM(TaskBasedParamEM):
    """
    The examination parameter of the TCM model
    """
    def _update(self, search_session, rank, session_params, history, last_session):
        doc = search_session.web_results[rank]

        attr = session_params[rank][TCM.param_names.attr].value()
        exam = session_params[rank][TCM.param_names.exam].value()
        fresh = session_params[0][TCM.param_names.fresh].value() \
                if (doc.id in history) else 1.0
        match = session_params[0][TCM.param_names.match].value()

        if doc.click:
            self._numerator += 1
        else:
            self._numerator += ((1 - attr) * exam * fresh * match /
                                (1 - exam * attr * fresh * match))

        self._denominator += 1


class TCMMatchEM(TaskBasedParamEM):
    """
    Parameter controlling if the query matches user task or not.
    """
    @staticmethod
    def get_no_clicks_given_match_prob(search_session, session_params, history):
        """Compute probability of not having clicks in the session given the task match."""
        fresh_param = session_params[0][TCM.param_names.fresh].value()
        no_clicks_prob = 1.0
        for rank, d in enumerate(search_session.web_results):
            attr = session_params[rank][TCM.param_names.attr].value()
            exam = session_params[rank][TCM.param_names.exam].value()
            fresh = fresh_param if (d.id in history) else 1.0

            no_clicks_prob *= (1 - attr * exam * fresh)
        return no_clicks_prob

    @staticmethod
    def get_match_given_session_prob(search_session, session_params, history, last_session):
        if any(d.click for d in search_session.web_results) or last_session:
            return 1.0
        else:
            match = session_params[0][TCM.param_names.match].value()
            new = session_params[0][TCM.param_names.new].value()
            p = TCMMatchEM.get_no_clicks_given_match_prob(search_session, session_params,
                                                          history)
            # Here we simplify the computation assuming that next sessions' clicks
            # do not depend on the current session given the value of N parameter.
            return 1.0 / (1 + (1.0 / match - 1) / (p * new))


    def _update(self, search_session, rank, session_params, history, last_session):
        self._numerator += self.get_match_given_session_prob(search_session, session_params,
                                                             history, last_session)
        self._denominator += 1


class TCMNewEM(TaskBasedParamEM):
    """Parameter controlling if there should be further sessions"""
    def _update(self, search_session, rank, session_params, history, last_session):
        match_prob = TCMMatchEM.get_match_given_session_prob(search_session,
                                                             session_params,
                                                             history, last_session)
        if last_session:
            self._numerator += 0
        else:
            self._numerator += match_prob

        self._denominator += match_prob


class TCMFreshEM(TaskBasedParamEM):
    def _update(self, search_session, rank, session_params, history, last_session):
        doc = search_session.web_results[rank]
        if doc.id in history:
            attr = session_params[rank][TCM.param_names.attr].value()
            exam = session_params[rank][TCM.param_names.exam].value()
            fresh = session_params[0][TCM.param_names.fresh].value()
            match = session_params[0][TCM.param_names.match].value()

            if doc.click:
                self._numerator += 1
            else:
                self._numerator += ((1 - fresh) * exam * attr * match /
                                    (1 - exam * attr * fresh * match))

            self._denominator += 1
