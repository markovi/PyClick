#
# Copyright (C) 2014  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
from abc import abstractmethod
import math
from click_models.ClickModel import ClickModel, ClickModelParam, ClickModelParamWrapper, FixedParam, RelevanceWrapper, \
    RelevanceWrapperRel
from click_models.SimpleDBN import SimpleDBNAttract, SimpleDBNSatisfy, SimpleDBN

__author__ = 'Ilya Markov'


class DBN(ClickModel):
    """
        Implements the Dynamic Bayesian Network model (DBN)
        with the full inference of parameters.

        A dynamic bayesian network click model for web search ranking.
        Chapelle, Olivier and Zhang, Ya.
        Proceedings of WWW, 2009, pages 1-10.
    """
    EXAMINATION = "exam"

    def init_params(self, init_param_values):
        params = {
            DBNAttract.NAME: RelevanceWrapper(init_param_values, DBNAttract.default()),
            DBNSatisfy.NAME: RelevanceWrapper(init_param_values, DBNSatisfy.default()),
            DBNGamma.NAME: DBNGammaWrapper(init_param_values, DBNGamma.default())
        }

        return params

    def get_updated_params(self, sessions, priors):
        updated_params = priors

        for session in sessions:
            params = self.get_session_params(self.params, session)
            param_values = self.get_param_values(params)

            current_params = self.get_session_params(updated_params, session)
            self.update_param_values(current_params, param_values, session)

        return updated_params

    def get_session_params(self, base_params, session):
        params = {param_name: [] for param_name in base_params}

        for rank, result in enumerate(session.web_results):
            rank_params = self.get_params(base_params, session, rank)
            for param_name in base_params:
                params[param_name].append(rank_params[param_name])

        return params

    def get_param_values(self, params):
        param_values = {param_name: [] for param_name in params}
        for param_name, param in params.items():
            for rank, param_value in enumerate(param):
                param_values[param_name].append(param[rank].get_value())

        return param_values

    def update_param_values(self, params, param_values, session):
        param_values[self.EXAMINATION] = self.get_observed_session_examination(param_values, session)

        for rank, result in enumerate(session.web_results):
            for param in params.values():
                param[rank].update_value(param_values, session, rank)

    def get_observed_session_examination(self, param_values, session):
        """
            Calculates the examination probability at each rank
            for a session with observed clicks.
        """
        session_exam = [1.0] * len(session.web_results)
        exam = 1

        for rank in xrange(len(session_exam)):
            session_exam[rank] = exam

            gamma = param_values[DBNGamma.NAME][rank]
            sat = param_values[DBNSatisfy.NAME][rank]
            if session.web_results[rank].click:
                exam *= gamma * (1 - sat)
            else:
                exam *= gamma

        return session_exam

    def get_predicted_session_examination(self, param_values, session):
        """
            Predicts the examination probability at each rank for a given session.
        """
        session_exam = [1.0] * len(session.web_results)
        exam = 1

        for rank in xrange(len(session_exam)):
            session_exam[rank] = exam

            gamma = param_values[DBNGamma.NAME][rank]
            sat = param_values[DBNSatisfy.NAME][rank]
            attr = param_values[DBNAttract.NAME][rank]
            exam *= gamma * (1 - attr * sat)

        return session_exam

    def get_log_click_probs(self, session):
        log_click_probs = []

        session_params = self.get_session_params(self.params, session)
        session_param_values = self.get_param_values(session_params)
        session_exam = self.get_observed_session_examination(session_param_values, session)

        for rank, click in enumerate(session.get_clicks()):
            exam = session_exam[rank]
            attr = session_param_values[DBNAttract.NAME][rank]

            if click:
                log_click_prob = math.log(exam * attr)
            else:
                log_click_prob = math.log(1 - exam * attr)

            log_click_probs.append(log_click_prob)

        return log_click_probs

    def predict_click_probs(self, session):
        click_probs = []

        session_params = self.get_session_params(self.params, session)
        session_param_values = self.get_param_values(session_params)
        session_exam = self.get_predicted_session_examination(session_param_values, session)

        for rank, results in enumerate(session.web_results):
            exam = session_exam[rank]
            attr = session_param_values[DBNAttract.NAME][rank]

            click_probs.append(exam * attr)

        return click_probs

    @staticmethod
    def get_prior_values():
        prior_values = SimpleDBN.get_prior_values()
        prior_values[DBNGamma.NAME] = 0.5
        return prior_values


class DBNRel(DBN):
    def init_params(self, init_param_values):
        params = {
            DBNAttract.NAME: RelevanceWrapperRel(init_param_values, DBNAttract.default()),
            DBNSatisfy.NAME: RelevanceWrapperRel(init_param_values, DBNSatisfy.default()),
            DBNGamma.NAME: DBNGammaWrapper(init_param_values, DBNGamma.default())
        }

        return params


class DBNParam(ClickModelParam):
    """
        General DBN parameter.
    """
    @abstractmethod
    def update_value(self, session_param_values, session, rank):
        pass

    def get_p_noclick_after_rank(self, session_param_values, rank):
        """
            Returns the probability of NO click after a given rank.
        """
        max_rank = len(session_param_values[DBNAttract.NAME]) - 1
        if rank == max_rank:
            return 1

        gamma = session_param_values[DBNGamma.NAME][rank]
        p_noclick = 1

        for r in range(max_rank, rank + 1, -1):
            attr = session_param_values[DBNAttract.NAME][r]
            p_noclick *= gamma * (1 - attr)
            p_noclick += 1 - gamma

        attr = session_param_values[DBNAttract.NAME][rank + 1]
        exam = session_param_values[DBN.EXAMINATION][rank + 1]
        p_noclick *= exam * (1 - attr)
        p_noclick += 1 - exam

        return p_noclick

    def get_p_click_after_rank(self, session_param_values, rank):
        """
            Returns the probability of observing a click after a given rank.
        """
        max_rank = len(session_param_values[DBNAttract.NAME]) - 1
        if rank == max_rank:
            return 0

        gamma = session_param_values[DBNGamma.NAME][rank]
        attr = session_param_values[DBNAttract.NAME][max_rank]

        p_click = attr
        for r in range(max_rank - 1, rank, -1):
            attr = session_param_values[DBNAttract.NAME][r]
            p_click *= gamma * (1 - attr)
            p_click += attr

        exam = session_param_values[DBN.EXAMINATION][rank + 1]
        p_click *= exam
        return p_click


class DBNAttract(DBNParam):
    """
        Attractiveness parameter of the DBN model: P(A = 1),
        i.e. pre-click relevance (based on a snippet).
    """
    NAME = SimpleDBNAttract.NAME

    def update_value(self, session_param_values, session, rank):
        attr = session_param_values[self.NAME][rank]
        exam = session_param_values[DBN.EXAMINATION][rank]

        if session.web_results[rank].click:
            self.numerator += 1
        else:
            self.numerator += attr * (1 - exam) / (1 - attr * exam)

        self.denominator += 1


class DBNSatisfy(DBNParam):
    """
        Satisfaction probability of the DBN model: P(S = 1 | C = 1),
        i.e. post-click relevance (based on examining a document).
    """
    NAME = SimpleDBNSatisfy.NAME

    def update_value(self, session_param_values, session, rank):
        if not session.web_results[rank].click:
            return

        attr = session_param_values[DBNAttract.NAME][rank]
        sat = session_param_values[self.NAME][rank]
        exam = session_param_values[DBN.EXAMINATION][rank]
        gamma = session_param_values[DBNGamma.NAME][rank]

        # p_noclick_after_rank = self.get_p_nonclick_after_rank(session_param_values, rank)
        # p_click_after_rank = self.get_p_click_after_rank(session_param_values, rank)

        exam_nosat = exam * (1 - attr * sat)

        p_click_after_rank = self.get_p_click_after_rank(session_param_values, rank)
        p_click_after_rank_given_nosat = p_click_after_rank / exam_nosat

        p_noclick_after_rank = self.get_p_noclick_after_rank(session_param_values, rank)
        p_noclick_after_rank_given_nosat = (p_noclick_after_rank - 1 + exam_nosat) / exam_nosat

        last_click_rank = session.get_last_click_rank()
        if rank < last_click_rank and any(session.get_clicks()):
            self.numerator += 0
            self.denominator += exam * attr * (1 - sat) * p_click_after_rank_given_nosat / p_click_after_rank
        else:
            self.numerator += exam * attr * sat / p_noclick_after_rank
            self.denominator += exam * attr * ((1 - sat) * p_noclick_after_rank_given_nosat + sat) / p_noclick_after_rank


class DBNGamma(DBNParam):
    """
        Gamma parameter of the DBN model: P(E_i+1 = 1 | E_i = 1, S_i = 0)
    """
    NAME = "gamma"

    def update_value(self, session_param_values, session, rank):
        max_rank = len(session.web_results) - 1
        if rank == max_rank:
            return

        attr = session_param_values[DBNAttract.NAME][rank]
        sat = session_param_values[DBNSatisfy.NAME][rank]
        exam = session_param_values[DBN.EXAMINATION][rank]
        gamma = session_param_values[self.NAME][rank]
        exam_next = session_param_values[DBN.EXAMINATION][rank + 1]

        p_click_after_rank = self.get_p_click_after_rank(session_param_values, rank)
        p_click_after_rank_given_exam = p_click_after_rank / exam_next

        p_noclick_after_rank = self.get_p_noclick_after_rank(session_param_values, rank)
        p_noclick_after_rank_given_exam = (p_noclick_after_rank - 1 + exam_next) / exam_next

        multiplier = exam * (1 - attr * sat)

        last_click_rank = session.get_last_click_rank()
        if rank < last_click_rank and any(session.get_clicks()):
            self.numerator += multiplier * gamma * p_click_after_rank_given_exam / p_click_after_rank
            self.denominator += multiplier * gamma * p_click_after_rank_given_exam / p_click_after_rank
        else:
            self.numerator += multiplier * gamma * p_noclick_after_rank_given_exam / p_noclick_after_rank
            self.denominator += multiplier * (gamma * p_noclick_after_rank_given_exam + 1 - gamma) / p_noclick_after_rank


class DBNGammaWrapper(ClickModelParamWrapper):
    def init_param_rule(self, init_param_func, **kwargs):
        """
            lambda (single parameter)
        """
        return init_param_func(**kwargs)

    def get_param(self, session, rank, **kwargs):
        return self.params

    def get_params_from_JSON(self, json_str):
        self.init_param()
        self.params.__dict__ = json_str

    def __str__(self):
        param_str = "%s\n" % self.name
        param_str += "%r\r" % self.params
        return param_str


