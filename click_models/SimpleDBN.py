#
# Copyright (C) 2014  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
import sys
import math
from click_models.ClickModel import ClickModel, ClickModelParam, PRETTY_LOG, RelevanceWrapper, RelevanceWrapperRel

__author__ = 'Ilya Markov'


class SimpleDBN(ClickModel):
    """
        Implements the Dynamic Bayesian Network model (DBN),
        where the continuation parameter gamma is set to 1.
        MLE is used to estimate parameter values.

        A dynamic bayesian network click model for web search ranking.
        Chapelle, Olivier and Zhang, Ya.
        Proceedings of WWW, 2009, pages 1-10.
    """
    def init_params(self, init_param_values):
        params = {
            SimpleDBNAttract.NAME: RelevanceWrapper(init_param_values, SimpleDBNAttract.default()),
            SimpleDBNSatisfy.NAME: RelevanceWrapper(init_param_values, SimpleDBNSatisfy.default())
        }

        return params

    def train(self, sessions):
        if len(sessions) <= 0:
            print >>sys.stderr, "The number of training sessions is zero."
            return

        self.params = self.init_params(self.init_param_values)

        for session in sessions:
            last_click_rank = self.get_last_click_rank(session.clicks)

            for rank, click in enumerate(session.clicks):
                params = self.get_params(self.params, session, rank)

                for param in params.values():
                    param.update_value(None, click,
                                       rank=rank,
                                       last_click_rank=last_click_rank)

        if not PRETTY_LOG:
            print >>sys.stderr, 'LL: %.10f' % self.get_loglikelihood(sessions)

    def get_log_click_probs(self, session):
        """
            Note that click probabilities for all ranks are calculated,
            even though parameters (attractiveness and satisfaction)
            for some of them might not be estimated!
        """
        log_click_probs = []
        exam_full = 1

        for rank, click in enumerate(session.clicks):
            params = self.get_params(self.params, session, rank)
            param_values = self.get_param_values(params)
            attract = param_values[SimpleDBNAttract.NAME]
            satisfy = param_values[SimpleDBNSatisfy.NAME]

            if click:
                log_click_prob = math.log(attract * exam_full)
                exam_full *= (1 - satisfy)
            else:
                log_click_prob = math.log(1 - attract * exam_full)

            log_click_probs.append(log_click_prob)

        return log_click_probs

    def predict_click_probs(self, session):
        click_probs = []
        exam_full = 1

        for rank, url in enumerate(session.urls):
            params = self.get_params(self.params, session, rank)
            param_values = self.get_param_values(params)

            attract = param_values[SimpleDBNAttract.NAME]
            satisfy = param_values[SimpleDBNSatisfy.NAME]

            click_prob = attract * exam_full
            exam_full *= 1 - attract * satisfy

            click_probs.append(click_prob)

        return click_probs

    @staticmethod
    def get_prior_values():
        return {SimpleDBNAttract.NAME: 0.5,
                SimpleDBNSatisfy.NAME: 0.5}


class SimpleDBNRel(SimpleDBN):
    """
        Uses relevance judgements.
    """
    def init_params(self, init_param_values):
        params = {
            SimpleDBNAttract.NAME: RelevanceWrapperRel(init_param_values, SimpleDBNAttract.default()),
            SimpleDBNSatisfy.NAME: RelevanceWrapperRel(init_param_values, SimpleDBNSatisfy.default())
        }

        return params


class SimpleDBNAttract(ClickModelParam):
    """
        Attraction probability of the DBN model: attract = P(A = 1),
        estimated using MLE.
    """
    NAME = "attract"

    def update_value(self, param_values, click, **kwargs):
        last_click_rank = kwargs['last_click_rank']
        rank = kwargs['rank']

        if rank <= last_click_rank:
            if click == 1:
                self.numerator += 1

            self.denominator += 1


class SimpleDBNSatisfy(ClickModelParam):
    """
        Satisfaction probability of the DBN model: satisfy = P(S = 1 | C = 1)
    """
    NAME = "satisfy"

    def update_value(self, param_values, click, **kwargs):
        last_click_rank = kwargs['last_click_rank']
        rank = kwargs['rank']

        if click and rank <= last_click_rank:
            if rank == last_click_rank:
                self.numerator += 1

            self.denominator += 1


