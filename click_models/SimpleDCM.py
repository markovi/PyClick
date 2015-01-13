#
# Copyright (C) 2014  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
import sys
import math

from click_models.ClickModel import ClickModel, ClickModelParam, PRETTY_LOG, ClickModelParamWrapper, RelevanceWrapper, \
    RelevanceWrapperRel
from click_models.Constants import MAX_DOCS_PER_QUERY


__author__ = 'Ilya Markov'


class SimpleDCM(ClickModel):
    """
        Implements the DCM model,
        where the relevance and gamma parameters
        are estimated using simple formulas from the original paper.

        Efficient multiple-click click_models in web search.
        Guo, Fan and Liu, Chao and Wang, Yi Min
        Proceedings of WSDM, 2009, pages 124-131.
    """
    def train(self, sessions):
        if len(sessions) <= 0:
            print >>sys.stderr, "The number of training sessions is zero."
            return

        self.params = self.init_params(self.init_param_values)

        for session in sessions:
            for rank, click in enumerate(session.get_clicks()):
                params = self.get_params(self.params, session, rank)

                for param in params.values():
                    param.update_value(None, click,
                                       rank=rank, clicks=session.get_clicks(),
                                       last_click_rank=session.get_last_click_rank())

        if not PRETTY_LOG:
            print >>sys.stderr, 'LL: %.10f' % self.get_loglikelihood(sessions)

    def init_params(self, init_param_values):
        params = {
            SimpleDCMRelevance.NAME: RelevanceWrapper(init_param_values, SimpleDCMRelevance.default()),
            SimpleDCMLambda.NAME: DCMLambdaWrapper(init_param_values, SimpleDCMLambda.default())
        }

        return params

    def predict_click_probs(self, session):
        click_probs = []
        exam_full = 1

        for rank, result in enumerate(session.web_results):
            params = self.get_params(self.params, session, rank)
            param_values = self.get_param_values(params)

            rel_value = param_values[SimpleDCMRelevance.NAME]
            lambda_value = param_values[SimpleDCMLambda.NAME]

            click_prob = rel_value * exam_full
            exam_full *= 1 - rel_value + rel_value * lambda_value

            click_probs.append(click_prob)

        return click_probs

    def get_log_click_probs(self, session):
        """
            Note that click probabilities for all ranks are calculated,
            even though parameters (relevance and lambda) for some of them
            might not be estimated!
        """
        log_click_probs = []
        exam_full = 1

        for rank, click in enumerate(session.get_clicks()):
            params = self.get_params(self.params, session, rank)
            param_values = self.get_param_values(params)
            rel_value = param_values[SimpleDCMRelevance.NAME]
            lambda_value = param_values[SimpleDCMLambda.NAME]

            if click:
                log_click_prob = math.log(rel_value * exam_full)
                exam_full *= lambda_value
            else:
                log_click_prob = math.log(1 - rel_value * exam_full)

            log_click_probs.append(log_click_prob)

        return log_click_probs

    @staticmethod
    def get_prior_values():
        return {SimpleDCMRelevance.NAME: 0.5,
                SimpleDCMLambda.NAME: 0.5}


class SimpleDCMRel(SimpleDCM):
    """
        Uses relevance judgements.
    """
    def init_params(self, init_param_values):
        params = super(SimpleDCMRel, self).init_params(init_param_values)
        params[SimpleDCMRelevance.NAME] = RelevanceWrapperRel(init_param_values, SimpleDCMRelevance.default())
        return params


class SimpleDCMRelevance(ClickModelParam):
    """
        The relevance parameter of the DCM model.
    """
    NAME = "rel"

    def update_value(self, param_values, click, **kwargs):
        last_click_rank = kwargs['last_click_rank']
        rank = kwargs['rank']

        if rank <= last_click_rank:
            if click == 1:
                self.numerator += 1

            self.denominator += 1


class SimpleDCMLambda(ClickModelParam):
    """
        The probability that a user will continue examining a SERP after a click.
    """
    NAME = "lambda"

    def update_value(self, param_values, click, **kwargs):
        last_click_rank = kwargs['last_click_rank']
        rank = kwargs['rank']

        if click and rank <= last_click_rank:
            if rank != last_click_rank:
                self.numerator += 1

            self.denominator += 1

        # Special case: when no click is observed in a session,
        # last clicked rank = last rank, lambda_{last rank} = 0
        clicks = kwargs['clicks']
        if not any(clicks) and rank == last_click_rank:
            self.denominator += 1


class DCMLambdaWrapper(ClickModelParamWrapper):
    def init_param_rule(self, init_param_func, **kwargs):
        """
            lambda: rank -> lambda
        """
        return [init_param_func(**kwargs) for r in xrange(MAX_DOCS_PER_QUERY)]

    def get_param(self, session, rank, **kwargs):
        return self.params[rank]

    def get_params_from_JSON(self, json_str):
        self.init_param()

        for r in xrange(MAX_DOCS_PER_QUERY):
            self.params[r].__dict__ = json_str[r]

    def __str__(self):
        param_str = "%s\n" % self.name

        for r in xrange(MAX_DOCS_PER_QUERY):
            param_str += "%d %r\n" % (r, self.params[r])

        return param_str

