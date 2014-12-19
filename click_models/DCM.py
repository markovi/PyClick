#
# Copyright (C) 2014  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
import sys
from click_models.ClickModel import ClickModelParam, PRETTY_LOG, MAX_ITERATIONS, RelevanceWrapper, RelevanceWrapperRel
from click_models.InputReader import MAX_DOCS_PER_QUERY
from click_models.SimpleDCM import SimpleDCMLambda, SimpleDCMRelevance, SimpleDCM, DCMLambdaWrapper, SimpleDCMRel

__author__ = 'markil'


class DCM(SimpleDCM):
    """
        Implements the DCM click_models with a full parameter inference.

        Efficient multiple-click click_models in web search.
        Guo, Fan and Liu, Chao and Wang, Yi Min
        Proceedings of WSDM, 2009, pages 124-131.
    """
    ALL_NONREL_AFTER_LAST_CLICK = "all_nonrel"

    def init_params(self, init_param_values):
        params = {
            DCMRelevance.NAME: RelevanceWrapper(init_param_values, DCMRelevance.default()),
            DCMLambda.NAME: DCMLambdaWrapper(init_param_values, DCMLambda.default())
        }

        return params

    def copy_params(self, base_params):
        params = self.init_params(self.get_prior_values())

        # Copy relevance
        rel_params = base_params[DCMRelevance.NAME].params
        for query_id, query_param in rel_params.items():
            for url_id, url_param in query_param.items():
                params[DCMRelevance.NAME].params[query_id][url_id] = DCMRelevance.copy(url_param)

        # Copy lambda
        lambda_params = base_params[DCMLambda.NAME].params
        for r, lambda_param in enumerate(lambda_params):
            params[DCMLambda.NAME].params[r] = DCMLambda.copy(lambda_param)

        return params

    def get_params(self, base_params, session, rank, **kwargs):
        last_click_rank = self.get_last_click_rank(session.clicks)

        params = super(DCM, self).get_params(base_params, session, rank)
        if rank >= last_click_rank:
            params[DCMLambda.NAME] = base_params[DCMLambda.NAME].get_param(session, last_click_rank)
            params[self.ALL_NONREL_AFTER_LAST_CLICK] = DCMRelevance(self.get_nonrel_prob(base_params, session))

        return params

    def update_param_values(self, params, param_values, session, rank):
        last_click_rank = self.get_last_click_rank(session.clicks)

        for param in params.values():
            param.update_value(param_values, session.clicks[rank],
                               rank=rank, last_click_rank=last_click_rank)

    def get_simple_dcm(self):
        return SimpleDCM(SimpleDCM.get_prior_values())

    def train(self, sessions):
        simple_dcm = self.get_simple_dcm()
        simple_dcm.train(sessions)

        self.params = self.copy_params(simple_dcm.params)

        for iteration_count in xrange(MAX_ITERATIONS):
            self.params = self.get_updated_params(sessions, self.copy_params(simple_dcm.params))

            if not PRETTY_LOG:
                print >>sys.stderr, 'Iteration: %d, LL: %.10f' % (iteration_count + 1, self.get_loglikelihood(sessions))

    @staticmethod
    def get_nonrel_prob(params, session):
        """
            Returns the probability of all documents after the last clicked rank
            being nonrelevant.
        """
        last_click_rank = DCM.get_last_click_rank(session.clicks)

        nonrel_prob_full = 1
        for r in xrange(last_click_rank + 1, len(session.clicks)):
            nonrel_prob = params[DCMRelevance.NAME].get_param(session, r).get_value()
            nonrel_prob_full *= 1 - nonrel_prob

        return nonrel_prob_full


class DCMRel(DCM):
    """
        Uses relevance judgements.
    """
    def init_params(self, init_param_values):
        params = super(DCMRel, self).init_params(init_param_values)
        params[DCMRelevance.NAME] = RelevanceWrapperRel(init_param_values, DCMRelevance.default())
        return params

    def copy_params(self, base_params):
        params = self.init_params(self.get_prior_values())

        # Copy relevance
        rel_params = base_params[DCMRelevance.NAME].params

        for grade, rel_param in rel_params.items():
            params[DCMRelevance.NAME].params[grade] = DCMRelevance.copy(rel_param)

        # Copy lambda
        lambda_params = base_params[DCMLambda.NAME].params
        for r, lambda_param in enumerate(lambda_params):
            params[DCMLambda.NAME].params[r] = DCMLambda.copy(lambda_param)

        return params

    def get_simple_dcm(self):
        return SimpleDCMRel(SimpleDCM.get_prior_values())


class DCMLambda(ClickModelParam):
    NAME = SimpleDCMLambda.NAME

    def update_value(self, param_values, click, **kwargs):
        last_click_rank = kwargs['last_click_rank']
        rank = kwargs['rank']

        if rank == last_click_rank:
            nonrel_param = param_values[DCM.ALL_NONREL_AFTER_LAST_CLICK]
            lambda_param = param_values[self.NAME]

            self.numerator += lambda_param * nonrel_param
            self.denominator += 1 - lambda_param + lambda_param * nonrel_param


class DCMRelevance(ClickModelParam):
    NAME = SimpleDCMRelevance.NAME

    def update_value(self, param_values, click, **kwargs):
        last_click_rank = kwargs['last_click_rank']
        rank = kwargs['rank']

        if rank > last_click_rank:
            rel_param = param_values[self.NAME]
            lambda_param = param_values[DCMLambda.NAME]

            self.numerator += rel_param * (1 - lambda_param)
            self.denominator += 1 - rel_param * lambda_param

