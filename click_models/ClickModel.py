#
# Copyright (C) 2014  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
from collections import defaultdict
import math
import sys
from abc import abstractmethod
import json
from click_models.InputReader import MAX_DOCS_PER_QUERY, InputReader


__author__ = 'Ilya Markov'


PRETTY_LOG = True
MAX_ITERATIONS = 40


class ClickModel(object):
    """
        An abstract click model.
    """
    def __init__(self, init_param_values):
        """
            Initializes the model.
        """
        self.init_param_values = init_param_values
        self.params = self.init_params(self.init_param_values)


    def train(self, sessions):
        """
            Trains the model.
        """
        if len(sessions) <= 0:
            print >>sys.stderr, "The number of training sessions is zero."
            return

        self.params = self.init_params(self.init_param_values)

        for iteration_count in xrange(MAX_ITERATIONS):
            self.params = self.get_updated_params(sessions, self.init_params(self.get_prior_values()))

            if not PRETTY_LOG:
                print >>sys.stderr, 'Iteration: %d, LL: %.10f' % (iteration_count + 1, self.get_loglikelihood(sessions))


    def get_updated_params(self, sessions, priors):
        """
            Calculates new parameter values
            based on their current values
            and observed clicks.
        """
        updated_params = priors

        for session in sessions:
            for rank, click in enumerate(session.clicks):
                params = self.get_params(self.params, session, rank)
                param_values = self.get_param_values(params)

                current_params = self.get_params(updated_params, session, rank)
                self.update_param_values(current_params, param_values, session, rank)

        return updated_params


    def get_params(self, base_params, session, rank, **kwargs):
        """
            Returns the model's parameters for a given rank in a given session.

            Can be verriden by subclasses.
        """
        params = {}
        for param_name in base_params:
            params[param_name] = base_params[param_name].get_param(session, rank, **kwargs)
        return params


    def update_param_values(self, params, param_values, session, rank):
        """
            Updates the values of the given parameters
            based on their previous values and on the observed click.
        """
        for param in params.values():
            param.update_value(param_values, session.clicks[rank])


    def get_param_values(self, params):
        """
            Returns a list of values for a given list of parameters.
        """
        param_values = {}
        for param_name, param in params.items():
            param_values[param_name] = param.get_value()

        return param_values


    def get_loglikelihood(self, sessions):
        """
            Returns the log-likelihood of the current model for the given sessions.
        """
        loglikelihood = 0

        for session in sessions:
            log_click_probs = self.get_log_click_probs(session)
            loglikelihood += sum(log_click_probs) / len(log_click_probs)

        loglikelihood /= len(sessions)
        return loglikelihood


    def get_perplexity(self, sessions):
        """
        Returns the perplexity and position perplexities for given sessions.
        """
        perplexity_at_rank = [0.0] * MAX_DOCS_PER_QUERY

        for session in sessions:
            log_click_probs = self.get_log_click_probs(session)
            for rank, log_click_prob in enumerate(log_click_probs):
                perplexity_at_rank[rank] += math.log(math.exp(log_click_prob), 2)

        perplexity_at_rank = [2 ** (-x / len(sessions)) for x in perplexity_at_rank]
        perplexity = sum(perplexity_at_rank) / len(perplexity_at_rank)

        return perplexity, perplexity_at_rank


    def get_log_click_probs(self, session):
        """
            Returns the list of log-click probabilities for all URLs in the given session.
        """
        log_click_probs = []

        for rank, click in enumerate(session.clicks):
            params = self.get_params(self.params, session, rank)
            param_values = self.get_param_values(params)
            click_prob = self.get_p_click(param_values)

            if click == 1:
                log_click_probs.append(math.log(click_prob))
            else:
                log_click_probs.append(math.log(1 - click_prob))

        return log_click_probs


    def test(self, sessions):
        """
            Evaluates the prediciton power of the click model for given sessions.
            Returns the log-likelihood, perplexity and position perplexity.
        """
        log_likelihood = self.get_loglikelihood(sessions)
        perplexity, position_perplexities = self.get_perplexity(sessions)

        return log_likelihood, perplexity, position_perplexities


    def to_JSON(self):
        """
            Converts the model into JSON.
        """
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


    def from_JSON(self, json_str):
        """
            Converts JSON into a model.
        """
        params_backup = self.params
        self.__dict__ = json.loads(json_str)

        for param_name, param in params_backup.items():
            param.from_JSON(self.params[param_name])
            self.params[param_name] = param


    def __str__(self):
        params_str = ''
        for param_name, param in self.params.items():
            params_str += '%s\n' % str(param)
        return params_str


    def __repr__(self):
        return str(self)


    @abstractmethod
    def init_params(self, init_param_values):
        """
            Initializes the model's parameters based on the given initial values.
        """
        pass


    @abstractmethod
    def predict_click_probs(self, session):
        """
            Predicts click probabilities for a given session.
        """
        pass


    @abstractmethod
    def get_p_click(self, param_values):
        """
            Returns the click probability P(C = 1)
            for the given parameter values.
        """
        pass


    @staticmethod
    def get_prior_values():
        """
            Returns priors for the model's parameters.
        """
        return {}

    @staticmethod
    def get_last_click_rank(clicks):
        """
            Returns the rank of the last clicked document in a session.
        """
        click_ranks = [r for r, click in enumerate(clicks) if click]
        last_click_rank = click_ranks[-1] if len(click_ranks) else len(clicks) - 1
        return last_click_rank


class ClickModelParam(object):
    """
        A generic parameter of a click model.
    """
    def __init__(self, init_value, **kwargs):
        if init_value != 0:
            self.numerator = init_value
            self.denominator = 1
        else:
            self.numerator = 0
            self.denominator = 0

    @classmethod
    def default(cls, **kwargs):
        return cls(0, **kwargs)

    @classmethod
    def init(cls, init_value, **kwargs):
        return cls(init_value, **kwargs)

    @classmethod
    def copy(cls, source, **kwargs):
        instance = cls(0, **kwargs)
        instance.numerator = source.numerator
        instance.denominator = source.denominator
        return instance

    @abstractmethod
    def update_value(self, param_values, click, **kwargs):
        return

    def get_value(self, **kwargs):
        if self.denominator == 0:
            return float('NaN')

        return self.numerator / float(self.denominator)

    def __str__(self):
        return str(self.get_value())

    def __repr__(self):
        return str(self)


class FixedParam(ClickModelParam):
    """
        The parameter with a fixed value which is never updated.
    """
    def update_value(self, param_values, click):
        pass


class ClickModelParamWrapper(object):
    """
        A wrapper for a click model parameter.
        Manages all parameters of the same type together
        and provides methods for initilizing, getting and setting their values.
    """
    def __init__(self, init_param_values, param_class, **kwargs):
        self.name = param_class.NAME
        self.factory = param_class
        self.init_param_values = init_param_values

        self.init_param(**kwargs)

    def init_param(self, **kwargs):
        """
            General initialization of click_models' parameters.
        """
        if not self.init_param_values.has_key(self.name):
            print >>sys.stderr, "No initial value for %s." % self.factory.__class__.__name__
            return

        self.params = self.init_param_rule(self.init_param_function, **kwargs)

    def init_param_function(self, **kwargs):
        self.init_value = self.init_param_values[self.name]
        if self.init_value >= 0:
            param = self.factory.init(self.init_value, **kwargs)
        else:
            param = self.factory.default(**kwargs)

        return param

    def from_JSON(self, json_str):
        factory_backup = self.factory
        self.__dict__ = json_str

        factory_backup.__dict__ = self.factory
        self.factory = factory_backup

        self.get_params_from_JSON(self.params)

    def __str__(self):
        return "%s\n%r\n" % (self.name, self.params)

    def __repr__(self):
        return str(self)

    @abstractmethod
    def init_param_rule(self, init_param_func, **kwargs):
        """
            Defines the way of initializing parameters.
        """
        pass

    @abstractmethod
    def get_param(self, session, rank, **kwargs):
        """
            Returns the value of the parameter for a given session and rank.
        """
        pass

    @abstractmethod
    def get_params_from_JSON(self, json_str):
        pass


class RelevanceWrapper(ClickModelParamWrapper):
    def init_param_rule(self, init_param_func, **kwargs):
        """
            relevance: query -> url -> probability of relevance
        """
        return defaultdict(lambda: defaultdict(lambda: init_param_func(**kwargs)))

    def get_param(self, session, rank, **kwargs):
        query = session.query
        url = session.urls[rank]
        return self.params[query][url]

    def get_params_from_JSON(self, json_str):
        self.init_param()

        for query_id, query_params in json_str.items():
            for url_id, url_param in query_params.items():
                self.params[query_id][url_id].__dict__ = url_param

    def __str__(self):
        params_num = 0
        for query_id, query_params in self.params.items():
            params_num += len(query_params)

        param_str = "%s\n" % self.name
        param_str += "%d parameters\n" % params_num
        return param_str


class RelevanceWrapperRel(RelevanceWrapper):
    """
        Relevance judgments are used to determine the relevance grade
        of each query-document pair.
    """
    def init_param_rule(self, init_param_func, **kwargs):
        """
            relevance: relevance grade -> probability of relevance
        """
        param = {}
        for grade, prob in InputReader.RELEVANCE_WEB.items():
            self.init_param_values[self.name] = prob
            param[grade] = init_param_func(**kwargs)

        return param

    def get_param(self, session, rank, **kwargs):
        grade = session.relevances[rank]
        return self.params[grade]

    def get_params_from_JSON(self, json_str):
        self.init_param()

        for grade in InputReader.RELEVANCE_WEB:
            self.params[grade].__dict__ = json_str[str(grade)]

    def __str__(self):
        param_str = "%s\n" % self.name

        for key in sorted(self.params.keys()):
            param_str += "%f: %f\n" % (key, self.params[key].get_value())

        return param_str


