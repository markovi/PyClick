#
# Copyright (C) 2014  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
from click_models.ClickModel import ClickModelParam, ClickModelParamWrapper, ClickModel, RelevanceWrapper, \
    RelevanceWrapperRel
from click_models.Constants import MAX_DOCS_PER_QUERY


__author__ = 'Ilya Markov'


class UBM(ClickModel):
    """
        Implements the UBM model.

        A user browsing model to predict search engine click data from past observations.
        Dupret, Georges E. and Piwowarski, Benjamin
        Proceedings of SIGIR, 2008, pages 331-338.
    """
    def init_params(self, init_param_values):
        self.param_helper = UBMParamHelper()
        params = {
            UBMRelevance.NAME: UBMRelevanceWrapper(init_param_values, UBMRelevance.default(),
                                                   param_helper = self.param_helper),
            UBMExamination.NAME: UBMExaminationWrapper(init_param_values, UBMExamination.default())
        }

        return params

    def get_p_click(self, param_values):
        rel = param_values[UBMRelevance.NAME]
        return rel * self.param_helper.get_full_examination(param_values)

    def get_p_click_at_dist(self, session, rank, distance, exam_type):
        """
            Returns the click probability for a given session
            at a given rank and distance.
        """
        params = self.get_params(self.params, session, rank, distance=distance)
        param_values = self.get_param_values(params)

        rel = param_values[UBMRelevance.NAME]
        exam = param_values[exam_type]

        return rel * exam

    def predict_click_probs(self, session):
        click_matrix = self.get_click_matrix(session, UBMExamination.NAME)
        click_probs = self.get_click_probs_from_matrix(session, click_matrix, UBMExamination.NAME,
                                                       lambda session, rank: rank)
        return click_probs

    def get_click_probs_from_matrix(self, session, click_matrix, exam_type, rank_transformation):
        click_probs = [0.0] * len(session.urls)

        for rank, url in enumerate(session.urls):
            rank_actual = rank_transformation(session, rank)

            # No click before rank
            click_probs[rank_actual] = self.get_p_click_at_dist(session, rank_actual, 0, exam_type) * \
                                       (click_matrix[rank][rank - 1] if rank > 0 else 1)

            # Iterate over clicks before rank
            for j in xrange(rank):
                rank_actual_clicked = rank_transformation(session, j)
                click_probs[rank_actual] += self.get_p_click_at_dist(session, rank_actual, rank - j, exam_type) * \
                                                    (click_matrix[j][rank - 1] if rank - j > 1 else 1) * \
                                                    click_probs[rank_actual_clicked]

        return click_probs

    def get_click_matrix(self, session, exam_type):
        """
            Returns a matrix NxN, where N is the number of documents in a result list.
            Rows are for clicked ranks, columns are for ranks.
            The cell (i, j), where j < i, contains the following probability:
            P(C_0 = 0, ..., C_j = 0).
            For i = j it is P(C_j = 1 | C_0 = 0, ..., C_j = 0).
            For j > i it is P(C_i+1 = 0, ..., C_j = 0 | C_i = 1)
        """
        click_matrix = [[1 for j in xrange(len(session.urls))]
                         for i in xrange(len(session.urls))]

        for i in xrange(len(session.urls)):
            for j in xrange(0, i):
                click_matrix[i][j] = (click_matrix[i][j - 1] if j > 0 else 1) * \
                                     (1 - self.get_p_click_at_dist(session, j, 0, exam_type))

            click_matrix[i][i] = self.get_p_click_at_dist(session, i, 0, exam_type)

            for j in xrange(i + 1, len(session.urls)):
                click_matrix[i][j] = (click_matrix[i][j - 1] if j - i > 1 else 1) * \
                                     (1 - self.get_p_click_at_dist(session, j, j - i, exam_type))

        return click_matrix

    def from_JSON(self, json_str):
        param_helper_backup = self.param_helper
        super(UBM, self).from_JSON(json_str)
        self.param_helper = param_helper_backup

    @staticmethod
    def get_prior_values():
        return {UBMRelevance.NAME: 0.5,
                UBMExamination.NAME: 0.5}


class UBMRel(UBM):
    """
        Uses relevance judgements.
    """
    def init_params(self, init_param_values):
        params = super(UBMRel, self).init_params(init_param_values)
        params[UBMRelevance.NAME] = UBMRelevanceWrapperRel(init_param_values, UBMRelevance.default(),
                                                          param_helper = self.param_helper)

        return params


class UBMParamHelper(object):
    def get_full_examination(self, param_values):
        """
            Returns the full examination probability based on the given parameter values.
        """
        return param_values[UBMExamination.NAME]


class UBMParam(ClickModelParam):
    """
        Parameter of a general vertical-aware click model.
    """
    def __init__(self, init_value, **kwargs):
        super(UBMParam, self).__init__(init_value)
        if 'param_helper' in kwargs:
            self.param_helper = kwargs['param_helper']
        else:
            self.param_helper = UBMParamHelper()


class UBMRelevance(UBMParam):
    """
        Probability of relevance/attractiveness: rel = P(A = 1 | E = 1).
    """
    NAME = "rel"

    def update_value(self, param_values, click):
        rel = param_values[self.NAME]
        exam_full = self.param_helper.get_full_examination(param_values)

        if click == 1:
            self.numerator += 1
        else:
            self.numerator += rel * (1 - exam_full) / (1 - rel * exam_full)

        self.denominator += 1


class UBMExamination(ClickModelParam):
    """
        Examination probability: exam = P(E = 1).
    """
    NAME = "exam"

    def update_value(self, param_values, click):
        rel = param_values[UBMRelevance.NAME]
        exam = param_values[self.NAME]

        if click == 1:
            self.numerator += 1
        else:
            self.numerator += exam * (1 - rel) / (1 - exam * rel)

        self.denominator += 1


class UBMExaminationWrapper(ClickModelParamWrapper):
    def init_param_rule(self, init_param_func, **kwargs):
        """
            examination: rank -> distance from last click -> examination probability
        """
        return [[init_param_func(**kwargs) for d in xrange(MAX_DOCS_PER_QUERY)]
                for r in xrange(MAX_DOCS_PER_QUERY)]

    def get_param(self, session, rank, **kwargs):
        distance = self.get_distance(session, rank, **kwargs)
        return self.params[rank][distance]

    def get_params_from_JSON(self, json_str):
        self.init_param()

        for r in xrange(MAX_DOCS_PER_QUERY):
            for d in xrange(MAX_DOCS_PER_QUERY):
                self.params[r][d].__dict__ = json_str[r][d]

    def __str__(self):
        param_str = "%s\n" % self.name

        for r in xrange(MAX_DOCS_PER_QUERY):
            param_str += "%r\n" % self.params[r]

        return param_str

    @staticmethod
    def get_distance(session, rank, **kwargs):
        dist_str = 'distance'

        if dist_str in kwargs:
            distance = kwargs[dist_str]
        else:
            prev_click_ranks = [r for r, click in enumerate(session.get_clicks()) if r < rank and click]
            distance = rank - prev_click_ranks[-1] if len(prev_click_ranks) else 0

        return distance


class UBMRelevanceWrapper(RelevanceWrapper):
    def get_params_from_JSON(self, json_str):
        self.init_param()

        for query_id, query_params in json_str.items():
            for url_id, url_param in query_params.items():
                param_helper_backup = self.params[query_id][url_id].param_helper
                self.params[query_id][url_id].__dict__ = url_param
                self.params[query_id][url_id].param_helper = param_helper_backup


class UBMRelevanceWrapperRel(RelevanceWrapperRel):
    def get_params_from_JSON(self, json_str):
        self.init_param()

        #TODO: fix
        for grade in InputReader.RELEVANCE_WEB:
            param_helper_backup = self.params[grade].param_helper
            self.params[grade].__dict__ = json_str[str(grade)]
            self.params[grade].param_helper = param_helper_backup


