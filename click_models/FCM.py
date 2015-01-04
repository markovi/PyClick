#
# Copyright (C) 2014  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
from click_models.ClickModel import ClickModelParam, ClickModelParamWrapper
from click_models.Constants import MAX_DOCS_PER_QUERY
from click_models.UBM import UBMExaminationWrapper, UBMRelevance, UBM, UBMRelevanceWrapper, UBMRelevanceWrapperRel
from click_models.VerticalAwareClickModel import VAExaminationNoAttract, VertAttract, \
    VertAttractWrapper, VAExaminationAttract, VerticalAwareParamHelper


__author__ = 'Ilya Markov'


class FCM(UBM):
    """
        Implements the Joint Vertical Model (FCM).

        Beyond ten blue links: enabling user click modeling in federated web search.
        Chen, Danqi and Chen, Weizhu and Wang, Haixun and Chen, Zheng and Yang, Qiang.
        Proceedings of WSDM, 2012, pages 463-472.
    """

    def init_params(self, init_param_values):
        self.param_helper = FCMParamHelper()
        params = {
            UBMRelevance.NAME: UBMRelevanceWrapper(init_param_values, UBMRelevance.default(),
                                                param_helper = self.param_helper),
            VAExaminationNoAttract.NAME: UBMExaminationWrapper(init_param_values, VAExaminationNoAttract.default(),
                                                             param_helper = self.param_helper),
            VertAttract.NAME: VertAttractWrapper(init_param_values, VertAttract.default(),
                                                   param_helper = self.param_helper),
            FCMBeta.NAME: FCMBetaWrapper(init_param_values, FCMBeta.default(),
                                         param_helper = self.param_helper)
        }

        return params

    def get_p_click_at_dist(self, session, rank, distance, exam_type):
        params = self.get_params(self.params, session, rank, distance=distance)
        param_values = self.get_param_values(params)
        return self.get_p_click(param_values)

    def get_param_values(self, params):
        param_values = super(FCM, self).get_param_values(params)
        param_values[FCMBeta.NAME] = params[FCMBeta.NAME].get_value(exam = params[VAExaminationNoAttract.NAME])

        return param_values

    @staticmethod
    def get_prior_values():
        return {VertAttract.NAME: 0.5,
                VAExaminationNoAttract.NAME: 0.5,
                UBMRelevance.NAME: 0.5,
                FCMBeta.NAME: 0.5}


class FCMRel(FCM):
    """
        Uses relevance judgements.
    """
    def init_params(self, init_param_values):
        params = super(FCMRel, self).init_params(init_param_values)
        params[UBMRelevance.NAME] = UBMRelevanceWrapperRel(init_param_values, UBMRelevance.default(),
                                                       param_helper = self.param_helper)

        return params


class FCMParamHelper(VerticalAwareParamHelper):
    def get_full_examination(self, param_values):
        vert_attr = param_values[VertAttract.NAME]
        exam = param_values[VAExaminationNoAttract.NAME]
        beta = param_values[FCMBeta.NAME]

        exam_full = exam + (1 - exam) * beta * vert_attr
        return exam_full

    def get_attract_examination(self, param_values):
        exam = param_values[VAExaminationNoAttract.NAME]
        beta = param_values[FCMBeta.NAME]

        exam_attr = exam + (1 - exam) * beta
        return exam_attr


class FCMBeta(ClickModelParam):
    """
        The beta parameter of the FCM model.

        In order to estimate beta,
        the examination probability in case of an attractive vertical (exam_attr)
        is estimated first.

        Since exam_attr = exam + (1 - exam) * beta,
        the parameter beta can be calculated as follows:
        beta = (exam_attr - exam) / (1 - exam)
    """

    NAME = "beta"

    def __init__(self, init_value, **kwargs):
        super(FCMBeta, self).__init__(init_value)
        self.exam_attr = VAExaminationAttract(init_value, **kwargs)

    def update_value(self, param_values, click, **kwargs):
        self.exam_attr.update_value(param_values, click)

    def get_value(self, **kwargs):
        exam_attr_prob = self.exam_attr.get_value()

        if VAExaminationNoAttract.NAME not in kwargs:
            return exam_attr_prob

        exam = kwargs[VAExaminationNoAttract.NAME]
        exam_prob = exam.get_value()

        beta = (exam_attr_prob - exam_prob) / (1 - exam_prob)
        return beta

    def __str__(self):
        return self.exam_attr.__str__()


class FCMBetaWrapper(ClickModelParamWrapper):
    def init_param_rule(self, init_param_func, **kwargs):
        """
            beta: distance from a vertical [-10..+10] -> beta
        """
        return [init_param_func(**kwargs) for dist in xrange(2 * MAX_DOCS_PER_QUERY - 1)]

    def get_param(self, session, rank, **kwargs):
        vert_pos = session.vert_pos
        dist = (MAX_DOCS_PER_QUERY - 1) + (rank - vert_pos)
        return self.params[dist]

    def get_params_from_JSON(self, json_str):
        self.init_param()

        for dist in xrange(2 * MAX_DOCS_PER_QUERY - 1):
            exam_attr_backup = self.params[dist].exam_attr
            param_helper_backup = exam_attr_backup.param_helper

            self.params[dist].__dict__ = json_str[dist]
            exam_attr_backup.__dict__ = self.params[dist].exam_attr

            exam_attr_backup.param_helper = param_helper_backup
            self.params[dist].exam_attr = exam_attr_backup

    def __str__(self):
        param_str = "%s\n" % self.name

        for indx, beta in enumerate(self.params):
            param_str += '%d %r\n' % (indx - MAX_DOCS_PER_QUERY + 1, beta)

        return param_str

