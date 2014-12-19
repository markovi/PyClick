#
# Copyright (C) 2014  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
from abc import abstractmethod

from click_models.ClickModel import ClickModelParamWrapper
from click_models.InputReader import MAX_DOCS_PER_QUERY
from click_models.UBM import UBMRelevance, UBMParamHelper, UBMParam


__author__ = 'markil'


class VerticalAwareParamHelper(UBMParamHelper):
    @abstractmethod
    def get_attract_examination(self, param_values):
        """
            Returns the examination probability when a vertical results is attractive.
        """
        pass


class VAExaminationNoAttract(UBMParam):
    """
        Examination probability when a vertical result is NOT attractive:
        exam = P(E = 1 | F = 0).
    """
    NAME = "exam"

    def update_value(self, param_values, click):
        rel = param_values[UBMRelevance.NAME]
        exam = param_values[self.NAME]
        exam_full = self.param_helper.get_full_examination(param_values)
        vert_attr = param_values[VertAttract.NAME]

        if click == 1:
            self.numerator += (1 - vert_attr) * exam / exam_full
            self.denominator += (1 - vert_attr) * exam / exam_full
        else:
            self.numerator += (1 - vert_attr) * exam * (1 - rel) / (1 - rel * exam_full)
            self.denominator += (1 - vert_attr) * (1 - rel * exam) / (1 - rel * exam_full)


class VAExaminationAttract(UBMParam):
    """
        Examination probability when a vertical result is attractive:
        exam = P(E = 1 | F = 1).
    """
    NAME = "exam_attract"

    def update_value(self, param_values, click):
        rel = param_values[UBMRelevance.NAME]
        exam_attr = self.param_helper.get_attract_examination(param_values)
        exam_full = self.param_helper.get_full_examination(param_values)
        vert_attr = param_values[VertAttract.NAME]

        if click == 1:
            self.numerator += vert_attr * exam_attr / exam_full
            self.denominator += vert_attr * exam_attr / exam_full
        else:
            self.numerator += vert_attr * exam_attr * (1 - rel) / (1 - rel * exam_full)
            self.denominator += vert_attr * (1 - rel * exam_attr) / (1 - rel * exam_full)


class VertAttract(UBMParam):
    """
        Vertical attractiveness: vert_attract = P(F = 1).
    """
    NAME = "vert_attract"

    def update_value(self, param_values, click):
        rel = param_values[UBMRelevance.NAME]
        exam_full = self.param_helper.get_full_examination(param_values)
        exam_attr = self.param_helper.get_attract_examination(param_values)
        vert_attr = param_values[self.NAME]

        if click == 1:
            self.numerator += vert_attr * exam_attr
            self.denominator += exam_full
        else:
            self.numerator += vert_attr * (1 - exam_attr * rel)
            self.denominator += 1 - exam_full * rel


class VertAttractWrapper(ClickModelParamWrapper):
    def init_param_rule(self, init_param_func, **kwargs):
        """
            vertical_attractiveness: vertical rank -> vertical attractiveness
        """
        return [init_param_func(**kwargs) for vr in xrange(MAX_DOCS_PER_QUERY)]

    def get_param(self, session, rank, **kwargs):
        vert_pos = session.vert_pos
        return self.params[vert_pos]

    def get_params_from_JSON(self, json_str):
        self.init_param()

        for vr in xrange(MAX_DOCS_PER_QUERY):
            param_helper_backup = self.params[vr].param_helper
            self.params[vr].__dict__ = json_str[vr]
            self.params[vr].param_helper = param_helper_backup

    def __str__(self):
        param_str = "%s\n" % self.name

        for indx, vert_attr in enumerate(self.params):
            param_str += '%d %r\n' % (indx, vert_attr)

        return param_str

