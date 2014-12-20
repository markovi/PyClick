#
# Copyright (C) 2014  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
import sys

from click_models.ClickModel import ClickModel, ClickModelParam, ClickModelParamWrapper, RelevanceWrapper, \
    RelevanceWrapperRel
from click_models.InputReader import *
from click_models.UBM import UBMExaminationWrapper, UBMRelevance, UBMParam, UBM, UBMRelevanceWrapper, UBMRelevanceWrapperRel
from click_models.VerticalAwareClickModel import VertAttract, VAExaminationNoAttract, VertAttractWrapper, VAExaminationAttract, \
    VerticalAwareParamHelper


__author__ = 'Ilya Markov'

class VCM(UBM):
    """
        Implements the Vertical Click Model (VCM).

        Incorporating vertical results into search click click_models.
        Wang, Chao and Liu, Yiqun and Zhang, Min and Ma, Shaoping and Zheng, Meihong and Qian, Jing and Zhang, Kuo.
        Proceedings of SIGIR, 2013, pages 503-512.
    """
    EXAM_ATTRACT_BACKWARD = "exam_attract_backward"

    def init_params(self, init_param_values):
        self.param_helper = VCMParamHelper()
        params = {
            UBMRelevance.NAME: UBMRelevanceWrapper(init_param_values, UBMRelevance.default(),
                                                param_helper = self.param_helper),
            VAExaminationNoAttract.NAME: UBMExaminationWrapper(init_param_values, VAExaminationNoAttract.default(),
                                                             param_helper = self.param_helper),
            VAExaminationAttract.NAME: VCMExaminationWrapper(init_param_values, VAExaminationAttract.default(),
                                                           param_helper = self.param_helper),
            VertAttract.NAME: VertAttractWrapper(init_param_values, VertAttract.default(),
                                                   param_helper = self.param_helper),
            VCMSigma.NAME: VertAttractWrapper(init_param_values, VCMSigma.default(),
                                                param_helper = self.param_helper)
        }

        return params

    def get_params(self, base_params, session, rank, **kwargs):
        params = super(VCM, self).get_params(base_params, session, rank, **kwargs)
        params[self.EXAM_ATTRACT_BACKWARD] = base_params[VAExaminationAttract.NAME].get_param_backward(session, rank, **kwargs)
        # params_new[self.EXAM_ATTRACT_BACKWARD] = self.params_new[ExaminationAttract.NAME].get_param_backward(session, rank)

        return params

    def predict_click_probs(self, session):
        click_matrix = self.get_click_matrix(session, VAExaminationNoAttract.NAME)
        click_probs_noattr = self.get_click_probs_from_matrix(session, click_matrix,
                                                              VAExaminationNoAttract.NAME,
                                                              lambda session, rank: rank)

        click_matrix = self.get_click_matrix_attract(session, VAExaminationAttract.NAME)
        click_probs_attr = self.get_click_probs_from_matrix(session, click_matrix,
                                                            VAExaminationAttract.NAME,
                                                            VCMExaminationWrapper.get_orig_rank_forward)

        click_matrix = self.get_click_matrix_attract_reverse(session)
        click_probs_attr_reverse = self.get_click_probs_from_matrix(session, click_matrix,
                                                                    self.EXAM_ATTRACT_BACKWARD,
                                                                    VCMExaminationWrapper.get_orig_rank_backward)

        click_probs = []

        for rank, url in enumerate(session.urls):
            params = self.get_params(self.params, session, rank)
            param_values = self.get_param_values(params)

            vert_attr = param_values[VertAttract.NAME]
            sigma = param_values[VCMSigma.NAME]

            p_click = (1 - vert_attr) * click_probs_noattr[rank] + \
                vert_attr * (1 - sigma) * click_probs_attr[rank] + \
                vert_attr * sigma * click_probs_attr_reverse[rank]

            click_probs.append(p_click)

        return click_probs

    # def get_click_probs_from_matrix_attract(self, session, click_matrix, exam_type, rank_transformation):
    #     click_probs = [0.0] * len(session.urls)
    #
    #     for rank, url in enumerate(session.urls):
    #         # rank_actual = vert_pos - rank - 1 if rank < vert_pos else rank
    #         rank_actual = rank_transformation(session, rank)
    #
    #         # No click before rank
    #         click_probs[rank_actual] = self.get_p_click_at_dist(session, rank_actual, 0, exam_type) * \
    #                   (click_matrix[rank][rank - 1] if rank > 0 else 1)
    #
    #         # Iterate over clicks before rank
    #         for j in xrange(rank):
    #             # rank_actual_clicked = vert_pos - j - 1 if j < vert_pos else j
    #             rank_actual_clicked = rank_transformation(session, j)
    #
    #             click_probs[rank_actual] += self.get_p_click_at_dist(session, rank_actual, rank - j, exam_type) * \
    #                        (click_matrix[j][rank - 1] if rank - j > 1 else 1) * \
    #                        click_probs[rank_actual_clicked]
    #
    #     return click_probs

    def get_click_matrix_attract(self, session, exam_type):
        vert_pos = session.vert_pos
        exam_type = VAExaminationAttract.NAME

        click_matrix = [[1 for j in xrange(len(session.urls))]
                         for i in xrange(len(session.urls))]

        # From the vertical forward
        for i in xrange(vert_pos, len(session.urls)):
            cri = VCMExaminationWrapper.get_attract_rank_forward(session, i)

            # From vert_pos to i
            for j in xrange(vert_pos, i):
                ri = VCMExaminationWrapper.get_attract_rank_forward(session, j)
                click_matrix[cri][ri] = (click_matrix[cri][ri - 1] if ri > 0 else 1) * \
                                        (1 - self.get_p_click_at_dist(session, j, 0, exam_type))

            # At rank i
            click_matrix[cri][cri] = self.get_p_click_at_dist(session, i, 0, exam_type)

            # From i to bottom + from top to vert_pos
            for j in range(i + 1, len(session.urls)) + range(0, vert_pos):
                ri = VCMExaminationWrapper.get_attract_rank_forward(session, j)
                click_matrix[cri][ri] = (click_matrix[cri][ri - 1] if ri > cri + 1 else 1) * \
                                        (1 - self.get_p_click_at_dist(session, j, ri - cri, exam_type))

            # From top to vert_pos
            # for j in xrange(0, vert_pos):
            #     ri = VCMExaminationWrapper.get_attract_rank_forward(session, j)
            #     click_matrix[cri][ri] = (click_matrix[cri][ri - 1] if ri > cri + 1 else 1) * \
            #                          (1 - self.get_p_click_at_dist(session, j, ri - cri, exam_type))

        # From top to the vertical
        for i in xrange(0, vert_pos):
            cri = VCMExaminationWrapper.get_attract_rank_forward(session, i)

            # From vert_pos to bottom + from top to i
            for j in range(vert_pos, len(session.urls)) + range(0, i):
                ri = VCMExaminationWrapper.get_attract_rank_forward(session, j)
                click_matrix[cri][ri] = (click_matrix[cri][ri - 1] if ri > 0 else 1) * \
                                        (1 - self.get_p_click_at_dist(session, j, 0, exam_type))

            # From top to i
            # for j in xrange(0, i):
            #     ri = VCMExaminationWrapper.get_attract_rank_forward(session, j)
            #     click_matrix[cri][ri] = (click_matrix[cri][ri - 1] if ri > 0 else 1) * \
            #                          (1 - self.get_p_click_at_dist(session, j, 0, exam_type))

            # At rank i
            click_matrix[cri][cri] = self.get_p_click_at_dist(session, i, 0, exam_type)

            # From i to vert_pos
            for j in xrange(i + 1, vert_pos):
                ri = VCMExaminationWrapper.get_attract_rank_forward(session, j)
                click_matrix[cri][ri] = (click_matrix[cri][ri - 1] if ri > cri + 1 else 1) * \
                                        (1 - self.get_p_click_at_dist(session, j, ri - cri, exam_type))

        return click_matrix

    def get_click_matrix_attract_reverse(self, session):
        vert_pos = session.vert_pos
        exam_type = VCM.EXAM_ATTRACT_BACKWARD

        click_matrix = self.get_click_matrix(session, exam_type)

        # From the vertical backward
        for i in xrange(vert_pos - 1, -1, -1):
            cri = vert_pos - i - 1

            # From vert_pos back to i
            for j in xrange(vert_pos - 1, i, -1):
                ri = vert_pos - j - 1
                click_matrix[cri][ri] = (click_matrix[cri][ri - 1] if ri > 0 else 1) * \
                                     (1 - self.get_p_click_at_dist(session, j, 0, exam_type))

            # At rank i
            click_matrix[cri][cri] = self.get_p_click_at_dist(session, i, 0, exam_type)

            # From i back to 0
            for j in xrange(i - 1, -1, -1):
                ri = vert_pos - j - 1
                click_matrix[cri][ri] = (click_matrix[cri][ri - 1] if ri > cri + 1 else 1) * \
                                     (1 - self.get_p_click_at_dist(session, j, i - j, exam_type))
            # Continue at vert_post
            # click_matrix[cri][vert_pos] = click_matrix[cri][vert_pos - 1] * \
            #                             (1 - self.get_p_click_at_dist(session, vert_pos, i + 1, exam_type))
            # Continue from vert_pos til the end of list
            for j in xrange(vert_pos, len(session.urls)):
                click_matrix[cri][j] = click_matrix[cri][j - 1] * \
                                     (1 - self.get_p_click_at_dist(session, j, j - cri, exam_type))

        # From the vertical forward
        for i in xrange(vert_pos, len(session.urls)):
            # No clicks from vert_post back to 0
            for j in xrange(vert_pos - 1, -1, -1):
                ri = vert_pos - j - 1
                click_matrix[i][ri] = (click_matrix[i][ri - 1] if ri > 0 else 1) * \
                                     (1 - self.get_p_click_at_dist(session, j, 0, exam_type))

        return click_matrix

    @staticmethod
    def get_prior_values():
        return {UBMRelevance.NAME: 0.5,
                VAExaminationNoAttract.NAME: 0.5,
                VAExaminationAttract.NAME: 0.5,
                VertAttract.NAME: 0.5,
                VCMSigma.NAME: 0.5}


class VCMRel(VCM):
    """
        Uses relevance judgements.
    """
    def init_params(self, init_param_values):
        params = super(VCMRel, self).init_params(init_param_values)
        params[UBMRelevance.NAME] = UBMRelevanceWrapperRel(init_param_values, UBMRelevance.default(),
                                                       param_helper = self.param_helper)

        return params


class VCMParamHelper(VerticalAwareParamHelper):
    def get_attract_examination(self, param_values):
        exam_attr_forward = param_values[VAExaminationAttract.NAME]
        exam_attr_backward = param_values[VCM.EXAM_ATTRACT_BACKWARD]

        sigma = param_values[VCMSigma.NAME]

        exam_attr_full = (1 - sigma) * exam_attr_forward + sigma * exam_attr_backward
        return exam_attr_full

    def get_full_examination(self, param_values):
        vert_attr = param_values[VertAttract.NAME]

        exam = param_values[VAExaminationNoAttract.NAME]
        exam_attr = self.get_attract_examination(param_values)

        exam_full = (1 - vert_attr) * exam + vert_attr * exam_attr
        return exam_full


class VCMSigma(UBMParam):
    """
        Vertical attractiveness: vert_attract = P(F = 1).
    """
    NAME = "sigma"

    def update_value(self, param_values, click):
        rel = param_values[UBMRelevance.NAME]

        exam_full = self.param_helper.get_full_examination(param_values)
        exam_attr_full = self.param_helper.get_attract_examination(param_values)
        exam_attr_backward = param_values[VCM.EXAM_ATTRACT_BACKWARD]

        vert_attr = param_values[VertAttract.NAME]
        sigma = param_values[self.NAME]

        if click == 1:
            self.numerator += vert_attr * sigma * exam_attr_backward / exam_full
            self.denominator += vert_attr * exam_attr_full / exam_full
        else:
            self.numerator += vert_attr * sigma * (1 - rel * exam_attr_backward)
            self.denominator += vert_attr * (1 - rel * exam_attr_full)


class VCMExaminationWrapper(UBMExaminationWrapper):
    def get_param(self, session, rank, **kwargs):
        distance = self.get_distance(session, rank, **kwargs)
        rank_attract_forward = self.get_attract_rank_forward(session, rank)
        return self.params[rank_attract_forward][distance]

    def get_param_backward(self, session, rank, **kwargs):
        distance = self.get_distance_backward(session, rank, **kwargs)
        rank_attract_backward = self.get_attract_rank_backward(session, rank)
        return self.params[rank_attract_backward][distance]

    @staticmethod
    def get_distance(session, rank, **kwargs):
        dist_str = 'distance'
        vert_pos = session.vert_pos
        session_len = len(session.urls)

        if dist_str in kwargs:
            distance = kwargs[dist_str]
        else:
            if rank >= vert_pos:
                prev_click_ranks = [r for r, click in enumerate(session.clicks)
                                    if vert_pos <= r < rank and click]
                distance = rank - prev_click_ranks[-1] if len(prev_click_ranks) else 0
            else:
                prev_click_ranks_from_vertical = [r for r, click in enumerate(session.clicks)
                                                  if vert_pos <= r < session_len and click]
                prev_click_ranks_from_top = [r for r, click in enumerate(session.clicks)
                                             if 0 <= r < rank and click]

                distance = rank - prev_click_ranks_from_top[-1] if len(prev_click_ranks_from_top) \
                    else rank + (session_len - prev_click_ranks_from_vertical[-1]) if len(prev_click_ranks_from_vertical) \
                    else 0

        return distance

    @staticmethod
    def get_distance_backward(session, rank, **kwargs):
        """
            Returns the distance from a last clicked document,
            starting from a vertical result backward.
        """
        dist_str = 'distance'
        vert_pos = session.vert_pos

        if dist_str in kwargs:
            distance = kwargs[dist_str]
        else:
            if rank < vert_pos:
                prev_click_ranks = [r for r, click in enumerate(session.clicks)
                                    if rank < r < vert_pos and click]
                distance = prev_click_ranks[0] - rank if len(prev_click_ranks) else 0
            else:
                prev_click_ranks_backward = [r for r, click in enumerate(session.clicks)
                                             if 0 <= r < vert_pos and click]
                prev_click_ranks_forward = [r for r, click in enumerate(session.clicks)
                                            if vert_pos <= r < rank and click]

                distance = rank - prev_click_ranks_forward[-1] if len(prev_click_ranks_forward) \
                    else (rank - vert_pos) + (prev_click_ranks_backward[0] + 1) if len(prev_click_ranks_backward) \
                    else 0

        return distance

    @staticmethod
    def get_orig_rank_forward(session, rank_attr_forward):
        vert_pos = session.vert_pos
        session_len = len(session.urls)

        return rank_attr_forward + vert_pos \
            if rank_attr_forward < session_len - vert_pos \
            else rank_attr_forward - (session_len - vert_pos)

    @staticmethod
    def get_orig_rank_backward(session, rank_attr_backward):
        vert_pos = session.vert_pos
        return vert_pos - rank_attr_backward - 1 if rank_attr_backward < vert_pos else rank_attr_backward

    @staticmethod
    def get_attract_rank_forward(session, rank_orig):
        vert_pos = session.vert_pos
        session_len = len(session.urls)

        return rank_orig + (session_len - vert_pos) \
            if rank_orig < vert_pos \
            else rank_orig - vert_pos

    @staticmethod
    def get_attract_rank_backward(session, rank_orig):
        vert_pos = session.vert_pos
        return vert_pos - rank_orig - 1 if rank_orig < vert_pos else rank_orig

