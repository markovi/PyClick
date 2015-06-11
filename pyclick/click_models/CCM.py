#
# Copyright (C) 2015  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
from __future__ import division

from enum import Enum
from pyclick.click_models.ClickModel import ClickModel
from pyclick.click_models.Inference import EMInference
from pyclick.click_models.Param import ParamEM
from pyclick.click_models.ParamContainer import QueryDocumentParamContainer, SingleParamContainer

__author__ = 'Ilya Markov, Luka Stout'


class CCM(ClickModel):
    """
    The click chain model (CCM) according to the following paper:
    Guo, Fan and Liu, Chao and Kannan, Anitha and Minka, Tom and Taylor, Michael and Wang, Yi-Min and Faloutsos, Christos.
    Click chain model in web search.
    Proceedings of WWW, pages 11-20, 2009.

    CCM contains a set of attractiveness parameters,
    which depend on a query and a document.
    It also contains three examination parameters with no dependencies.
    """

    param_names = Enum('CCMParamNames','attr exam_noclick exam_click_nonrel exam_click_rel')
    """
    The names of the DBN parameters.

    :attr: the attractiveness parameter.
    Determines whether a user clicks on a search results after examining it.
    :exam_noclick: the examination parameter.
    Determines whether a user continues examining search results after not clicking on the current one.
    :exam_click_nonrel: the examination parameter.
    Determines whether a user continues examining search results after clicking on a non-relevant result.
    :exam_click_rel: the examination parameter.
    Determines whether a user continues examining search results after clicking on a relevant result.
    """

    def __init__(self):
        self.params = {self.param_names.attr: QueryDocumentParamContainer(CCMAttrEM),
                       self.param_names.exam_noclick: SingleParamContainer(CCMExamNoclickEM),
                       self.param_names.exam_click_nonrel: SingleParamContainer(CCMExamClickEM),
                       self.param_names.exam_click_rel: SingleParamContainer(CCMExamClickEM)}

        self._inference = EMInference()

    def get_session_params(self, search_session):
        session_params = []

        exam_probs = self._get_session_exam(search_session)
        click_afters = self._get_session_click_after(search_session)

        for rank, result in enumerate(search_session.web_results):
            param_index = {QueryDocumentParamContainer.QUERY_IND: search_session.query,
                           QueryDocumentParamContainer.DOC_IND: result.id}

            rel = self._get_param_at_index(self.param_names.attr, CCMAttrEM, param_index)
            tau_no_click = self._get_param_at_index(self.param_names.exam_noclick, CCMExamNoclickEM, param_index)
            tau_click = self._get_param_at_index(self.param_names.exam_click, CCMExamClickEM, param_index)
            

            param_dict = {self.param_names.attr: rel,
                    self.param_names.exam_noclick: tau_no_click,
                    self.param_names.exam_click: tau_click,
                    'exam' : exam_probs[rank],
                    'click_after' : click_afters[rank]}
            
            session_params.append(param_dict)

        return session_params

    # TODO: @Ilya: check the correctness, e.g., line 73
    # (seems like it should depend on the current click: click => rel, no click => 1-rel)
    def get_conditional_click_probs(self, search_session):
        session_params = self.get_session_params(search_session)
        click_probs = []
        actual_clicks = search_session.get_clicks()
        last_click = search_session.get_last_click_rank()
        for rank, result in enumerate(search_session.web_results):
            rel = session_params[rank][self.param_names.attr].value()
            if rank <= last_click:
                click_probs.append(rel)
            else:
                tau_click = session_params[rank][CCM.param_names.exam_click].value()
                rel_prev = session_params[rank-1][self.param_names.attr].value()
                tau_no_click = session_params[rank][CCM.param_names.exam_noclick].value()

                if actual_clicks[rank-1]:
                    click_probs.append( rel * tau_click * rel_prev )
                else:
                    rel_last_click = session_params[last_click][self.param_names.attr].value()
                    chance_since_last_click = tau_click * rel_last_click
                    for rank_2, result in list(enumerate(search_session.web_results))[last_click+1:rank]:
                        rel_2 = session_params[rank_2][self.param_names.attr].value()
                        chance_since_last_click *= tau_no_click * rel_2
                    click_probs.append( rel * chance_since_last_click )
        for rank, click in enumerate(search_session.get_clicks()):
            if not click:
                click_probs[rank] = 1 - click_probs[rank]
        return click_probs

    # TODO: derive the correct formula
    def predict_click_probs(self, search_session):
        session_params = self.get_session_params(search_session)
        click_probs = []
        for rank, result in enumerate(search_session.web_results):
            rel = session_params[rank][self.param_names.attr].value()
            exam = session_params[rank]['exam']
            click_prob = rel * exam
            click_probs.append(click_prob)
        return click_probs

    def _get_session_exam(self, search_session):
        """ Method to calculate Epsilon. """
        exam_probs = [1]

        for rank, result in list(enumerate(search_session.web_results[:-1])):
            param_index = {QueryDocumentParamContainer.QUERY_IND: search_session.query,
                            QueryDocumentParamContainer.DOC_IND: result.id}

            prev_exam = exam_probs[-1]

            rel = self._get_param_at_index(self.param_names.attr, CCMAttrEM, param_index).value()
            tau_click = self._get_param_at_index(self.param_names.exam_click, CCMExamClickEM, param_index).value()
            tau_no_click = self._get_param_at_index(self.param_names.exam_noclick, CCMExamNoclickEM, param_index).value()

            exam_probs.append( prev_exam * ( rel * tau_click + (1-rel) * tau_no_click ) )
        return exam_probs
    
    def _get_session_click_after(self, search_session):
        """ P(C_{>=r} = 1 | E_r = 1), see DBN implementation for more details """
        probs = [0]

        for rank, result in enumerate(search_session.web_results[::-1]):
            param_index = {QueryDocumentParamContainer.QUERY_IND: search_session.query,
                            QueryDocumentParamContainer.DOC_IND: result.id}

            prev_prob = probs[-1]
            rel = self._get_param_at_index(self.param_names.attr, CCMAttrEM, param_index).value()
            tau_no_click = self._get_param_at_index(self.param_names.exam_noclick, CCMExamNoclickEM, param_index).value()


            prob = rel + (1-rel) * tau_no_click * prev_prob 
            probs.append(prob)
        return probs[::-1]


    def predict_relevance(self, query, doc):
        param_index = {QueryDocumentParamContainer.QUERY_IND: query,
                  QueryDocumentParamContainer.DOC_IND: doc}

        rel = self._get_param_at_index(self.param_names.attr, CCMAttrEM, param_index).value()
        return rel


class CCMAttrEM(ParamEM):
    """
    The attractiveness parameter of the CCM model.
    The value of the parameter is inferred using the EM algorithm.
    """

    def __init__(self):
        self._numerator1 = 1
        self._denominator1 = 2

        self._numerator2 = 1
        self._denominator2 = 2

    def value(self):
        v = (self._numerator1 / (2 * self._denominator1)) + (self._numerator2 / (2 * self._denominator2))
        return v
    
    def update(self, search_session, rank, session_params):
        click = search_session.web_results[rank].click

        self._denominator1 += 1
        if click:
            self._numerator1 += 1

            self._denominator2 += 1
            if not search_session.click_after_rank(rank):
                rel = session_params[rank][CCM.param_names.attr].value()
                tau_click = session_params[rank][CCM.param_names.exam_click].value()
                click_after = session_params[rank]['click_after']

                denom = 1 - (tau_click  * (1-rel)) * click_after
                self._numerator2 += rel/denom

        elif not search_session.click_after_rank(rank):

            rel = session_params[rank][CCM.param_names.attr].value()
            exam = session_params[rank]['exam']
            click_after = session_params[rank]['click_after']

            num = (1 - exam) * rel
            denom = 1 - exam * click_after
            self._numerator1 += num/denom       


class CCMExamNoclickEM(ParamEM):
    
    def update(self, search_session, rank, session_params):
        if search_session.web_results[rank].click:
            return

        self._denominator += 1

        if search_session.click_after_rank(rank):
            self._numerator += 1
        else:
            tau_no_click = session_params[rank][CCM.param_names.exam_noclick].value()
            click_after = session_params[rank]['click_after']
            
            num = (1 - click_after) * tau_no_click
            denom = 1 - click_after * tau_no_click
            self._numerator += num/denom
        

class CCMExamClickEM(ParamEM):
    
    def update(self, search_session, rank, session_params):
        if not search_session.web_results[rank].click:
            return

        self._denominator += 1

        if search_session.click_after_rank(rank):
            self._numerator += 1
        else:
            tau_click = session_params[rank][CCM.param_names.exam_click].value()
            click_after = session_params[rank]['click_after']

            num = (1 - click_after) * tau_click
            denom = 1 - click_after * tau_click
            self._numerator += num/denom
