#
# Copyright (C) 2015  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
from pyclick.click_models.Inference import EMInference

__author__ = 'Ilya Markov'


class TaskCentricEMInference(EMInference):
    """
    The expectation-maximization (EM) approach to parameter inference
    for task-centric click models.
    """

    def infer_params(self, click_model, search_tasks):
        if search_tasks is None or len(search_tasks) == 0:
            return

        for iteration in range(self.iter_num):
            new_click_model = click_model.__class__()

            for search_task in search_tasks:
                for search_session in search_task.search_sessions:
                    current_session_params = click_model.get_session_params(search_session)
                    new_session_params = new_click_model.get_session_params(search_session)

                    for rank, result in enumerate(search_session.web_results):
                        for param_name, param in new_session_params[rank].items():
                            param.update(search_task, search_session, rank, current_session_params)

            click_model.params = new_click_model.params
