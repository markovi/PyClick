#
# Copyright (C) 2015  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
from abc import abstractmethod
from collections import defaultdict

__author__ = 'Ilya Markov'


class Param(object):
    """An abstract parameter of a click model."""

    def __str__(self):
        if self.value():
            return '%.4f' % self.value()
        else:
            return 'None'

    def __repr__(self):
        return str(self)

    @abstractmethod
    def value(self):
        """
        Returns the current value of the parameter.

        :returns: The current value of the parameter.
        """
        pass

    @abstractmethod
    def update(self, search_session, rank):
        """
        Updates the value of the parameter based on the observed search session.
        Note that this parameter corresponds to a search result at the given rank.

        :param search_session: The observed search session.
        :param rank: The rank of a search result, which corresponds to the current parameter.
        """
        pass


class ParamMLE(Param):
    """A parameter used in the maximum likelihood estimation."""

    def __init__(self):
        self._numerator = 1
        self._denominator = 2

    def value(self):
        return self._numerator / float(self._denominator)

    @abstractmethod
    def update(self, search_session, rank):
        pass


class ParamEM(Param):
    """A parameter used in the expectation-maximization inference."""

    PROB_MIN = 0.000001  # probaiblity to use instead of 0 to protect from the math domain errors

    def __init__(self):
        self._numerator = 1
        self._denominator = 5

    def value(self):
        return min(self._numerator / float(self._denominator), 1 - self.PROB_MIN)

    @abstractmethod
    def update(self, search_session, rank, session_params):
        """
        Updates the value of the parameter based on the given search session
        and the values of other parameters.
        """
        pass

    def finish_iteration(self):
        """Called at the end of EM iteration after update was called for all sessions."""
        pass

class TaskBasedParamEM(ParamEM):
    """A parameter that uses sessions in the same task and is trained using EM algorithm."""
    def __init__(self):
        self.per_task_sessions = defaultdict(lambda: [])
        super(TaskBasedParamEM, self).__init__()

    def update(self, search_session, rank, session_params):
        self.per_task_sessions[search_session.task_id].append(
                (search_session, rank, session_params))

    @abstractmethod
    def _update(self, search_session, rank, session_params, history, last_session):
        """
        Update internal state using history (previously shown docs) and the last_session flag
        (set to true if this is the last session in the task).
        """
        pass

    def finish_iteration(self):
        for task_sessions in self.per_task_sessions.itervalues():
            history = set()
            for num, (search_session, rank, session_params) in enumerate(task_sessions):
                last_session = num == len(task_sessions) - 1
                self._update(search_session, rank, session_params, history, last_session)
                for d in search_session.web_results:
                    # TODO(chuklin): in the original paper history is not just the occurence
                    # of the document, but rather examination of it. Ideally, one should
                    # sum over position of the first examined occurence: equation (27).
                    history.add(d.id)

        # Reset the per-task dict
        self.per_task_sessions.clear()




class ParamStatic(Param):
    """A parameter with a fixed value."""

    def __init__(self, param):
        self.param = param

    def value(self):
        return self.param

    def update(self, search_session, rank, *args):
        pass
