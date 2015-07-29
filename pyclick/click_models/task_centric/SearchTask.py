#
# Copyright (C) 2015  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
from collections import OrderedDict

__author__ = 'Ilya Markov'


class SearchTask(object):
    """A search task consisting of multiple search sessions."""

    def __init__(self, task):
        self._task = task
        self.search_sessions = []

    def __str__(self):
        return "%s:%r" % (self._task, [search_session.query for search_session in self.search_sessions])

    def __repr__(self):
        return str(self)

    @staticmethod
    def get_search_tasks(search_sessions):
        """
        Groups search sessions by task and returns the list of all tasks.

        :param search_sessions: Task-centric search sessions.
        :returns: The list of tasks.
        """
        search_tasks = OrderedDict()

        for search_session in search_sessions:
            if search_session.task not in search_tasks:
                search_tasks[search_session.task] = SearchTask(search_session.task)

            search_tasks[search_session.task].search_sessions.append(search_session)

        return search_tasks.values()
