#
# Copyright (C) 2015  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
from pyclick.search_session.SearchSession import SearchSession

__author__ = 'Ilya Markov'


class TaskCentricSearchSession(SearchSession):
    """A single search session that is a part of a larger search task."""

    def __init__(self, task, query):
        super(TaskCentricSearchSession, self).__init__(query)
        self.task = task
