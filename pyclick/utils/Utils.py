#
# Copyright (C) 2015  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#

__author__ = 'Ilya Markov'


class Utils:
    """
    Utility methods.
    """

    @staticmethod
    def get_unique_queries(search_sessions):
        """
        Extracts and returns the set of unique queries contained in a given list of search sessions.

        :param search_sessions: The list of search sessions.
        :return: The set of unique queries.
        """
        queries = set()
        for search_session in search_sessions:
            queries.add(search_session.query)
        return queries

    @staticmethod
    def filter_sessions(search_sessions, queries):
        """
        Filters the given list of search sessions
        so that it contains only a given list of queries.
        Returns the filtered list of sessions.

        :param search_sessions: The unfiltered list of search sessions.
        :param queries: The list of queries to be retained.
        :return: The filtered list of search sessions.
        """
        search_sessions_filtered = []
        for search_session in search_sessions:
            if search_session.query in queries:
                search_sessions_filtered.append(search_session)
        return search_sessions_filtered
