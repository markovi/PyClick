#
# Copyright (C) 2015  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
from pyclick.click_models.task_centric.TaskCentricSearchSession import TaskCentricSearchSession
from pyclick.search_session.SearchResult import SearchResult

__author__ = 'Ilya Markov, Bart Vredebregt, Nick de Wolf'


class YandexRelPredChallengeParser:
    """
    A parser for the publicly available dataset, released by Yandex (https://www.yandex.com)
    for the Relevance Prediction Challenge (http://imat-relpred.yandex.ru/en).
    """

    @staticmethod
    def parse(sessions_filename, sessions_max=None):
        """
        Parses search sessions, formatted according to the Yandex Relevance Prediction Challenge (RPC)
        (http://imat-relpred.yandex.ru/en/datasets).
        Returns a list of SearchSession objects.

        An RPC file contains lines of two formats:
        1. Query action
        SessionID TimePassed TypeOfAction QueryID RegionID ListOfURLs
        2. Click action
        SessionID TimePassed TypeOfAction URLID

        :param sessions_filename: The name of the file with search sessions formatted according to RPC.
        :param sessions_max: The maximum number of search sessions to return.
        If not set, all search sessions are parsed and returned.

        :returns: A list of parsed search sessions, wrapped into SearchSession objects.
        """
        sessions_file = open(sessions_filename, "r")
        sessions = []

        for line in sessions_file:
            if sessions_max and len(sessions) >= sessions_max:
                break

            entry_array = line.strip().split("\t")

            # If the entry has 6 or more elements it is a query
            if len(entry_array) >= 6 and entry_array[2] == "Q":
                task = entry_array[0]
                query = entry_array[3]
                results = entry_array[5:]
                session = TaskCentricSearchSession(task, query)

                for result in results:
                    result = SearchResult(result, 0)
                    session.web_results.append(result)

                sessions.append(session)

            # If the entry has 4 elements it is a click
            elif len(entry_array) == 4 and entry_array[2] == "C":
                if entry_array[0] == task:
                    clicked_result = entry_array[3]
                    if clicked_result in results:
                        index = results.index(clicked_result)
                        session.web_results[index].click = 1

            # Else it is an unknown data format so leave it out
            else:
                continue

        return sessions
