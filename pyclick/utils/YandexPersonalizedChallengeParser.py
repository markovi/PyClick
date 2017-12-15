#
# Copyright (C) 2015  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
from __future__ import print_function

from pyclick.click_models.task_centric.TaskCentricSearchSession import TaskCentricSearchSession
from pyclick.search_session.SearchResult import SearchResult

__author__ = 'Ilya Markov'


class YandexPersonalizedChallengeParser:
    """
    A parser for the publicly available dataset, released by Yandex (https://www.yandex.com)
    for the Personalized Web Search Challenge (https://www.kaggle.com/c/yandex-personalized-web-search-challenge).
    """

    @staticmethod
    def parse(sessions_filename, sessions_max=None):
        """
        Parses search sessions, formatted according to the Yandex Personalized Web Search Challenge (PWSC)
        (http://imat-relpred.yandex.ru/en/datasets).
        Returns a list of SearchSession objects.

        An PWSC file contains lines of three formats:
        1. Session metadata
        SessionID TypeOfRecord Day USERID
        2. Query action
        SessionID TimePassed TypeOfRecord SERPID QueryID ListOfTerms ListOfURLsAndDomains
        3. Click action
        SessionID TimePassed TypeOfRecord SERPID URLID

        :param sessions_filename: The name of the file with search sessions formatted according to PWSC.
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

            if len(entry_array) == 4 and entry_array[1] == 'M':
                pass

            elif len(entry_array) >= 7 and (entry_array[2] == 'Q' or entry_array[2] == 'T'):
                task = entry_array[0]
                serp = entry_array[3]
                query = entry_array[4]
                urls_domains = entry_array[6:]
                session = TaskCentricSearchSession(task, query)

                results = []
                for url_domain in urls_domains:
                    result = url_domain.strip().split(',')[0]
                    url_domain = SearchResult(result, 0)
                    results.append(result)

                    session.web_results.append(url_domain)

                sessions.append(session)

            elif len(entry_array) == 5 and entry_array[2] == 'C':
                if entry_array[0] == task and entry_array[3] == serp:
                    clicked_result = entry_array[4]
                    if clicked_result in results:
                        index = results.index(clicked_result)
                        session.web_results[index].click = 1

            else:
                print('Unknown data format: %s' % line)
                continue

        return sessions
