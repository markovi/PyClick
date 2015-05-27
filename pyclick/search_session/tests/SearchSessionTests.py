#
# Copyright (C) 2015  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
import unittest

from nose_parameterized.parameterized import parameterized

from pyclick.search_session import SearchResult
from search_session.SearchSession import SearchSession


__author__ = 'Ilya Markov'


class SearchSessionTestCase(unittest.TestCase):
    RANK_MAX = 10

    @parameterized.expand([
        ('standard_clicks', [1, 0, 1, 1, 0, 0, 0, 0, 0, 0], [1, 0, 1, 1, 0, 0, 0, 0, 0, 0]),
    ])
    def test_get_clicks(self, name, clicks, expected):
        session = SearchSession('some_query')
        for i in range(self.RANK_MAX):
            result = SearchResult('doc%d' % i, clicks[i])
            session.web_results.append(result)
        self.assertListEqual(session.get_clicks(), expected)

    @parameterized.expand([
        ('standard_clicks', [1, 0, 1, 1, 0, 0, 0, 0, 0, 0], 3),
        ('no_clicks', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 9),
        ('all_clicks', [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 9),
    ])
    def test_get_last_click_rank(self, name, clicks, expected):
        session = SearchSession('some_query')
        for i in range(self.RANK_MAX):
            result = SearchResult('doc%d' % i, clicks[i])
            session.web_results.append(result)
        self.assertEqual(session.get_last_click_rank(), expected)

    def test_to_from_JSON(self):
        session = SearchSession('some_query')
        for i in range(self.RANK_MAX):
            result = SearchResult('doc%d' % i, i % 2)
            session.web_results.append(result)
        session_encoded = session.to_JSON()
        session_decoded = SearchSession.from_JSON(session_encoded)

        self.assertEqual(session.query, session_decoded.query)
        for i in range(len(session.web_results)):
            self.assertEqual(session.web_results[i].__dict__, session_decoded.web_results[i].__dict__)
