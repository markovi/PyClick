#
# Copyright (C) 2015  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
import json

from pyclick.search_session import SearchResult


__author__ = 'Ilya Markov'


class SearchSession(object):
    """
    A single-query search session, which consists of a query
    and a list of corresponding web documents shown on a SERP.
    """

    def __init__(self, query):
        self.query = query
        self.web_results = []

    def get_clicks(self):
        """
        Returns the list of clicks corresponding to web_results.
        In particular, get_clicks()[i] is 1 if web_result[i] was clicked and 0 otherwise.
        """
        return [result.click for result in self.web_results]

    def get_last_click_rank(self):
        """
        Returns the rank of the last-clicked document (starting from 0).
        If no document is clicked, returns len(web_results).
        """
        clicks = self.get_clicks()
        click_ranks = [r for r, click in enumerate(clicks) if click]
        last_click_rank = click_ranks[-1] if len(click_ranks) else len(clicks)
        return last_click_rank

    def to_JSON(self):
        """Converts the session into JSON."""
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)

    @classmethod
    def from_JSON(cls, json_str):
        """Extracts a session for a JSON string."""
        session = cls("")
        session.__dict__ = json.loads(json_str)

        web_results = []
        for web_result_json in session.web_results:
            web_result = SearchResult.from_JSON(web_result_json)
            web_results.append(web_result)
        session.web_results = web_results

        return session

    def __str__(self):
        return self.to_JSON()

    def __repr__(self):
        return str(self)
