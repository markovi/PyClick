#
# Copyright (C) 2015  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
__author__ = 'Ilya Markov'


class SearchResult(object):
    """
    A search result, which contains a unique identifier and user interactions with the result.
    """

    def __init__(self, search_result_id, click):
        self.id = search_result_id
        """An identifier of the search result."""
        self.click = 0
        """A click on the search result. Can be either 1 (click) or 0 (no click)."""

        if click in [0, 1]:
            self.click = click
        else:
            raise RuntimeError("Invalid click value: %r" % click)

    @classmethod
    def from_JSON(cls, json_str):
        result = cls("", 0)
        result.__dict__ = json_str
        return result