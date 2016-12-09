#
# Copyright (C) 2015  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
from abc import abstractmethod
from collections import defaultdict
import json

__author__ = 'Ilya Markov'


class ParamContainer(object):
    """An abstract container of parameters of a click model."""

    def __init__(self, param_class, *args):
        """
        Initializes the container.

        :param param_class: The class of parameters to be stored in the container.
        :param args: The arguments needed to create a parameter instance (optional).
        """
        self._container = None
        self._param_class = param_class
        self._param_args = args

    def size(self):
        """
        Returns the number of parameters in the container.

        :returns: The number of parameters in the container.
        """
        return len(self._container)

    def to_json(self):
        """
        Converts the parameter container into JSON and returns the corresponding string.

        :returns: The JSON representation of the container.
        """
        return json.dumps(self._container, default=lambda o: o.__dict__)

    @abstractmethod
    def from_json(self, json_str):
        """
        Initializes the parameter container from the given JSON string.

        :param json_str: The JSON representation of the container.
        """

    @abstractmethod
    def get(self, *args):
        """
        Returns a click model parameter that corresponds to given indices.
        NOTE: The names and semantics of the indices must be specified by a subclass!

        :returns: A click model parameter that corresponds to given indices.
        """
        pass

    @abstractmethod
    def set(self, param, *args):
        """
        Sets the given parameter according to given indices.
        NOTE: The names and semantics of the indices must be specified by a subclass!
        """
        pass

    @abstractmethod
    def get_for_session_at_rank(self, search_session, rank):
        """
        Returns a click model parameter that corresponds to the search result
        in a given session at a given rank.

        :returns: A click model parameter that corresponds to the specified search result.
        """
        pass

    @abstractmethod
    def __iadd__(self, other):
        """
        Concatenates the current parameter container and the _other_ parameter container.
        Returns the concatenated parameter container.

        :param other: The parameter container to concatenate with the current one.
        :returns: The concatenated parameter container.
        """
        pass

    @abstractmethod
    def __iter__(self):
        """
        Returns the iterator over the elements of the container.

        :returns: The iterator over the elements of the container.
        """
        pass

    def apply_each(self, func):
        """
        Applies the given func to each parameter in this container.

        :param func: The function to apply.
        """
        iterator = iter(self)
        try:
            while True:
                func(iterator.next())
        except StopIteration:
            pass


class QueryDocumentParamContainer(ParamContainer):
    """A container of click model parameters that depend on a query-document pair."""

    PARAMS_PRINT_MAX = 100
    """
    The maximum number of parameters to output in the string representation of the container.
    Set to -1 to output all parameters.
    """

    def __init__(self, param_class, *args):
        super(QueryDocumentParamContainer, self).__init__(param_class, *args)
        self._container = defaultdict(lambda: defaultdict(lambda: self._param_class(*self._param_args)))

    def get(self, query, search_result):
        """
        Returns a click model parameter that corresponds to the given query and search result.

        :param query: The query.
        :param search_result: The search result.
        :return: A click model parameter that corresponds to the given query and search result.
        """
        return self._container[query][search_result]

    def set(self, param, query, search_result):
        """
        Sets the given parameter for the given query and search result.

        :param query: The query.
        :param search_result: The search result.
        """
        self._container[query][search_result] = param

    def get_for_session_at_rank(self, search_session, rank):
        query = search_session.query
        result = search_session.web_results[rank].id
        return self.get(query, result)

    def from_json(self, json_str):
        json_container = json.loads(json_str)
        for query in json_container:
            for result in json_container[query]:
                self._container[query][result] = self._param_class(*self._param_args)
                self._container[query][result].from_json(json_container[query][result])

    def __str__(self):
        param_str = ''
        counter = 0
        for query in self._container:
            if counter > self.PARAMS_PRINT_MAX >= 0:
                break
            #TODO: convert defaultdict into dict
            param_str += '%s: %r\n' % (query, dict(self._container[query]))
            counter += len(self._container[query])
        return param_str

    def __repr__(self):
        return str(self)

    def __iadd__(self, other):
        assert type(self) == type(other)

        for query in other._container:
            for search_result in other._container[query]:
                self._container[query][search_result] += other._container[query][search_result]

        return self

    def __iter__(self):
        return self._iterator()

    def _iterator(self):
        for query in self._container:
            for result in self._container[query]:
                yield self._container[query][result]


class RankParamContainer(ParamContainer):
    """A container of click model parameters that depend on rank."""

    MAX_RANK_DEFAULT = 10
    """The default maximum rank."""

    def __init__(self, param_class, max_rank, *args):
        """
        Initializes the container with a given maximum rank.

        :param param_class: The class of parameters to be stored in the container.
        :param max_rank: The maximum rank.
        """
        super(RankParamContainer, self).__init__(param_class, *args)
        self._container = [self._param_class(*self._param_args) for i in range(max_rank)]
        self.max_rank = max_rank

    @classmethod
    def default(cls, param_class, *args):
        """
        Creates a container with the default maximum rank of 10.

        :param param_class: The class of parameters to be stored in the container.
        :param args: The arguments needed to create a parameter instance (optional).
        """
        return cls(param_class, cls.MAX_RANK_DEFAULT, *args)

    def get(self, rank):
        """
        Returns a click model parameter that corresponds to the given rank.

        :param rank: The rank.
        :returns: A click model parameter that corresponds to the given rank.
        """
        return self._container[rank]

    def set(self, param, rank):
        """
        Sets the given parameter at the given rank.

        :param rank: The rank.
        """
        self._container[rank] = param

    def get_for_session_at_rank(self, search_session, rank):
        return self.get(rank)

    def from_json(self, json_str):
        json_container = json.loads(json_str)
        for rank, param in enumerate(self._container):
            param.from_json(json_container[rank])

    def __str__(self):
        return '%s\n' % ' '.join([str(item) for item in self._container])

    def __repr__(self):
        return str(self)

    def __iadd__(self, other):
        assert type(self) == type(other)
        assert len(self._container) == len(other._container)

        for rank in range(len(other._container)):
            self._container[rank] += other._container[rank]

        return self

    def __iter__(self):
        return iter(self._container)


class RankPrevClickParamContainer(ParamContainer):
    """
    A container of click model parameters that depend on rank
    and on the rank of the previously clicked result (see, e.g., UBM).
    """

    MAX_RANK_DEFAULT = 10
    """The default maximum rank."""

    def __init__(self, param_class, max_rank, *args):
        """
        Initializes the container with given maximum ranks.

        :param param_class: The class of parameters to be stored in the container.
        :param max_rank: The maximum rank.
        """
        super(RankPrevClickParamContainer, self).__init__(param_class, *args)
        self._container = [[self._param_class(*self._param_args) for i in range(max_rank)] for j in range(max_rank)]
        self.max_rank = max_rank

    @classmethod
    def default(cls, param_class, *args):
        """
        Creates a container with the default maximum rank of 10.

        :param param_class: The class of parameters to be stored in the container.
        :param args: The arguments needed to create a parameter instance (optional).
        """
        return cls(param_class, cls.MAX_RANK_DEFAULT, *args)

    def size(self):
        return len(self._container) * len(self._container[0])

    def get(self, rank, rank_prev_click):
        """
        Returns a click model parameter that corresponds to the given ranks.

        :param rank: The rank of a search result.
        :param rank_prev_click: The rank of the previously clicked search result.
        :returns: A click model parameter that corresponds to the given ranks.
        """
        return self._container[rank][rank_prev_click]

    def set(self, param, rank, rank_prev_click):
        """
        Sets the given parameter at the given ranks.

        :param rank: The rank of a search result.
        :param rank_prev_click: The rank of the previously clicked search result.
        """
        self._container[rank][rank_prev_click] = param

    def get_for_session_at_rank(self, search_session, rank):
        return self.get(rank, self._get_prev_clicked_rank(search_session, rank))

    def from_json(self, json_str):
        json_container = json.loads(json_str)
        for rank, _ in enumerate(self._container):
            for rank_prev_click, param in enumerate(self._container[rank]):
                param.from_json(json_container[rank][rank_prev_click])

    def __str__(self):
        return '\n'.join([' '.join(['{:8s}'.format(item) for item in row]) for row in self._container])

    def __repr__(self):
        return str(self)

    def __iadd__(self, other):
        assert type(self) == type(other)
        assert len(self._container) == len(other._container)
        assert len(self._container[0]) == len(other._container[0])

        container_size = (len(self._container), len(self._container[0]))

        for rank1 in range(container_size[0]):
            for rank2 in range(container_size[1]):
                self._container[rank1][rank2] += other._container[rank1][rank2]

        return self

    def __iter__(self):
        return self._iterator()

    def _iterator(self):
        for rank in xrange(len(self._container)):
            for rank_prev_click in xrange(len(self._container[0])):
                yield self._container[rank][rank_prev_click]

    @staticmethod
    def _get_prev_clicked_rank(search_session, rank):
        """
        Given the rank, returns the rank of the previously clicked search result.
        If none of the above results was clicked,
        returns M-1, where M is the number of results in a given search session.

        :param search_session: The current search session.
        :param rank: The rank of a search result.

        :returns: The rank of the previously clicked search result.
        """
        prev_clicks = [rank_click for rank_click, click in enumerate(
                search_session.get_clicks()[:rank]) if click]
        prev_click_rank = prev_clicks[-1] if len(prev_clicks) else len(search_session.web_results) - 1
        return prev_click_rank


class SingleParamContainer(ParamContainer):
    """
    A container of a click model parameter that does not depend on anything,
    e.g., continuation probability in the DBN model.
    """

    def __init__(self, param_class, *args):
        super(SingleParamContainer, self).__init__(param_class, *args)
        self._container = self._param_class(*self._param_args)

    def size(self):
        return 1

    def get(self):
        return self._container

    def set(self, param):
        self._container = param

    def get_for_session_at_rank(self, search_session, rank):
        return self.get()

    def from_json(self, json_str):
        self._container.from_json(json.loads(json_str))

    def __str__(self):
        return '%s\n' % str(self._container)

    def __repr__(self):
        return str(self)

    def __iadd__(self, other):
        assert type(self) == type(other)

        self._container += other._container

        return self

    def __iter__(self):
        return iter([self._container])
