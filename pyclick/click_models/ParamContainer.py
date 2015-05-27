#
# Copyright (C) 2015  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
from abc import abstractmethod

__author__ = 'Ilya Markov'


class ParamContainer(object):
    """An abstract container of parameters of a click model."""

    def __init__(self, param_class):
        """
        Initializes the container.

        :param param_class: The class of parameters to be stored in the container.
        """
        self._container = None
        self._param_class = param_class

    def size(self):
        """
        Returns the number of parameters in the container.

        :returns: The number of parameters in the container.
        """
        return len(self._container)

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


class QueryDocumentParamContainer(ParamContainer):
    """A container of click model parameters that depend on a query-document pair."""

    def __init__(self, param_class):
        super(QueryDocumentParamContainer, self).__init__(param_class)
        self._container = {}

    def get(self, query, search_result):
        """
        Returns a click model parameter that corresponds to the given query and search result.

        :param query: The query.
        :param search_result: The search result.
        :return: A click model parameter that corresponds to the given query and search result.
        """
        if query not in self._container:
            self._container[query] = {}

        if search_result not in self._container[query]:
            self._container[query][search_result] = self._param_class()

        return self._container[query][search_result]

    def set(self, param, query, search_result, *args):
        """
        Sets the given parameter for the given query and search result.

        :param query: The query.
        :param search_result: The search result.
        """
        if query not in self._container:
            self._container[query] = {}
        self._container[query][search_result] = param

    def __str__(self):
        param_str = ''
        for query in self._container:
            param_str += '%s: %r\n' % (query, self._container[query])
        return param_str

    def __repr__(self):
        return str(self)


class RankParamContainer(ParamContainer):
    """A container of click model parameters that depend on rank."""

    MAX_RANK_DEFAULT = 10
    """The default maximum rank."""

    def __init__(self, param_class, max_rank):
        """
        Initializes the container with a given maximum rank.

        :param param_class: The class of parameters to be stored in the container.
        :param max_rank: The maximum rank.
        """
        super(RankParamContainer, self).__init__(param_class)
        self._container = [self._param_class() for i in range(max_rank)]

    @classmethod
    def default(cls, param_class):
        """
        Creates a container with the default maximum rank of 10.

        :param param_class: The class of parameters to be stored in the container.
        """
        return cls(param_class, cls.MAX_RANK_DEFAULT)

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

    def __str__(self):
        return '%s\n' % ' '.join([str(item) for item in self._container])

    def __repr__(self):
        return str(self)


class RankSquaredParamContainer(ParamContainer):
    """
    A container of click model parameters that are double-dependent on rank.
    For example, examination probability in the UBM model
    depends on the rank of the current document and the rank of the previously clicked document.
    """

    MAX_RANK_DEFAULT = 10
    """The default maximum rank."""

    def __init__(self, param_class, max_rank1, max_rank2):
        """
        Initializes the container with given maximum ranks.

        :param param_class: The class of parameters to be stored in the container.
        :param max_rank1: The maximum rank on the first dimension.
        :param max_rank2: The maximum rank on the second dimension.
        """
        super(RankSquaredParamContainer, self).__init__(param_class)
        self._container = [[self._param_class() for i in range(max_rank1)] for j in range(max_rank2)]

    @classmethod
    def default(cls, param_class):
        """
        Creates a container with the default maximum ranks of 10.

        :param param_class: The class of parameters to be stored in the container.
        """
        return cls(param_class, cls.MAX_RANK_DEFAULT, cls.MAX_RANK_DEFAULT)

    def size(self):
        return len(self._container) * len(self._container[0])

    def get(self, rank1, rank2):
        """
        Returns a click model parameter that corresponds to the given ranks.

        :param rank1: The rank on the first dimension.
        :param rank2: The rank on the second dimension.
        :returns: A click model parameter that corresponds to the given ranks.
        """
        return self._container[rank1][rank2]

    def set(self, param, rank1, rank2):
        """
        Sets the given parameter at the given ranks.

        :param rank1: The rank on the first dimension.
        :param rank2: The rank on the second dimension.
        """
        self._container[rank1][rank2] = param

    def __str__(self):
        return '\n'.join([' '.join(['{:8s}'.format(item) for item in row]) for row in self._container])

    def __repr__(self):
        return str(self)


class SingleParamContainer(ParamContainer):
    """
    A container of a click model parameter that does not depend on anything,
    e.g., continuation probability in the DBN model.
    """

    def __init__(self, param_class):
        super(SingleParamContainer, self).__init__(param_class)
        self._container = self._param_class()

    def size(self):
        return 1

    def get(self):
        return self._container

    def set(self, param):
        self._container = param

    def __str__(self):
        return '%s\n' % str(self._container)

    def __repr__(self):
        return str(self)