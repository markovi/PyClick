#
# Copyright (C) 2015  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
from abc import abstractmethod
from collections import defaultdict

__author__ = 'Ilya Markov'


class Param(object):
    """An abstract parameter of a click model."""

    def __str__(self):
        if self.value():
            return '%.4f' % self.value()
        else:
            return 'None'

    def __repr__(self):
        return str(self)

    def from_json(self, json_str):
        self.__dict__ = json_str

    @abstractmethod
    def value(self):
        """
        Returns the current value of the parameter.

        :returns: The current value of the parameter.
        """
        pass

    @abstractmethod
    def update(self, search_session, rank):
        """
        Updates the value of the parameter based on the observed search session.
        Note that this parameter corresponds to a search result at the given rank.

        :param search_session: The observed search session.
        :param rank: The rank of a search result, which corresponds to the current parameter.
        """
        pass

    @abstractmethod
    def __iadd__(self, other):
        """
        Concatenates the current parameter and the _other_ parameter.
        Returns the concatenated parameter.

        :param other: The parameter to concatenate with the current one.
        :returns: The concatenated parameter.
        """
        pass


class ParamMLE(Param):
    """A parameter used in the maximum likelihood estimation."""

    def __init__(self):
        self._numerator = 1
        self._denominator = 2

    def value(self):
        return self._numerator / float(self._denominator)

    @abstractmethod
    def update(self, search_session, rank):
        pass

    def __iadd__(self, other):
        assert type(self) == type(other)

        self._numerator += other._numerator - 1
        self._denominator += other._denominator - 2

        return self


class ParamEM(Param):
    """A parameter used in the expectation-maximization inference."""

    PROB_MIN = 0.000001
    """The probability to use instead of 0 to protect from the math domain errors."""

    def __init__(self):
        self._numerator = 1
        self._denominator = 2

    def value(self):
        return min(self._numerator / float(self._denominator), 1 - self.PROB_MIN)

    def update(self, search_session, rank, session_params):
        """
        Updates the value of the parameter based on the given search session
        and the values of other parameters.

        :param search_session: The currently observed search session.
        :param rank: The currently observed rank.
        :param session_params: The current values of the parameters corresponding to the current search session.
            These values are calculated on the previous iteration of EM
            (or the default values are used in case this is the first iteration).
        """
        if self._is_update_needed(search_session, rank):
            self._numerator += self._get_numerator_update(search_session, rank, session_params)
            self._denominator += self._get_denominator_update(search_session, rank, session_params)

    @classmethod
    def _get_numerator_update(cls, search_session, rank, session_params):
        """
        Calculates and returns the update for the numerator of the parameter.

        :param search_session: The currently observed search session.
        :param rank: The currently observed rank.
        :param session_params: The current values of the parameters corresponding to the current search session.
            These values are calculated on the previous iteration of EM
            (or the default values are used in case this is the first iteration).

        :returns: The update for the numerator of the parameter.
        """
        pass

    @classmethod
    def _get_denominator_update(cls, search_session, rank, session_params):
        """
        Calculates and returns the update for the denominator of the parameter.

        :param search_session: The currently observed search session.
        :param rank: The currently observed rank.
        :param session_params: The current values of the parameters corresponding to the current search session.
            These values are calculated on the previous iteration of EM
            (or the default values are used in case this is the first iteration).

        :returns: The update for the denominator of the parameter.
        """
        pass

    @classmethod
    def _is_update_needed(cls, search_session, rank):
        """
        Checks whether the parameter should be updated given the observed search session and the current rank.
        Returns True if the update is needed and False otherwise.

        :param search_session: The currently observed search session.
        :param rank: The currently observed rank.

        :returns" True if the update is needed and False otherwise.
        """
        return True

    def __iadd__(self, other):
        assert type(self) == type(other)

        self._numerator += other._numerator - 1
        self._denominator += other._denominator - 2

        return self


class ParamStatic(Param):
    """A parameter with a fixed value."""

    def __init__(self, param=0):
        self.param = param

    def value(self):
        return self.param

    def update(self, search_session, rank, *args):
        pass

    def __iadd__(self, other):
        """
        The value of the static parameter cannot be changed.
        """
        assert type(self) == type(other)
        return self
