#
# Copyright (C) 2014  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
import sys
from session.Session import *

__author__ = 'Ilya Markov'


def parse_wsdm_sessions(sessions_filename):
    """
    Parses WSCD search sessions in the given file into Session objects.
    """
    sessions_file = open(sessions_filename, "r")
    sessions = []

    for line in sessions_file:
        for session_str in line.split(";"):
            if session_str.strip() == "":
                continue

            session_str = session_str.strip()
            session_str = session_str.split("\t")
            query = session_str[0]

            session_str = session_str[1].split(":")
            docs = session_str[0].strip().split(",")
            clicks = session_str[1].strip().split(",")

            session = Session(query)
            for doc in docs:
                web_result = Result(doc, -1, 1 if doc in clicks else 0)
                session.add_web_result(web_result)

            sessions.append(session)
            return sessions


if __name__ == '__main__':
    """
    An example of using PyClick with the Yandex WSCD dataset.
    """
    train_sessions = parse_wsdm_sessions("data/trainSample")
    test_sessions = parse_wsdm_sessions("data/testSample")
