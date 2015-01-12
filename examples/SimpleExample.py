#!/usr/bin/env python
#
# Copyright (C) 2014  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
import sys
from click_models.DBN import DBN
from click_models.DCM import DCM
from click_models.SimpleDBN import SimpleDBN
from click_models.SimpleDCM import SimpleDCM
from click_models.UBM import UBM
from session.Session import *

__author__ = 'Ilya Markov'


def parse_wsdm_sessions(sessions_filename):
    """Parses search sessions in the given file into Session objects."""

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


def main(train_filename, test_filename):
    train_sessions = parse_wsdm_sessions(train_filename)
    test_sessions = parse_wsdm_sessions(test_filename)

    #TODO: fix initialization
    for click_model_class in [SimpleDCM, SimpleDBN, DBN, UBM]:
        print "==== %s ====" % click_model_class.__name__
        click_model = click_model_class(click_model_class.get_prior_values())
        click_model.train(train_sessions)

        print click_model

        print "Log-likelihood and perplexity"
        print click_model.test(test_sessions)
        print ""


# An example of using PyClick.
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print "USAGE: %s <file with train sessions> <file with test sessions>" % sys.argv[0]
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
