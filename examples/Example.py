#
# Copyright (C) 2015  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
import sys

import time

from pyclick.click_models.Evaluation import LogLikelihood, Perplexity
from pyclick.click_models.UBM import UBM
from pyclick.click_models.DBN import DBN
from pyclick.click_models.SDBN import SDBN
from pyclick.click_models.DCM import DCM
from pyclick.click_models.CCM import CCM
from pyclick.click_models.CTR import DCTR, RCTR, GCTR
from pyclick.click_models.CM import CM
from pyclick.click_models.PBM import PBM
from pyclick.click_models.alternative_inference.BBM import BBM
from pyclick.click_models.alternative_inference.UBMProbit import UBMProbit
from pyclick.utils.Utils import Utils
from pyclick.utils.YandexRelPredChallengeParser import YandexRelPredChallengeParser


__author__ = 'Ilya Markov'


if __name__ == "__main__":
    print "==============================="
    print "This is an example of using PyClick for training and testing click models."
    print "==============================="

    if len(sys.argv) < 4:
        print "USAGE: %s <click_model> <dataset> <sessions_max>" % sys.argv[0]
        print "\tclick_model - the name of a click model to use."
        print "\tdataset - the path to the dataset from Yandex Relevance Prediction Challenge"
        print "\tsessions_max - the maximum number of one-query search sessions to consider"
        print ""
        sys.exit(1)

    click_model = globals()[sys.argv[1]]()
    search_sessions_path = sys.argv[2]
    search_sessions_num = int(sys.argv[3])

    search_sessions = YandexRelPredChallengeParser().parse(search_sessions_path, search_sessions_num)

    train_test_split = int(len(search_sessions) * 0.75)
    train_sessions = search_sessions[:train_test_split]
    train_queries = Utils.get_unique_queries(train_sessions)

    test_sessions = Utils.filter_sessions(search_sessions[train_test_split:], train_queries)
    test_queries = Utils.get_unique_queries(test_sessions)

    print "-------------------------------"
    print "Training on %d search sessions (%d unique queries)." % (len(train_sessions), len(train_queries))
    print "-------------------------------"

    start = time.time()
    click_model.train(train_sessions)
    end = time.time()
    print "\tTrained %s click model in %i secs:\n%r" % (click_model.__class__.__name__, end - start, click_model)

    print "-------------------------------"
    print "Testing on %d search sessions (%d unique queries)." % (len(test_sessions), len(test_queries))
    print "-------------------------------"

    loglikelihood = LogLikelihood()
    perplexity = Perplexity()

    start = time.time()
    ll_value = loglikelihood.evaluate(click_model, test_sessions)
    end = time.time()
    print "\tlog-likelihood: %f; time: %i secs" % (ll_value, end - start)

    start = time.time()
    perp_value = perplexity.evaluate(click_model, test_sessions)[0]
    end = time.time()
    print "\tperplexity: %f; time: %i secs" % (perp_value, end - start)
