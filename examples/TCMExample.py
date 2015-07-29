#
# Copyright (C) 2015  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#
import sys
from pyclick.click_models.task_centric.SearchTask import SearchTask

from pyclick.click_models.task_centric.TCM import TCM
from pyclick.utils.Utils import Utils
from pyclick.utils.YandexRelPredChallengeParser import YandexRelPredChallengeParser


__author__ = 'Ilya Markov, Aleksandr Chuklin'


if __name__ == "__main__":
    print "==============================="
    print "This is an example of using PyClick for training and testing the TCM click model."
    print "==============================="

    if len(sys.argv) < 3:
        print "USAGE: %s <dataset> <sessions_max>" % sys.argv[0]
        print "\tdataset - the path to the dataset from Yandex Relevance Prediction Challenge"
        print "\tsessions_max - the maximum number of one-query search sessions to consider"
        print ""
        sys.exit(1)

    click_model = TCM()
    search_sessions_path = sys.argv[1]
    search_sessions_num = int(sys.argv[2])

    search_sessions = YandexRelPredChallengeParser().parse(search_sessions_path, search_sessions_num)

    train_test_split = int(len(search_sessions) * 0.75)
    train_sessions = search_sessions[:train_test_split]
    train_queries = Utils.get_unique_queries(train_sessions)
    train_tasks = SearchTask.get_search_tasks(train_sessions)

    # test_sessions = Utils.filter_sessions(search_sessions[train_test_split:], train_queries)
    # test_queries = Utils.get_unique_queries(test_sessions)
    # test_tasks = SearchTask.get_search_tasks(test_sessions)

    print "-------------------------------"
    print "Training on %d search tasks (%d search sessions, %d unique queries)." % \
          (len(train_tasks), len(train_sessions), len(train_queries))
    print "-------------------------------"

    click_model.train(train_tasks)
    print "\tTrained %s click model:\n%r" % (click_model.__class__.__name__, click_model)

    # print "-------------------------------"
    # print "Testing on %d search sessions (%d unique queries)." % (len(test_sessions), len(test_queries))
    # print "Repeated URLs:", TaskCentricSearchSession.count_repeated_urls(test_sessions).most_common(5)
    # print "-------------------------------"
