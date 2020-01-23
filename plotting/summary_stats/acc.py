import os
import sys

import pandas as pd

class LogReader(object):
    def __init__(self, logFile, pruneAfter=5):
        self.accFile = pd.read_csv(logFile, delimiter=',\t', engine='python')
        self.pruneAfter = pruneAfter

    def get_best_test_acc(self):
    #{{{
        bestValIdx = self.accFile['Val_Top1'].dropna()[self.pruneAfter:].idxmax()
        bestTest = self.accFile['Test_Top1'].dropna()[bestValIdx]
        return bestTest
    #}}}
