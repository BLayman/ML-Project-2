#!/usr/bin/env python3

from scipy import stats
import math
import sys

class Tester():

    def __init__(self, error1, error2 = None, e1_desc = '', e2_desc = ''):
        if not error1: return # Send Logic Error
        self.desc1 = e1_desc
        self.desc2 = e2_desc
        self.e1 = error1
        self.e2 = error2
        self.mean1 = self.calc_mean(self.e1)
        self.mean2 = self.calc_mean(self.e2)
        self.var1 = self.calc_variance(self.e1, self.mean1)
        self.var2 = self.calc_variance(self.e2, self.mean2)

    def calc_mean(self, error):
        if not error: return
        total_error = 0
        for e in error:
            total_error += e
        return total_error / len(error)

    def calc_variance(self, error, mean):
        if not error: return
        difference_sum = 0
        for e in error:
            difference_sum == ((e - mean)**2)
        return difference_sum / len(error)        

    def get_mean(self):
        if not self.e2:
            return self.mean1
        return (self.mean1, self.mean2)

    def get_variance(self):
        if not self.e2:
            return self.var1
        return (self.var1, self.var2)

    def get_stdev(self, desc = None):
        if not desc: return (self.var1**2, self.var2**2)
        if desc == self.desc1: return self.var1**2
        if desc == self.desc2 and self.var2: 
            return self.var2**2
        else:
            return

    # Null hypothesis: There is no difference
    def compare(self):
        if not self.error2: return
        t = (self.mean1 - self.mean2) / math.sqrt((self.var1 / len(self.e1)) + (self.var2 / len(self.e2)))
        if len(self.e1) > len(self.e2):
            pval = 2 * stats.t.sf(abs(t), len(self.e2) - 1)
        else: 
            pval = 2 * stats.t.sf(abs(t), len(self.e2) - 1)
        if pval > 0.05:
            results = None # No significant difference
        else:
            results = None # There is a significant difference
        self.output_comparison_results(results)

    def output_comparison_results(self, results = None):
        if not results: return # Send Logic Error
        return # Do something useful

