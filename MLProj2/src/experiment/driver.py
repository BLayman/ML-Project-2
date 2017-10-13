#!/usr/bin/env python3

"""
arg0: Invoking script
arg1: # of parameters for the rosenbrock function
arg2: # of MLP hidden layers
arg3: # of MLP hidden neurons
arg4: # of MLP output neurons
arg5: # of RFB hidden neurons
arg6: # of RFB output neurons
arg7: # of data points
"""

from scipy import stats
from generatedata import GenerateData as GD
from backprop.bpAlg import BPAlg as BP
from radialBasis.radialBasisOut import radialBasisOut as RFB
# from radialBasis.rfbAlg import RFBAlg as RFB
import sys


def get_mean(error):
    if not error:
        return
    total_error = 0
    for e in error:
        total_error += e
    return total_error / len(error)

def get_variance(error = [], mean = -1):
    if mean == -1:
        return
    difference_sum = 0
    for e in error:
        difference_sum += ((e - mean)**2)
    return difference_sum / len(error)

# Using a t-score because we assume normal distribution
# Null hypothesis: bp_mean = rfb_mean
def compare(bp_mean, bp_var, rbf_mean, rfb_var, size):
    if not bp_var or not rfb_var:
        return
    t = (bp_mean - rfb_mean) / math.sqrt((bp_var / size) + rfb_var / size))
    pval = 2 * stats.t.sf(math.abs(t), size - 1)
    if pval >= 0.05:
        pass # No significant difference
    else:
        pass # There was a significant difference


def output_results(results):
    if not results:
        print("A logic error has occurred")
        return
    return

if __name__ == "__main__":
    try:
        train = GD(int(sys.argv[7]), int(sys.argv[1]))
        train_data = train.get_data()
        train_target = train.get_target_vector()
        bp = BP()
        bp_network = bp.train(data, target, int(sys.arv[2]), int(sys.argv[3], int(sys.argv[4])))
        rfb = RFB()
        rfb_network = rfb.train(data, target, int(sys.argv[5]), int(sys.argv[6]))

        test = GD(int(sys.argv[7]), int(sys.argv[1]))
        test_data = test.get_data()
        test_target = test.get_target_vector()
    
        bp_error = bp.test(test_data, test_target, bp_network)
        bp_mean = get_mean(bp_error)
        
        rfb_error = rfb.test(test_data, test_target, rfb_network)
        rfb_mean = get_mean(rfb_error)

        results = compare(bp_mean, get_variance(bp_mean, bp_error), rfb_mean, get_variance(rfb_mean, rfb_error), len(test_data))
        output_results(results)
    except IndexError:
        print("Not enough arguments provided.")
    finally:
        sys.exit()









