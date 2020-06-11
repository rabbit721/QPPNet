import pickle, argparse
import sys
from parse import *
import matplotlib.pyplot as plt
import numpy as np

def parse_int(string):
    return int(string)

def parse_float(string):
    return float(string)

format_str = "epoch: {epoch:Int}; iter_num: {iters:Int}; " \
             "total_loss: {total_loss:Float}; " \
             "test_loss: {test_loss:Float}; pred_err: {pred_err:Float}; " \
             "R(q): {r_q:Float}" \
             "losses: Seq Scan [{seq_scan_loss:Float}]; " \
             "Sort [{sort_loss:Float}]; " \
             "Hash [{hash_loss:Float}]; " \
             "Hash Join [{hash_join_loss:Float}]; " \
             "Merge Join [{merge_join_loss:Float}]; " \
             "Aggregate [{aggregate_loss:Float}]; " \
             "Nested Loop [{nested_loop_loss:Float}]; " \
             "Limit [{limit_loss:Float}]; " \
             "Subquery Scan [{subq_scan_loss:Float}]; " \
             "Materialize [{materalize_loss:Float}]; " \
             "Gather Merge [{gather_merge_loss:Float}]; " \
             "Gather [{gather_loss:Float}]; "

parser = argparse.ArgumentParser(description="Parses command.")
parser.add_argument("-in", "--log_file", help="Your log file for parse.")
parser.add_argument("--var_name", type=str, default='total_loss', help="Item to be plotted.")
parser.add_argument("-len", "--plot_length", type=int, default=10000, help = "# of epoch to be plotted (default: 10000)")
parser.add_argument("-smooth", "--smooth_interval", type=int, default=50, help = "# intervals to take avg (default: 50))")


options = parser.parse_args()
data = []

with open(options.log_file, 'r') as f:
    for line in f:
        if 'total_loss' not in line:
            continue
        #print(line)
        res = parse(format_str, line, dict(Int=parse_int, Float=parse_float))
        data.append(res.named)

'''
with open(options.dump_name, 'wb') as outf:
    pickle.dump(complete, outf, pickle.HIGHEST_PROTOCOL)
'''


#GA = np.array([item['G_A'] for item in res]).astype(float)
currarray = np.array([item[options.var_name] for item in data]).astype(float)
currarray = currarray[:((len(currarray)//options.smooth_interval) * options.smooth_interval)]
print("# of epochs will be plotted: " + str(len(currarray)))

smoothed = [np.average(currarray[i*options.smooth_interval: (i+1)*options.smooth_interval])
            for i in range(min(len(currarray), options.plot_length) // options.smooth_interval)]
#plt.plot(epochs, GA, 'r')

#if i != 6:
x_range = range(options.plot_length)
plt.plot(x_range[:min(len(currarray), options.plot_length):options.smooth_interval], smoothed, 'b', label=options.var_name)

plt.legend()
plt.title(options.var_name)

plt.savefig('./loss_graphs/' + options.var_name + '-new.png')
plt.figure()
