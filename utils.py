
#fname='try3.csv'
fname='tpcds-run.csv'
import csv, json, os
import matplotlib.pyplot as plt
import numpy as np

'''
def get_all_plans(fname):
    jsonstrs = []
    curr = ""
    with open(fname,'r') as f:
        for row in f:
            if 'Timing is on' in row:
                continue
            if row[:6] == "Time: ":
                # jsonstrs.append((curr, row.strip("\n")))
                jsonstrs.append(curr.strip(' ').strip('QUERY PLAN').strip('-'))
                curr = ""
            else:
                curr += row.replace('+', "").replace("(1 row)\n", "").strip('\n').strip(' ')
    strings = [s for s in jsonstrs if s[-1] == ']']
    jss = [json.loads(s)[0]['Plan'] for s in strings]
    # jss is a list of json-transformed dicts, one for each query
    return jss
'''

def get_time(fname):
    times = []

    with open(fname,'r') as f:
        for row in f:
            if 'Execution Time' in row:
                #print(row)
                times.append(float(row.split('"Execution Time":')[1]))

    #print(prevprev, prev)
    return times
    return

################################################################################
'''
jsonstrs = []
curr = ""
with open(fname,'r') as f:
    for row in f:
        if 'Timing is on' in row:
            continue
        if row[:6] == "Time: ":
            # jsonstrs.append((curr, row.strip("\n")))
            jsonstrs.append(curr.strip(' ').strip('QUERY PLAN').strip('-'))
            curr = ""
        else:
            curr += row.replace('+', "").replace("(1 row)\n", "").strip('\n').strip(' ')

'''
def get_time_mean_var(data_dir):
    fnames = [(int(fname.split('temp')[1].strip('.csv')), fname)
              for fname in os.listdir(data_dir) if 'csv' in fname]
    all_time = []
    fnames.sort()
    names = []
    for i, fname in fnames:
        if i >= 20: continue
        arr = np.array(get_time(data_dir + fname))
        all_time.append(arr)
        print(fname, "mean: ", np.mean(arr), 'variance: ', np.var(arr))
        names.append(str(i))
    all_time = np.array(all_time).astype(np.float32)
    print(all_time.shape)
    plt.boxplot(all_time.T, labels=names)
    '''
    f, axes = plt.subplots(2, 2, sharey=False, squeeze=False)
    f.set_figheight(10)
    f.set_figwidth(12.8)
    ax1, ax2 = axes[0][0], axes[0][1]
    ax1.set_title('All Templates')
    ax1.boxplot(all_time.T, labels=names)

    ax2.set_title('All Except 17, 20, 21')
    idx = [i for i in range(20) if i !=16 and i!=19 and i!=20]
    print(all_time[idx, :].T.shape, np.array(names)[idx].shape)
    ax2.boxplot(all_time[idx, :].T, labels=np.array(names)[idx])

    ax3, ax4 = axes[1][0], axes[1][1]
    ax3.set_title('Temp 20, Mean: {}, Var: {}'.format(np.mean(all_time[19]), np.var(all_time[19])))
    ax3.boxplot(all_time[[19], :].T, labels=np.array(names)[[19]])

    ax4.set_title('\nTemp 17, Mean: {}, Var: {}'\
                  .format(np.mean(all_time[16]), np.var(all_time[16])))
    ax4.boxplot(all_time[[16], :].T, labels=np.array(names)[[16]])
    f.suptitle('TPC-H query exec time by temp (320 queries per temp)')
    '''
    plt.savefig('tpch_exec_time_0^1SF.png')
'''
strings = [s for s in jsonstrs if s[-1] == ']']
#
my_col = {}
for s in strings:
    js = [json.loads(s)[0]['Plan']]
    def explore(js):
        for jss in js:
            if jss["Node Type"] not in my_col:
                my_col[jss['Node Type']] = [[], {k: [jss[k]] for k in list(jss.keys())}]
            else:
                for k in jss.keys():
                    if k not in my_col[jss['Node Type']][1]:
                        my_col[jss['Node Type']][1][k] = [jss[k]]
                    else:
                        my_col[jss['Node Type']][1][k].append(jss[k])
            if 'Plans' in jss:
                #if jss['Node Type'] == 'Seq Scan':
                #    print(jss['Plans'])
                my_col[jss['Node Type']][0].append(len(jss['Plans']))
                explore(jss['Plans'])
            else:
                my_col[jss['Node Type']][0].append(0)
    explore(js)
'''
