
#fname='try3.csv'
fname='tpcds-run.csv'
import csv, json

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

################################################################################
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
