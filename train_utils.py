import numpy as np
import collections, os, json
from attr_rel_dict import *

# need a huge array
# get the columns of all relations
num_rel = 8
max_num_attr = 16
num_q = 22


def get_basics(plan_dict):
    return [plan_dict['Plan Width'], plan_dict['Plan Rows'],
            plan_dict['Total Cost']]

def get_rel_one_hot(rel_name):
    arr = [0] * num_rel
    arr[rel_names.index(rel_name)] = 1
    return arr

def get_rel_attr_one_hot(rel_name, filter_line):
    arr = [0] * max_num_attr
    attr_list = rel_attr_list_dict[rel_name]
    for idx, attr in enumerate(attr_list):
        if attr in filter_line:
            arr[idx] = 1
    return arr

def get_seq_scan_input(plan_dict):
    # plan_dict: dict where the plan_dict['node_type'] = 'Seq Scan'
    rel_vec = get_rel_one_hot(plan_dict['Relation Name'])
    if 'Filter' not in plan_dict:
        rel_attr_vec = [0] * max_num_attr
    else:
        rel_attr_vec = get_rel_attr_one_hot(plan_dict['Relation Name'],
                                            plan_dict['Filter'])
    return get_basics(plan_dict) + rel_vec + rel_attr_vec

def get_hash_input(plan_dict):
    return get_basics(plan_dict) + [plan_dict['Hash Buckets']]

def get_join_input(plan_dict):
    type_vec = [0] * len(join_types)
    type_vec[join_types.index(plan_dict['Join Type']).lower()] = 1
    par_rel_vec = [0] * len(parent_rel_types)
    par_rel_vec[parent_rel_types.index(plan_dict['Parent Relationship'].lower())] = 1
    return get_basics(plan_dict) + type_vec + par_rel_vec

def get_sort_input(plan_dict):
    sort_meth = [0] * len(sort_algos)
    sort_meth[sort_algos.index(plan_dict['Sort Method'].lower())] = 1
    return get_basics(plan_dict) +  sort_meth

def get_aggreg_input(plan_dict):
    strat_vec = [0] * len(aggreg_strats)
    strat_vec[aggreg_strats.index(plan_dict['Strategy'].lower())] = 1
    partial_mode_vec = [0] if plan_dict['Parallel Aware'] == 'false' else [1]
    return get_basics(plan_dict) + strat_vec + partial_mode_vec

GET_INPUT = \
{
    "Hash Join": get_join_input,
    "Merge Join": get_join_input,
    "Seq Scan": get_seq_scan_input,
    "Sort": get_sort_input,
    "Hash": get_hash_input,
    "Aggregate": get_aggreg_input
}

GET_INPUT = collections.defaultdict(lambda: get_basics, GET_INPUT)

###############################################################################
#       Parsing data from csv files that contain json output of queries       #
###############################################################################
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

def create_dataset(data_dir):
    assert('res_by_temp/' == data_dir[-14:])
    fnames = os.listdir(data_dir)
    data = []
    for fname in fnames:
        data += get_all_plans(data_dir + fname)

    return data


def get_input(data): # Helper for sample_data
    """
    Parameter: data is a list of plan_dict; all entry is from the same
    query template and thus have the same query plan;

    Returns: a single plan dict of similar structure, where each node has
        node_type     ---- a string, same as before
        feat_vec      ---- numpy array of size (batch_size x feat_size)
        children_plan ---- a list of children's plan_dicts where each plan_dict
                           has feat_vec encompassing that child in all
                           co-plans
    """
    new_samp_dict = {}
    new_samp_dict["node_type"] = data[0]["Node Type"]
    feat_vec = [GET_INPUT[jss["Node Type"]](jss) for jss in data]
    total_time = [jss['Actual Total Time'] for jss in data]

    child_plan_lst = []
    if 'Plans' in data[0]:
        # get the children's dict containing feat_vecs
        #if jss['Node Type'] == 'Seq Scan':
        #    print(jss['Plans'])
        for i in range(len(data[0]['Plans'])):
            child_plan_dict = get_input([jss['Plans'][i] for jss in data])
            child_plan_lst.append(child_plan_dict)

    new_samp_dict["feat_vec"] = np.array(feat_vec).astype(np.float32)
    new_samp_dict["children_plan"] = child_plan_lst
    new_samp_dict["total_time"] = np.array(total_time).astype(np.float32)
    if 'Subplan Name' in data[0]:
        new_samp_dict['is_subplan'] = True
    else:
        new_samp_dict['is_subplan'] = False

    return new_samp_dict

###############################################################################
#       Sampling subbatch data from the dataset; total size is batch_size     #
###############################################################################
def sample_data(dataset, batch_size):
    # dataset: all queries used in training
    samp = np.random.choice(np.arange(len(dataset)), batch_size, replace=False)
    samp_group = [[]] * num_q
    for idx in samp:
        samp_group[idx // 32].append(dataset[idx])

    return [get_input(grp) for grp in samp_group]
