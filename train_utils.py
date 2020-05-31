import numpy as np
import collections, os, json, pickle
from attr_rel_dict import *

# need a huge array
# get the columns of all relations
num_rel = 8
max_num_attr = 16

# need to normalize Plan Width, Plan Rows, Total Cost, Hash Bucket
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
    type_vec[join_types.index(plan_dict['Join Type'].lower())] = 1
    par_rel_vec = [0] * len(parent_rel_types)
    if 'Parent Relationship' in plan_dict:
        par_rel_vec[parent_rel_types.index(plan_dict['Parent Relationship'].lower())] = 1
    return get_basics(plan_dict) + type_vec + par_rel_vec

def get_sort_key_input(plan_dict):
    kys = plan_dict['Sort Key']
    one_hot = [0] * (num_rel * max_num_attr)
    for key in kys:
        key = key.replace('(', ' ').replace(')', ' ')
        for subkey in key.split(" "):
            if subkey != ' ' and '.' in subkey:
                rel_name, attr_name = subkey.split(' ')[0].split('.')
                one_hot[rel_names.index(rel_name) * max_num_attr
                        + rel_attr_list_dict[rel_name].index(attr_name.lower())] = 1

    return one_hot

def get_sort_input(plan_dict):
    sort_meth = [0] * len(sort_algos)
    sort_meth[sort_algos.index(plan_dict['Sort Method'].lower())] = 1
    return get_basics(plan_dict) + get_sort_key_input(plan_dict) + sort_meth

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

class DataSet():
    def __init__(self, data_dir, opt):
        fnames = os.listdir(data_dir)
        data = []
        for fname in fnames:
            if 'csv' in fname:
              data += self.get_all_plans(data_dir + fname)
        self.dataset = data
        self.datasize = len(self.dataset)
        self.num_sample_per_q = opt.num_sample_per_q
        self.num_q = opt.num_q

        if not opt.test:
            self.mean_range_dict = self.normalize()
            with open('mean_range_dict.pickle', 'wb') as f:
                pickle.dump(self.mean_range_dict, f)
        else:
            with open(opt.mean_range_dict, 'rb') as f:
                self.mean_range_dict = pickle.load(f)

        print(self.mean_range_dict)
        self.batch_size = opt.batch_size

    def normalize(self): # compute the mean and std vec of each operator
        feat_vec_col = {operator : [] for operator in all_dicts}

        def parse_input(data): # Helper for sample_data
            feat_vec = [GET_INPUT[data[0]["Node Type"]](jss) for jss in data]
            if 'Plans' in data[0]:
                for i in range(len(data[0]['Plans'])):
                    parse_input([jss['Plans'][i] for jss in data])
            feat_vec_col[data[0]["Node Type"]].append(np.array(feat_vec).astype(np.float32))

        for i in range(self.datasize // self.num_sample_per_q):
            parse_input(self.dataset[i*self.num_sample_per_q:(i+1)*self.num_sample_per_q])

        def cmp_mean_range(feat_vec_lst):
          if len(feat_vec_lst) == 0:
            return (0, 1)
          else:
            total_vec = np.concatenate(feat_vec_lst)
            return (np.mean(total_vec, axis=0),
                    np.max(total_vec, axis=0)+np.finfo(np.float32).eps)

        mean_range_dict = {operator : cmp_mean_range(feat_vec_col[operator]) \
                         for operator in all_dicts}
        return mean_range_dict

    def get_all_plans(self, fname):
        jsonstrs = []
        curr = ""
        prev = None
        prevprev = None
        with open(fname,'r') as f:
            for row in f:
                newrow = row.replace('+', "").replace("(1 row)\n", "").strip('\n').strip(' ')
                if 'CREATE' not in newrow and 'DROP' not in newrow and 'Tim' != newrow[:3]:
                    curr += newrow
                if prevprev is not None and 'Execution Time' in prevprev:
                    jsonstrs.append(curr.strip(' ').strip('QUERY PLAN').strip('-'))
                    curr = ""
                prevprev = prev
                prev = newrow
        strings = [s for s in jsonstrs if s[-1] == ']']
        jss = [json.loads(s)[0]['Plan'] for s in strings]
        # jss is a list of json-transformed dicts, one for each query
        return jss

    def get_input(self, data, i): # Helper for sample_data
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

        # normalize feat_vec
        feat_vec = (feat_vec -
                    self.mean_range_dict[new_samp_dict["node_type"]][0]) \
                    / self.mean_range_dict[new_samp_dict["node_type"]][1]


        total_time = [jss['Actual Total Time'] for jss in data]
        child_plan_lst = []
        if 'Plans' in data[0]:
            for i in range(len(data[0]['Plans'])):
                child_plan_dict = self.get_input([jss['Plans'][i] for jss in data], 'dum')
                child_plan_lst.append(child_plan_dict)

        #print(i, [d["Node Type"] for d in data], feat_vec)
        new_samp_dict["feat_vec"] = np.array(feat_vec).astype(np.float32)
        new_samp_dict["children_plan"] = child_plan_lst
        new_samp_dict["total_time"] = np.array(total_time).astype(np.float32) / 10

        if 'Subplan Name' in data[0]:
            new_samp_dict['is_subplan'] = True
        else:
            new_samp_dict['is_subplan'] = False
        return new_samp_dict

    ###############################################################################
    #       Sampling subbatch data from the dataset; total size is batch_size     #
    ###############################################################################
    def sample_data(self):
        # dataset: all queries used in training
        samp = np.random.choice(np.arange(self.datasize), self.batch_size, replace=False)
        #print(samp)
        samp_group = [[] for _ in range(self.num_q)]
        for idx in samp:
            # assuming we have 32 queries from each template
            samp_group[idx // self.num_sample_per_q].append(self.dataset[idx])

        return [self.get_input(grp, i) for i, grp in enumerate(samp_group) if len(grp) != 0]
