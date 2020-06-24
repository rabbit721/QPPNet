import numpy as np
import collections, os, json, pickle
from attr_rel_dict import *
import pickle

# need a huge array
# get the columns of all relations
num_rel = 8
max_num_attr = 16
num_index = 23
SCALE = 100

with open('attr_val_dict.pickle', 'rb') as f:
    attr_val_dict = pickle.load(f)


# need to normalize Plan Width, Plan Rows, Total Cost, Hash Bucket
def get_basics(plan_dict):
    return [plan_dict['Plan Width'], plan_dict['Plan Rows'],
            plan_dict['Total Cost']]

def get_rel_one_hot(rel_name):
    arr = [0] * num_rel
    arr[rel_names.index(rel_name)] = 1
    return arr

def get_index_one_hot(index_name):
    arr = [0] * num_index
    arr[index_names.index(index_name)] = 1
    return arr

def get_rel_attr_one_hot(rel_name, filter_line):
    attr_list = rel_attr_list_dict[rel_name]

    med_vec, min_vec, max_vec = [0] * max_num_attr, [0] * max_num_attr, \
                                [0] * max_num_attr

    for idx, attr in enumerate(attr_list):
        if attr in filter_line:
            med_vec[idx] = attr_val_dict['med'][rel_name][idx]
            min_vec[idx] = attr_val_dict['min'][rel_name][idx]
            max_vec[idx] = attr_val_dict['max'][rel_name][idx]
    return min_vec + med_vec + max_vec

def get_scan_input(plan_dict):
    # plan_dict: dict where the plan_dict['node_type'] = 'Seq Scan'
    rel_vec = get_rel_one_hot(plan_dict['Relation Name'])
    try:
        rel_attr_vec = get_rel_attr_one_hot(plan_dict['Relation Name'],
                                            plan_dict['Filter'])
    except:
        try:
            rel_attr_vec = get_rel_attr_one_hot(plan_dict['Relation Name'],
                                                plan_dict['Recheck Cond'])
        except:
            if 'Filter' in plan_dict:
                print('************************* default *************************')
                print(plan_dict)
            rel_attr_vec = [0] * max_num_attr * 3

    return get_basics(plan_dict) + rel_vec + rel_attr_vec


def get_index_scan_input(plan_dict):
    # plan_dict: dict where the plan_dict['node_type'] = 'Index Scan'

    rel_vec = get_rel_one_hot(plan_dict['Relation Name'])
    index_vec = get_index_one_hot(plan_dict['Index Name'])

    try:
        rel_attr_vec = get_rel_attr_one_hot(plan_dict['Relation Name'],
                                            plan_dict['Index Cond'])
    except:
        if 'Index Cond' in plan_dict:
            print('********************* default rel_attr_vec *********************')
            print(plan_dict)
        rel_attr_vec = [0] * max_num_attr * 3

    res = get_basics(plan_dict) + rel_vec + rel_attr_vec + index_vec \
          + [1 if plan_dict['Scan Direction'] == 'Forward' else 0]

    return res

def get_bitmap_index_scan_input(plan_dict):
    # plan_dict: dict where the plan_dict['node_type'] = 'Bitmap Index Scan'
    index_vec = get_index_one_hot(plan_dict['Index Name'])

    return get_basics(plan_dict) + index_vec

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
                if rel_name in rel_names:
                    one_hot[rel_names.index(rel_name) * max_num_attr
                            + rel_attr_list_dict[rel_name].index(attr_name.lower())] = 1

    return one_hot

def get_sort_input(plan_dict):
    sort_meth = [0] * len(sort_algos)
    if 'Sort Method' in plan_dict:
        if "external" not in plan_dict['Sort Method'].lower():
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
    "Seq Scan": get_scan_input,
    "Index Scan": get_index_scan_input,
    "Index Only Scan": get_index_scan_input,
    "Bitmap Heap Scan": get_scan_input,
    "Bitmap Index Scan": get_bitmap_index_scan_input,
    "Sort": get_sort_input,
    "Hash": get_hash_input,
    "Aggregate": get_aggreg_input
}

GET_INPUT = collections.defaultdict(lambda: get_basics, GET_INPUT)

###############################################################################
#       Parsing data from csv files that contain json output of queries       #
###############################################################################

class DataSet():
    def __init__(self, opt):
        self.num_sample_per_q = int(opt.num_sample_per_q * 0.9)
        self.batch_size = opt.batch_size
        self.num_q = opt.num_q

        fnames = [fname for fname in os.listdir(opt.data_dir) if 'csv' in fname]
        fnames = sorted(fnames,
                        key=lambda fname: int(fname.split('temp')[1][:-4]))

        data = []
        all_groups_test = []

        self.grp_idxes = []
        self.num_grps = [0] * self.num_q
        for i, fname in enumerate(fnames):
            temp_data = self.get_all_plans(opt.data_dir + fname)
            #print(temp_data)
            data += temp_data[:self.num_sample_per_q]

            ##### this is for train #####
            enum, num_grp = self.grouping(temp_data[:self.num_sample_per_q])
            self.grp_idxes += enum
            self.num_grps[i] = num_grp

            ##### this is for test #####
            enum, num_grp = self.grouping(temp_data[self.num_sample_per_q:])
            groups = [[] for j in range(num_grp)]
            for j, grp_idx in enumerate(enum):
                groups[grp_idx].append(temp_data[self.num_sample_per_q+j])
            all_groups_test += groups

            #print(num_grp)

        self.dataset = data
        self.datasize = len(self.dataset)
        print(self.num_grps)

        if not opt.test_time:
            self.mean_range_dict = self.normalize()

            with open('mean_range_dict.pickle', 'wb') as f:
                pickle.dump(self.mean_range_dict, f)
        else:
            with open(opt.mean_range_dict, 'rb') as f:
                self.mean_range_dict = pickle.load(f)

        print(self.mean_range_dict)

        test_dataset = [self.get_input(grp, 'dum') for grp in all_groups_test]
        self.test_dataset = test_dataset

    def normalize(self): # compute the mean and std vec of each operator
        feat_vec_col = {operator : [] for operator in all_dicts}

        def parse_input(data): # Helper for sample_data
            feat_vec = [GET_INPUT[data[0]["Node Type"]](jss) for jss in data]
            if 'Plans' in data[0]:
                for i in range(len(data[0]['Plans'])):
                    parse_input([jss['Plans'][i] for jss in data])
            feat_vec_col[data[0]["Node Type"]].append(np.array(feat_vec).astype(np.float32))

        for i in range(self.datasize // self.num_sample_per_q):
            try:
                if self.num_grps[i] == 1:
                    parse_input(self.dataset[i*self.num_sample_per_q:(i+1)*self.num_sample_per_q])
                else:
                    groups = [[] for j in range(self.num_grps[i])]
                    offset = i*self.num_sample_per_q
                    for j, plan_dict in enumerate(self.dataset[offset:offset+self.num_sample_per_q]):
                        groups[self.grp_idxes[offset + j]].append(plan_dict)
                    for grp in groups:
                        parse_input(grp)
            except:
                print('i: {}'.format(i))

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
                if not ('[' in row or '{' in row or ']' in row or '}' in row \
                        or ':' in row):
                    continue
                newrow = row.replace('+', "").replace("(1 row)\n", "").strip('\n').strip(' ')
                if 'CREATE' not in newrow and 'DROP' not in newrow and 'Tim' != newrow[:3]:
                    curr += newrow
                if prevprev is not None and 'Execution Time' in prevprev:
                    jsonstrs.append(curr.strip(' ').strip('QUERY PLAN').strip('-'))
                    curr = ""
                prevprev = prev
                prev = newrow
        #print(prevprev, prev)
        strings = [s for s in jsonstrs if s[-1] == ']']
        jss = [json.loads(s)[0]['Plan'] for s in strings]
        # jss is a list of json-transformed dicts, one for each query
        return jss

    def grouping(self, data):
        def hash(plan_dict):
            res = plan_dict['Node Type']
            if 'Plans' in plan_dict:
                for chld in plan_dict['Plans']:
                    res += hash(chld)
            return res
        counter = 0
        string_hash = []
        enum = []
        for plan_dict in data:
            string = hash(plan_dict)
            #print(string)
            try:
                idx = string_hash.index(string)
                enum.append(idx)
            except:
                idx = counter
                counter += 1
                enum.append(idx)
                string_hash.append(string)
        #print(string_hash, counter)
        assert(counter>0)
        return enum, counter

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
        new_samp_dict["subbatch_size"] = len(data)
        feat_vec = np.array([GET_INPUT[jss["Node Type"]](jss) for jss in data])

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
        new_samp_dict["total_time"] = np.array(total_time).astype(np.float32) / SCALE

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
        samp_group = [[[] for j in range(self.num_grps[i])]
                                for i in range(self.num_q)]
        for idx in samp:
            # assuming we have 32 queries from each template
            grp_idx = self.grp_idxes[idx]
            samp_group[idx // self.num_sample_per_q][grp_idx].append(self.dataset[idx])

        parsed_input = []
        for i, temp in enumerate(samp_group):
            for grp in temp:
                if len(grp) != 0:
                    parsed_input.append(self.get_input(grp, i))
        #print(parsed_input)
        return parsed_input

    '''
    def create_test_data(self, opt):
        fnames = [fname for fname in os.listdir(opt.test_data_dir) if 'csv' in fname]
        fnames = sorted(fnames,
                        key=lambda fname: int(fname.split('temp')[1][:-4]))
        data = []
        all_groups = []
        for i, fname in enumerate(fnames):
            temp_data = self.get_all_plans(opt.test_data_dir + fname)
            enum, num_grp = self.grouping(temp_data)
            groups = [[] for j in range(num_grp)]
            for j, grp_idx in enumerate(enum):
                groups[grp_idx].append(temp_data[j])
            all_groups += groups
        test_dataset = [self.get_input(grp, 'dum') for grp in all_groups]
        return test_dataset
    '''
