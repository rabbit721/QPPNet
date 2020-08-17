import collections, pickle
import pickle
import json
import numpy as np
import torch
from collections import Counter, defaultdict
from dataset.tpch_dataset.tpch_utils import TPCHDataSet

SCALE = 10

EPS = 0.001
TRAIN_TEST_SPLIT = 0.8

ALL_OPS = ['UPDATE', 'LIMIT', 'SORT_ITERATE', 'OP_DECIMAL_PLUS_OR_MINUS',
           'IDX_SCAN', 'AGG_ITERATE', 'INSERT', 'OP_INTEGER_COMPARE', 'OUTPUT',
           'AGG_BUILD', 'DELETE', 'OP_INTEGER_PLUS_OR_MINUS', 'SORT_BUILD']

###############################################################################
#       Parsing data from csv files that contain json output of queries       #
###############################################################################

class TPCCDataSet(TPCHDataSet):
    def __init__(self, opt):
        self.batch_size = opt.batch_size
        self.num_q = 1

        self.SCALE = SCALE
        self.input_func = None

        all_data = self.get_all_plans(opt.data_dir)
        print(" \n".join([str(ent) for ent in all_data[:10]]))
        enum, num_grp = self.grouping(all_data)

        count = Counter(enum)
        print(count)

        all_groups = [[] for j in range(num_grp)]
        for j, grp_idx in enumerate(enum):
            all_groups[grp_idx].append(all_data[j])

        self.grp_idxes = []
        train_data = []
        train_groups = [[] for j in range(num_grp)]
        test_groups = [[] for j in range(num_grp)]

        for idx, grp in enumerate(all_groups):
            all_samp_num = len(grp)
            num_sample_per_q = int(all_samp_num * TRAIN_TEST_SPLIT)
            train_data += grp[:num_sample_per_q]
            train_groups[idx] += grp[:num_sample_per_q]
            test_groups[idx] += grp[num_sample_per_q: all_samp_num]
            self.grp_idxes += [idx] * num_sample_per_q

        self.num_grps = [num_grp]

        print([len(grp) for grp in train_groups])

        self.dataset = train_data
        self.datasize = len(self.dataset)

        if not opt.test_time:
            self.mean_range_dict = self.normalize(train_groups)
            with open('mean_range_dict.pickle', 'wb') as f:
                pickle.dump(self.mean_range_dict, f)
        else:
            with open(opt.mean_range_dict, 'rb') as f:
                self.mean_range_dict = pickle.load(f)

        print(self.mean_range_dict)

        test_dataset = [self.get_input(grp) for grp in test_groups]
        self.test_dataset = test_dataset
        self.all_dataset = [self.get_input(grp) for grp in all_groups]

    def get_input(self, data): # Helper for sample_data
        """
            Vectorize the input of a list of plan_dicts that have the same query plan structure structure (of the same template/group)

            Args:
            - data: a list of plan_dict, each plan_dict correspond to a query plan in the dataset;
                    requires that all plan_dicts is of the same query template/group

            Returns:
            - new_samp_dict: a dictionary, where each level has the following attribute:
                -- node_type      : name of the operator that the pipeline corresponds to
                -- real_node_type : the pipeline name
                -- subbatch_size  : number of queries in data
                -- feat_vec       : a numpy array of shape (batch_size x feat_dim) that's
                                   the vectorized inputs for all queries in data
                -- children_plan  : list of dictionaries with each being an output of
                                   a recursive call to get_input on a child of current node
                -- total_time     : a vector of prediction target for each query in data
                -- is_subplan     : if the queries are subplans; defined only if the input query plans have children
        """
        new_samp_dict = {}

        new_samp_dict["node_type"] = data[0]["Node Type"]
        new_samp_dict["real_node_type"] = data[0]["Node Name"]
        new_samp_dict["subbatch_size"] = len(data)

        feat_vec = np.array([jss['Feature Vec'] for jss in data])

        # normalize feat_vec
        feat_vec = (feat_vec + EPS) / (self.mean_range_dict[new_samp_dict["node_type"]][0] + EPS)
        feat_vec += np.random.default_rng().normal(loc=0, scale=0.1, size=feat_vec.shape)

        total_time = [jss['Actual Total Time'] for jss in data]
        child_plan_lst = []
        if 'Plans' in data[0]:
            for i in range(len(data[0]['Plans'])):
                child_plan_dict = self.get_input([jss['Plans'][i] for jss in data])
                child_plan_dict['is_subplan'] = False
                child_plan_lst.append(child_plan_dict)

        new_samp_dict["feat_vec"] = np.array(feat_vec).astype(np.float32)
        new_samp_dict["children_plan"] = child_plan_lst
        new_samp_dict["total_time"] = np.array(total_time).astype(np.float32) / SCALE

        return new_samp_dict

    def normalize(self, train_groups): # compute the mean and std vec of each operator
        feat_vec_col = defaultdict(list)

        def parse_input(data):
            feat_vec = [jss['Feature Vec'] for jss in data]

            if 'Plans' in data[0]:
                for i in range(len(data[0]['Plans'])):
                    parse_input([jss['Plans'][i] for jss in data])
            feat_vec_col[data[0]["Node Type"]].append(np.array(feat_vec).astype(np.float32))

        for grp in train_groups:
            parse_input(grp)

        def cmp_mean_range(feat_vec_lst):
            if len(feat_vec_lst) == 0:
                return (0, 1)
            else:
                total_vec = np.concatenate(feat_vec_lst)
                return (np.mean(total_vec, axis=0),
                        np.max(total_vec, axis=0))

        mean_range_dict = {operator : cmp_mean_range(feat_vec_col[operator]) \
                           for operator in feat_vec_col}
        return mean_range_dict

    def get_all_plans(self, fname):
        jss = []
        currtree = {"Actual Total Time": 0}
        prev = (None, None)
        f = open(fname, 'r')
        lines = f.readlines()[1:]
        for line in lines:
            tokens = [tok.strip(' ') for tok in line.strip('\n').split(",")]
            if len(tokens) < 1:
                continue

            group = tuple(tokens[:2])

            if prev[0] is not None:
                if prev[0] != group[0] or prev[1] >= group[1]:
                    jss.append(currtree)
                    currtree = {"Actual Total Time": 0}
                else:
                    nexttree = {"Plans": [currtree]}
                    nexttree['Actual Total Time'] = currtree['Actual Total Time']
                    currtree = nexttree

            node_type = f"tpcc_q{tokens[0]}_p{tokens[1]}"
            all_feats = np.array(
                        [[ALL_OPS.index(feat_name)
                         for feat_name in tokens[4].split(';')]] +
                        [tok.split(';') for tok in tokens[5:11]]).astype(float).T

            # sort by operator index in list of ALL_OPS
            all_feats = all_feats[all_feats[:, 0].argsort()]

            real_node_type = ";".join([str(idx) for idx in all_feats[:, 0].tolist()])
            feat_vec = all_feats[:, 1:].reshape(-1).tolist()

            currtree['Node Name'] = node_type
            currtree['Node Type'] = real_node_type
            currtree['Actual Total Time'] += float(tokens[-1])
            currtree['Feature Vec'] = feat_vec

            prev = group
        # jss is a list of json-transformed dicts, one for each query
        return jss

    ###############################################################################
    #       Sampling subbatch data from the dataset; total size is batch_size     #
    ###############################################################################
    def sample_data(self):
        # self.dataset: all queries used in training
        samp = np.random.choice(np.arange(self.datasize), self.batch_size, replace=False)

        samp_group = [[] for j in range(self.num_grps[0])]
        for idx in samp:
            grp_idx = self.grp_idxes[idx]
            samp_group[grp_idx].append(self.dataset[idx])

        parsed_input = []
        for i, grp in enumerate(samp_group):

            if len(grp) != 0:
                parsed_input.append(self.get_input(grp))

        return parsed_input

    def evaluate(self):
        samp = np.random.choice(np.arange(self.datasize), self.batch_size, replace=False)
        #print(samp)
        samp_group = [[] for j in range(self.num_grps[0])]
        for idx in samp:
            grp_idx = self.grp_idxes[idx]
            samp_group[grp_idx].append(self.dataset[idx])

        parsed_input = []
        for i, grp in enumerate(samp_group):
            if len(grp) != 0:
                parsed_input.append(self.get_input(grp))

        return parsed_input
