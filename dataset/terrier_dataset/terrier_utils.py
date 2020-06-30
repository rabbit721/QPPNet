import collections, pickle
import pickle
import json
from dataset.tpch_dataset.tpch_utils import TPCHDataSet

with open('terrier_group_dict.json', 'r') as f:
    pname_group_dict = json.load(f)

def get_input_for_all(plan_dict):
    pname_group_dict[plan_dict["Node Type"]]

TR_GET_INPUT = collections.defaultdict(lambda: get_input_for_all)

###############################################################################
#       Parsing data from csv files that contain json output of queries       #
###############################################################################

class TerrierDataSet(TPCHDataSet):
    def __init__(self, opt):
        self.batch_size = opt.batch_size
        self.num_q = opt.num_q

        self.input_func = TR_GET_INPUT

        fnames = ["execution.csv"]
        data = []
        all_groups_test = []

        all_data = self.get_all_plans(opt.data_dir + fname)
            #print(temp_data)

        ##### this is for train #####
        enum, num_grp = self.grouping(all_data)
        self.grp_idxes = enum
        self.num_grp = num_grp

        groups = [[] for j in range(num_grp)]
        for j, grp_idx in enumerate(enum):
            groups[grp_idx].append(temp_data[j])

        ##### this is for test #####
        all_groups_test = []
        train_data = []

        for grp in groups:
            all_groups_test.append(grp[int(len(grp)*0.9):])
            data += grp[:int(len(grp)*0.9)]


        data = [grp[:int(len(grp)*0.9)] for grp in groups]

        all_groups_test = groups

        #print(num_grp)

        self.dataset = data
        self.datasize = len(self.dataset)

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

        def parse_input(data):
            feat_vec = [self.input_func[data[0]["Node Type"]](jss) for jss in data]
            if 'Plans' in data[0]:
                for i in range(len(data[0]['Plans'])):
                    parse_input([jss['Plans'][i] for jss in data])
            feat_vec_col[data[0]["Node Type"]].append(np.array(feat_vec).astype(np.float32))

        for i in range(self.datasize // self.num_sample_per_q):
            try:
                groups = [[] for j in range(self.num_grp)]
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
        jss = []
        currtree = dict()
        prev = None
        f = open(fname, 'r')
        lines = f.readlines()
        lines.reverse()
        for line in lines:
            tokens = line.strip('\n').split(",")
            if len(tokens[0].split('_')) < 2:
                continue
            group = tokens[0].split('_')[1]
            if prev is not None:
                if prev != group:
                    jss.append(currtree)
                    currtree = dict()
                else:
                    currtree = {"Plans": [currtree]}
            currtree['Node Type'] = tokens[0]
            currtree['Actual Total Time'] = tokens[-1]
            prev = group
        # jss is a list of json-transformed dicts, one for each query
        return jss

    def grouping(self, data):
        counter = 0
        string_hash = []
        enum = []
        for plan_dict in data:
            enum.append(pname_group_dict[plan_dict['Node Type']])
        return enum, max(enum)
