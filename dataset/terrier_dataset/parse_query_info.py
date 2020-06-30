

import terrier_query_info as tqi
import json
'''
col_set = set()
new_dict = {}
ct = 0
for pname in terrier_group_dict:
    var = pname.split("tpch")[1].upper()
    feat_lst = getattr(tqi, var)
    col = sorted([str(op) for op, vec in feat_lst])
    if str(col) in col_set:
        new_dict[pname] = ct - 1
    else:
        col_set.add(str(col))
        ct += 1
        new_dict[pname] = ct - 1
with open("terrier_group_dict.json", "w+") as f:
    json.dump(new_dict, f)

print(new_dict)
'''

with open("terrier_group_dict.json", "r") as f:
    group_dict = json.load(f)

input_len_dict = dict()
mem_map = getattr(tqi, "MEM_ADJUST_MAP")
for pname in group_dict:
    plst = getattr(tqi, pname.split("tpch")[1].upper())
    input_len_dict[pname] = len(plst) * 4 + (1 if pname in mem_map else 0)
    if int(pname.split('p')[2]) > 1:
        input_len_dict[pname] += 32

with open("input_dim_dict.json", "w+") as f:
    json.dump(input_len_dict, f, indent=4)
# print((col_set))
# print(ct)
