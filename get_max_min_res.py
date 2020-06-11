rel_attr_list_dict = \
{
    'customer':
        ['c_custkey',
         'c_name',
         'c_address',
         'c_nationkey',
         'c_phone',
         'c_acctbal',
         'c_mktsegment',
         'c_comment'],
    'lineitem':
        ['l_orderkey',
         'l_partkey',
         'l_suppkey',
         'l_linenumber',
         'l_quantity',
         'l_extendedprice',
         'l_discount',
         'l_tax',
         'l_returnflag',
         'l_linestatus',
         'l_shipdate',
         'l_commitdate',
         'l_receiptdate',
         'l_shipinstruct',
         'l_shipmode',
         'l_comment'],
    'nation':
        ['n_nationkey',
         'n_name',
         'n_regionkey',
         'n_comment'],
    'orders':
        ['o_orderkey',
         'o_custkey',
         'o_orderstatus',
         'o_totalprice',
         'o_orderdate',
         'o_orderpriority',
         'o_clerk',
         'o_shippriority',
         'o_comment'],
    'part':
        ['p_partkey',
         'p_name',
         'p_mfgr',
         'p_brand',
         'p_type',
         'p_size',
         'p_container',
         'p_retailprice',
         'p_comment'],
    'partsupp':
        ['ps_partkey',
         'ps_suppkey',
         'ps_availqty',
         'ps_supplycost',
         'ps_comment'],
    'region':
        ['r_regionkey',
         'r_name',
         'r_comment'],
    'supplier':
        ['s_suppkey',
         's_name',
         's_address',
         's_nationkey',
         's_phone',
         's_acctbal',
         's_comment']
}

'''

with open('attr_queries_min.sql', 'w+') as f:
    for table, cols in rel_attr_list_dict.items():
        for col in cols:
            f.write("select min({}) from {};\n".format(col, table))


with open('attr_queries_max.sql', 'w+') as f:
    for table, cols in rel_attr_list_dict.items():
        for col in cols:
            f.write("select max({}) from {};\n".format(col, table))

with open('attr_queries_med.sql', 'w+') as f:
    for table, cols in rel_attr_list_dict.items():
        for col in cols:
            f.write("select percentile_disc(0.5) within group (order by {}) from {};\n".format(col, table))
'''

def convert(input):
    try:
        res = float(input)
    except:
        res = 0
    return res


med_dict, min_dict, max_dict = dict(), dict(), dict()

with open('attr_max_min_med/max_attrs.txt', 'r') as f:
    lines = [convert(line) for line in f]
    counter = 0
    for table, cols in rel_attr_list_dict.items():
        max_dict[table] = lines[counter: counter + len(cols)]
        counter += len(cols)


with open('attr_max_min_med/min_attrs.txt', 'r') as f:
    lines = [convert(line) for line in f]
    counter = 0
    for table, cols in rel_attr_list_dict.items():
        min_dict[table] = lines[counter: counter + len(cols)]
        counter += len(cols)

with open('attr_max_min_med/med_attrs.txt', 'r') as f:
    lines = [convert(line) for line in f]
    counter = 0
    for table, cols in rel_attr_list_dict.items():
        med_dict[table] = lines[counter: counter + len(cols)]
        counter += len(cols)

attr_val_dict = {'med':med_dict, 'min': min_dict, 'max' : max_dict}

import pickle
with open('attr_val_dict.pickle', 'wb') as f:
    pickle.dump(attr_val_dict, f)
