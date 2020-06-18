
# all operators used in tpc-h
all_dicts = ['Aggregate', 'Gather Merge', 'Sort', 'Seq Scan', 'Index Scan',
             'Bitmap Heap Scan', 'Bitmap Index Scan',
             'Limit', 'Hash Join', 'Hash', 'Nested Loop', 'Materialize',
             'Merge Join', 'Subquery Scan', 'Gather']

join_types = ['semi', 'inner', 'anti', 'full', 'right']

parent_rel_types = ['inner', 'outer', 'subquery']

sort_algos = ['quicksort', 'top-n heapsort']

aggreg_strats = ['plain', 'sorted', 'hashed']


rel_names = ['customer', 'lineitem', 'nation', 'orders', 'part', 'partsupp',
             'region', 'supplier']

index_names = ['c_ck', 'c_nk', 'p_pk', 's_sk', 's_nk', 'ps_pk', 'ps_sk',
               'ps_pk_sk', 'ps_sk_pk', 'o_ok', 'o_ck', 'o_od', 'l_ok', 'l_pk',
               'l_sk', 'l_sd', 'l_cd', 'l_rd', 'l_pk_sk', 'l_sk_pk', 'n_nk',
               'n_rk', 'r_rk']

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
