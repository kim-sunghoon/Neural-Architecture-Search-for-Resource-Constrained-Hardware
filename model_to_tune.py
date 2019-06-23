SIMPLE = [
    {'num_filters': 32, 'filter_height': 3, 'filter_width': 3,
     'pool_size': 1},
    {'num_filters': 32, 'filter_height': 3, 'filter_width': 3,
     'pool_size': 2},
    {'num_filters': 64, 'filter_height': 3, 'filter_width': 3,
     'pool_size': 1},
    {'num_filters': 64, 'filter_height': 3, 'filter_width': 3,
     'pool_size': 2},
    {'num_filters': 128, 'filter_height': 3, 'filter_width': 3,
     'pool_size': 1},
    {'num_filters': 128, 'filter_height': 3, 'filter_width': 3,
     'pool_size': 2}]

NAS15 = [
    {'filter_height': 3, 'filter_width': 3, 'num_filters': 36,  # 0
     'anchor_point': []},
    {'filter_height': 3, 'filter_width': 3, 'num_filters': 48,  # 1
     'anchor_point': [1]},
    {'filter_height': 3, 'filter_width': 3, 'num_filters': 36,  # 2
     'anchor_point': [1, 1]},
    {'filter_height': 5, 'filter_width': 5, 'num_filters': 36,  # 3
     'anchor_point': [1, 1, 1]},
    {'filter_height': 3, 'filter_width': 7, 'num_filters': 48,  # 4
     'anchor_point': [0, 0, 1, 1]},
    {'filter_height': 7, 'filter_width': 7, 'num_filters': 48,  # 5
     'anchor_point': [0, 1, 1, 1, 1]},
    {'filter_height': 7, 'filter_width': 7, 'num_filters': 48,  # 6
     'anchor_point': [0, 1, 1, 1, 1, 1]},
    {'filter_height': 7, 'filter_width': 3, 'num_filters': 36,  # 7
     'anchor_point': [1, 0, 0, 0, 0, 1, 1]},
    {'filter_height': 7, 'filter_width': 1, 'num_filters': 36,  # 8
     'anchor_point': [1, 0, 0, 0, 1, 1, 0, 1]},
    {'filter_height': 7, 'filter_width': 7, 'num_filters': 36,  # 9
     'anchor_point': [1, 0, 1, 1, 1, 1, 1, 1, 1]},
    {'filter_height': 5, 'filter_width': 7, 'num_filters': 36,  # 10
     'anchor_point': [1, 1, 0, 0, 1, 1, 1, 1, 1, 1]},
    {'filter_height': 7, 'filter_width': 7, 'num_filters': 48,  # 11
     'anchor_point': [1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1]},
    {'filter_height': 7, 'filter_width': 5, 'num_filters': 48,  # 12
     'anchor_point': [1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1]},
    {'filter_height': 7, 'filter_width': 5, 'num_filters': 48,  # 13
     'anchor_point': [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1]},
    {'filter_height': 7, 'filter_width': 5, 'num_filters': 48,  # 14
     'anchor_point': [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1]}]

paras = NAS15
# for layer in paras:
#     layer['act_num_int_bits'] = 1
#     layer['act_num_frac_bits'] = 4
#     layer['weight_num_int_bits'] = 2
#     layer['weight_num_frac_bits'] = 5

paras = [
    {'filter_height': 3, 'filter_width': 3,
     'stride_height': 1, 'stride_width': 1,
     'num_filters': 48, 'pool_size': 2,
     'act_num_int_bits': 2, 'act_num_frac_bits': 4,
     'weight_num_int_bits': 1, 'weight_num_frac_bits': 6},
    {'filter_height': 3, 'filter_width': 7,
     'stride_height': 1, 'stride_width': 2,
     'num_filters': 36, 'pool_size': 1,
     'act_num_int_bits': 2, 'act_num_frac_bits': 6,
     'weight_num_int_bits': 1, 'weight_num_frac_bits': 3},
    {'filter_height': 7, 'filter_width': 1,
     'stride_height': 3, 'stride_width': 1,
     'num_filters': 48, 'pool_size': 1,
     'act_num_int_bits': 3, 'act_num_frac_bits': 6,
     'weight_num_int_bits': 3, 'weight_num_frac_bits': 5},
    {'filter_height': 5, 'filter_width': 3,
     'stride_height': 1, 'stride_width': 2,
     'num_filters': 36, 'pool_size': 2,
     'act_num_int_bits': 1, 'act_num_frac_bits': 6,
     'weight_num_int_bits': 2, 'weight_num_frac_bits': 4},
    {'filter_height': 3, 'filter_width': 7,
     'stride_height': 1, 'stride_width': 1,
     'num_filters': 24, 'pool_size': 1,
     'act_num_int_bits': 3, 'act_num_frac_bits': 2,
     'weight_num_int_bits': 1, 'weight_num_frac_bits': 3},
    {'filter_height': 1, 'filter_width': 1,
     'stride_height': 2, 'stride_width': 1,
     'num_filters': 64, 'pool_size': 2,
     'act_num_int_bits': 2, 'act_num_frac_bits': 2,
     'weight_num_int_bits': 1, 'weight_num_frac_bits': 2}
     ]
