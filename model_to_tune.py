import utility

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

# for layer in paras:
#     layer['act_num_int_bits'] = 1
#     layer['act_num_frac_bits'] = 4
#     layer['weight_num_int_bits'] = 2
#     layer['weight_num_frac_bits'] = 5

# 6 layer linear with stride and pooling 30000 LUT and 1000 throughput
# paras = [
#     {'filter_height': 3, 'filter_width': 3,
#      'stride_height': 1, 'stride_width': 1,
#      'num_filters': 48, 'pool_size': 2,
#      'act_num_int_bits': 2, 'act_num_frac_bits': 4,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 6},
#     {'filter_height': 3, 'filter_width': 7,
#      'stride_height': 1, 'stride_width': 2,
#      'num_filters': 36, 'pool_size': 1,
#      'act_num_int_bits': 2, 'act_num_frac_bits': 6,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 3},
#     {'filter_height': 7, 'filter_width': 1,
#      'stride_height': 3, 'stride_width': 1,
#      'num_filters': 48, 'pool_size': 1,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 6,
#      'weight_num_int_bits': 3, 'weight_num_frac_bits': 5},
#     {'filter_height': 5, 'filter_width': 3,
#      'stride_height': 1, 'stride_width': 2,
#      'num_filters': 36, 'pool_size': 2,
#      'act_num_int_bits': 1, 'act_num_frac_bits': 6,
#      'weight_num_int_bits': 2, 'weight_num_frac_bits': 4},
#     {'filter_height': 3, 'filter_width': 7,
#      'stride_height': 1, 'stride_width': 1,
#      'num_filters': 24, 'pool_size': 1,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 2,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 3},
#     {'filter_height': 1, 'filter_width': 1,
#      'stride_height': 2, 'stride_width': 1,
#      'num_filters': 64, 'pool_size': 2,
#      'act_num_int_bits': 2, 'act_num_frac_bits': 2,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 2}
#      ]

# 6 layer linear with stride and pooling 30000 LUT and 500 throughput
# paras = [
#     {'filter_height': 5, 'filter_width': 7,
#      'stride_height': 1, 'stride_width': 1,
#      'num_filters': 24, 'pool_size': 1,
#      'act_num_int_bits': 2, 'act_num_frac_bits': 6,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 6},
#     {'filter_height': 5, 'filter_width': 3,
#      'stride_height': 1, 'stride_width': 2,
#      'num_filters': 36, 'pool_size': 1,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 6,
#      'weight_num_int_bits': 3, 'weight_num_frac_bits': 4},
#     {'filter_height': 7, 'filter_width': 5,
#      'stride_height': 3, 'stride_width': 1,
#      'num_filters': 24, 'pool_size': 1,
#      'act_num_int_bits': 2, 'act_num_frac_bits': 5,
#      'weight_num_int_bits': 3, 'weight_num_frac_bits': 5},
#     {'filter_height': 5, 'filter_width': 1,
#      'stride_height': 1, 'stride_width': 1,
#      'num_filters': 48, 'pool_size': 1,
#      'act_num_int_bits': 1, 'act_num_frac_bits': 4,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 5},
#     {'filter_height': 3, 'filter_width': 5,
#      'stride_height': 1, 'stride_width': 1,
#      'num_filters': 64, 'pool_size': 2,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 5,
#      'weight_num_int_bits': 2, 'weight_num_frac_bits': 6},
#     {'filter_height': 3, 'filter_width': 1,
#      'stride_height': 2, 'stride_width': 1,
#      'num_filters': 64, 'pool_size': 1,
#      'act_num_int_bits': 1, 'act_num_frac_bits': 1,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 6}]

# 6 layer linear with stride and pooling 30000 LUT and 1000 throughput
# paras = [
#     {'filter_height': 3, 'filter_width': 5,
#      'stride_height': 2, 'stride_width': 2,
#      'num_filters': 24, 'pool_size': 1,
#      'act_num_int_bits': 2, 'act_num_frac_bits': 2,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 5},
#     {'filter_height': 3, 'filter_width': 1,
#      'stride_height': 1, 'stride_width': 1,
#      'num_filters': 64, 'pool_size': 2,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 3,
#      'weight_num_int_bits': 3, 'weight_num_frac_bits': 6},
#     {'filter_height': 3, 'filter_width': 5,
#      'stride_height': 1, 'stride_width': 2,
#      'num_filters': 48, 'pool_size': 1,
#      'act_num_int_bits': 2, 'act_num_frac_bits': 4,
#      'weight_num_int_bits': 3, 'weight_num_frac_bits': 5},
#     {'filter_height': 5, 'filter_width': 3,
#      'stride_height': 1, 'stride_width': 2,
#      'num_filters': 48, 'pool_size': 2,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 0,
#      'weight_num_int_bits': 2, 'weight_num_frac_bits': 5},
#     {'filter_height': 3, 'filter_width': 1,
#      'stride_height': 1, 'stride_width': 3,
#      'num_filters': 48, 'pool_size': 1,
#      'act_num_int_bits': 0, 'act_num_frac_bits': 3,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 6},
#     {'filter_height': 1, 'filter_width': 3,
#      'stride_height': 2, 'stride_width': 2,
#      'num_filters': 48, 'pool_size': 1,
#      'act_num_int_bits': 2, 'act_num_frac_bits': 4,
#      'weight_num_int_bits': 3, 'weight_num_frac_bits': 3}]


# 6 layer linear with stride and pooling 100000 LUT and 500 throughput
# paras = [
#     {'filter_height': 3, 'filter_width': 7,
#      'stride_height': 1, 'stride_width': 1,
#      'num_filters': 64, 'pool_size': 2,
#      'act_num_int_bits': 2, 'act_num_frac_bits': 6,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 4},
#     {'filter_height': 7, 'filter_width': 7,
#      'stride_height': 2, 'stride_width': 3,
#      'num_filters': 48, 'pool_size': 1,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 5,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 5},
#     {'filter_height': 3, 'filter_width': 5,
#      'stride_height': 2, 'stride_width': 1,
#      'num_filters': 64, 'pool_size': 1,
#      'act_num_int_bits': 1, 'act_num_frac_bits': 4,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 6},
#     {'filter_height': 1, 'filter_width': 3,
#      'stride_height': 1, 'stride_width': 1,
#      'num_filters': 64, 'pool_size': 1,
#      'act_num_int_bits': 1, 'act_num_frac_bits': 5,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 5},
#     {'filter_height': 5, 'filter_width': 1,
#      'stride_height': 1, 'stride_width': 1,
#      'num_filters': 48, 'pool_size': 2,
#      'act_num_int_bits': 2, 'act_num_frac_bits': 6,
#      'weight_num_int_bits': 2, 'weight_num_frac_bits': 3},
#     {'filter_height': 3, 'filter_width': 3,
#      'stride_height': 1, 'stride_width': 3,
#      'num_filters': 64, 'pool_size': 2,
#      'act_num_int_bits': 2, 'act_num_frac_bits': 6,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 4}]

# 6 layer linear with stride and pooling 100000 LUT and 1000 throughput
# paras = [
#     {'filter_height': 3, 'filter_width': 5,
#      'stride_height': 1, 'stride_width': 2,
#      'num_filters': 48, 'pool_size': 1,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 2,
#      'weight_num_int_bits': 2, 'weight_num_frac_bits': 4},
#     {'filter_height': 7, 'filter_width': 3,
#      'stride_height': 2, 'stride_width': 2,
#      'num_filters': 24, 'pool_size': 2,
#      'act_num_int_bits': 2, 'act_num_frac_bits': 3,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 3},
#     {'filter_height': 5, 'filter_width': 7,
#      'stride_height': 1, 'stride_width': 1,
#      'num_filters': 36, 'pool_size': 1,
#      'act_num_int_bits': 2, 'act_num_frac_bits': 0,
#      'weight_num_int_bits': 3, 'weight_num_frac_bits': 3},
#     {'filter_height': 7, 'filter_width': 1,
#      'stride_height': 2, 'stride_width': 1,
#      'num_filters': 24, 'pool_size': 1,
#      'act_num_int_bits': 2, 'act_num_frac_bits': 1,
#      'weight_num_int_bits': 3, 'weight_num_frac_bits': 3},
#     {'filter_height': 1, 'filter_width': 3,
#      'stride_height': 1, 'stride_width': 3,
#      'num_filters': 48, 'pool_size': 1,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 6,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 5},
#     {'filter_height': 3, 'filter_width': 1,
#      'stride_height': 1, 'stride_width': 2,
#      'num_filters': 48, 'pool_size': 1,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 1,
#      'weight_num_int_bits': 2, 'weight_num_frac_bits': 6}]

# 6 layer linear with stride and pooling 300000 LUT and 500 throughput
# paras = [
#     {'filter_height': 1, 'filter_width': 7,
#      'stride_height': 1, 'stride_width': 1,
#      'num_filters': 64, 'pool_size': 1,
#      'act_num_int_bits': 2, 'act_num_frac_bits': 6,
#      'weight_num_int_bits': 3, 'weight_num_frac_bits': 5},
#     {'filter_height': 7, 'filter_width': 5,
#      'stride_height': 1, 'stride_width': 1,
#      'num_filters': 36, 'pool_size': 1,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 2,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 6},
#     {'filter_height': 3, 'filter_width': 5,
#      'stride_height': 1, 'stride_width': 1,
#      'num_filters': 36, 'pool_size': 2,
#      'act_num_int_bits': 2, 'act_num_frac_bits': 5,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 6},
#     {'filter_height': 7, 'filter_width': 1,
#      'stride_height': 1, 'stride_width': 3,
#      'num_filters': 24, 'pool_size': 1,
#      'act_num_int_bits': 2, 'act_num_frac_bits': 5,
#      'weight_num_int_bits': 2, 'weight_num_frac_bits': 5},
#     {'filter_height': 7, 'filter_width': 3,
#      'stride_height': 1, 'stride_width': 1,
#      'num_filters': 64, 'pool_size': 1,
#      'act_num_int_bits': 2, 'act_num_frac_bits': 4,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 4},
#     {'filter_height': 1, 'filter_width': 7,
#      'stride_height': 3, 'stride_width': 1,
#      'num_filters': 64, 'pool_size': 1,
#      'act_num_int_bits': 1, 'act_num_frac_bits': 5,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 5}]

# 6 layer linear with stride and pooling 300000 LUT and 1000 throughput
# paras = [
#     {'filter_height': 5, 'filter_width': 5,
#      'stride_height': 1, 'stride_width': 1,
#      'num_filters': 64, 'pool_size': 1,
#      'act_num_int_bits': 2, 'act_num_frac_bits': 5,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 3},
#     {'filter_height': 1, 'filter_width': 7,
#      'stride_height': 1, 'stride_width': 1,
#      'num_filters': 24, 'pool_size': 1,
#      'act_num_int_bits': 1, 'act_num_frac_bits': 4,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 5},
#     {'filter_height': 3, 'filter_width': 5,
#      'stride_height': 1, 'stride_width': 1,
#      'num_filters': 64, 'pool_size': 2,
#      'act_num_int_bits': 1, 'act_num_frac_bits': 6,
#      'weight_num_int_bits': 2, 'weight_num_frac_bits': 5},
#     {'filter_height': 7, 'filter_width': 1,
#      'stride_height': 1, 'stride_width': 1,
#      'num_filters': 36, 'pool_size': 1,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 3,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 4},
#     {'filter_height': 3, 'filter_width': 7,
#      'stride_height': 2, 'stride_width': 3,
#      'num_filters': 48, 'pool_size': 1,
#      'act_num_int_bits': 1, 'act_num_frac_bits': 6,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 6},
#     {'filter_height': 5, 'filter_width': 7,
#      'stride_height': 2, 'stride_width': 2,
#      'num_filters': 24, 'pool_size': 1,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 1,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 5}]

# 6 layer linear without stride and pooling 30000 LUT and 500 throughput
# paras = [
#     {'filter_height': 1, 'filter_width': 3,
#      'num_filters': 24, 'pool_size': 1,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 2,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 3},
#     {'filter_height': 3, 'filter_width': 5,
#      'num_filters': 48, 'pool_size': 2,
#      'act_num_int_bits': 1, 'act_num_frac_bits': 4,
#      'weight_num_int_bits': 2, 'weight_num_frac_bits': 4},
#     {'filter_height': 5, 'filter_width': 7,
#      'num_filters': 36, 'pool_size': 1,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 4,
#      'weight_num_int_bits': 3, 'weight_num_frac_bits': 4},
#     {'filter_height': 1, 'filter_width': 3,
#      'num_filters': 64, 'pool_size': 2,
#      'act_num_int_bits': 1, 'act_num_frac_bits': 4,
#      'weight_num_int_bits': 2, 'weight_num_frac_bits': 4},
#     {'filter_height': 7, 'filter_width': 3,
#      'num_filters': 36, 'pool_size': 2,
#      'act_num_int_bits': 2, 'act_num_frac_bits': 5,
#      'weight_num_int_bits': 2, 'weight_num_frac_bits': 4},
#     {'filter_height': 5, 'filter_width': 1,
#      'num_filters': 24, 'pool_size': 2,
#      'act_num_int_bits': 1, 'act_num_frac_bits': 2,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 6}]

# 6 layer linear without stride and pooling 30000 LUT and 1000 throughput
# paras = [
#     {'filter_height': 5, 'filter_width': 3, 'num_filters': 48, 'pool_size': 1,
#      'act_num_int_bits': 2, 'act_num_frac_bits': 6,
#      'weight_num_int_bits': 2, 'weight_num_frac_bits': 6},
#     {'filter_height': 3, 'filter_width': 1, 'num_filters': 48, 'pool_size': 2,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 2,
#      'weight_num_int_bits': 2, 'weight_num_frac_bits': 5},
#     {'filter_height': 1, 'filter_width': 7, 'num_filters': 36, 'pool_size': 2,
#      'act_num_int_bits': 2, 'act_num_frac_bits': 5,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 5},
#     {'filter_height': 7, 'filter_width': 3, 'num_filters': 36, 'pool_size': 1,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 1,
#      'weight_num_int_bits': 2, 'weight_num_frac_bits': 4},
#     {'filter_height': 5, 'filter_width': 5, 'num_filters': 24, 'pool_size': 1,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 3,
#      'weight_num_int_bits': 3, 'weight_num_frac_bits': 4},
#     {'filter_height': 1, 'filter_width': 1, 'num_filters': 24, 'pool_size': 1,
#      'act_num_int_bits': 2, 'act_num_frac_bits': 5,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 6}]

# 6 layer linear without stride and pooling 100000 LUT and 500 throughput
# paras = [
#     {'filter_height': 7, 'filter_width': 3, 'num_filters': 48, 'pool_size': 2,
#      'act_num_int_bits': 2, 'act_num_frac_bits': 0,
#      'weight_num_int_bits': 3, 'weight_num_frac_bits': 6},
#     {'filter_height': 1, 'filter_width': 3, 'num_filters': 48, 'pool_size': 1,
#      'act_num_int_bits': 1, 'act_num_frac_bits': 6,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 5},
#     {'filter_height': 3, 'filter_width': 5, 'num_filters': 48, 'pool_size': 1,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 6,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 5},
#     {'filter_height': 5, 'filter_width': 3, 'num_filters': 64, 'pool_size': 1,
#      'act_num_int_bits': 1, 'act_num_frac_bits': 1,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 5},
#     {'filter_height': 7, 'filter_width': 7, 'num_filters': 48, 'pool_size': 2,
#      'act_num_int_bits': 2, 'act_num_frac_bits': 5,
#      'weight_num_int_bits': 3, 'weight_num_frac_bits': 4},
#     {'filter_height': 1, 'filter_width': 7, 'num_filters': 48, 'pool_size': 1,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 4,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 6}]

# 6 layer linear without stride and pooling 100000 LUT and 1000 throughput
# paras = [
#     {'filter_height': 5, 'filter_width': 1, 'num_filters': 48, 'pool_size': 1,
#      'act_num_int_bits': 1, 'act_num_frac_bits': 2,
#      'weight_num_int_bits': 3, 'weight_num_frac_bits': 3},
#     {'filter_height': 5, 'filter_width': 3, 'num_filters': 48, 'pool_size': 2,
#      'act_num_int_bits': 2, 'act_num_frac_bits': 1,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 5},
#     {'filter_height': 1, 'filter_width': 5, 'num_filters': 36, 'pool_size': 1,
#      'act_num_int_bits': 1, 'act_num_frac_bits': 6,
#      'weight_num_int_bits': 3, 'weight_num_frac_bits': 4},
#     {'filter_height': 7, 'filter_width': 7, 'num_filters': 64, 'pool_size': 2,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 4,
#      'weight_num_int_bits': 2, 'weight_num_frac_bits': 6},
#     {'filter_height': 7, 'filter_width': 3, 'num_filters': 64, 'pool_size': 2,
#      'act_num_int_bits': 1, 'act_num_frac_bits': 2,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 5},
#     {'filter_height': 5, 'filter_width': 3, 'num_filters': 48, 'pool_size': 1,
#      'act_num_int_bits': 1, 'act_num_frac_bits': 4,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 4}]

# 6 layer linear with stride and pooling nas network A1
# paras = [
#     {'filter_height': 3, 'filter_width': 3,
#      'stride_height': 1, 'stride_width': 1,
#      'num_filters': 64, 'pool_size': 1},
#     {'filter_height': 7, 'filter_width': 5,
#      'stride_height': 1, 'stride_width': 1,
#      'num_filters': 48, 'pool_size': 1},
#     {'filter_height': 5, 'filter_width': 5,
#      'stride_height': 2, 'stride_width': 1,
#      'num_filters': 48, 'pool_size': 1},
#     {'filter_height': 3, 'filter_width': 5,
#      'stride_height': 1, 'stride_width': 1,
#      'num_filters': 64, 'pool_size': 1},
#     {'filter_height': 5, 'filter_width': 7,
#      'stride_height': 1, 'stride_width': 1,
#      'num_filters': 36, 'pool_size': 1},
#     {'filter_height': 3, 'filter_width': 1,
#      'stride_height': 1, 'stride_width': 2,
#      'num_filters': 64, 'pool_size': 2}]

# 6 layer linear with stride and pooling nas network 2
# paras = [
#     {'filter_height': 3, 'filter_width': 3,
#      'stride_height': 1, 'stride_width': 1,
#      'num_filters': 64, 'pool_size': 1},
#     {'filter_height': 3, 'filter_width': 5,
#      'stride_height': 1, 'stride_width': 1,
#      'num_filters': 36, 'pool_size': 1},
#     {'filter_height': 5, 'filter_width': 5,
#      'stride_height': 1, 'stride_width': 1,
#      'num_filters': 48, 'pool_size': 1},
#     {'filter_height': 3, 'filter_width': 5,
#      'stride_height': 1, 'stride_width': 1,
#      'num_filters': 64, 'pool_size': 1},
#     {'filter_height': 5, 'filter_width': 5,
#      'stride_height': 3, 'stride_width': 1,
#      'num_filters': 48, 'pool_size': 1},
#     {'filter_height': 3, 'filter_width': 3,
#      'stride_height': 1, 'stride_width': 2,
#      'num_filters': 64, 'pool_size': 2}]

# 6 layer linear with stride and pooling nas network 3
# paras = [
#     {'filter_height': 3, 'filter_width': 3,
#      'stride_height': 1, 'stride_width': 1,
#      'num_filters': 24, 'pool_size': 1},
#     {'filter_height': 5, 'filter_width': 5,
#      'stride_height': 1, 'stride_width': 1,
#      'num_filters': 36, 'pool_size': 1},
#     {'filter_height': 5, 'filter_width': 5,
#      'stride_height': 2, 'stride_width': 1,
#      'num_filters': 64, 'pool_size': 1},
#     {'filter_height': 5, 'filter_width': 5,
#      'stride_height': 1, 'stride_width': 1,
#      'num_filters': 64, 'pool_size': 1},
#     {'filter_height': 5, 'filter_width': 5,
#      'stride_height': 1, 'stride_width': 2,
#      'num_filters': 24, 'pool_size': 1},
#     {'filter_height': 3, 'filter_width': 3,
#      'stride_height': 1, 'stride_width': 2,
#      'num_filters': 64, 'pool_size': 1}]

# 6 layer linear without stride with pooling nas network 1
# paras = [
#     {'filter_height': 3, 'filter_width': 7,
#      'num_filters': 64, 'pool_size': 1},
#     {'filter_height': 5, 'filter_width': 5,
#      'num_filters': 64, 'pool_size': 2},
#     {'filter_height': 1, 'filter_width': 3,
#      'num_filters': 48, 'pool_size': 1},
#     {'filter_height': 5, 'filter_width': 3,
#      'num_filters': 48, 'pool_size': 2},
#     {'filter_height': 7, 'filter_width': 7,
#      'num_filters': 64, 'pool_size': 1},
#     {'filter_height': 7, 'filter_width': 3,
#      'num_filters': 48, 'pool_size': 2}]

# 6 layers linear with stride and pooling 100000 LUT and 500 throughput
# paras = [
#     {'filter_height': 3, 'filter_width': 7,
#      'stride_height': 1, 'stride_width': 1,
#      'num_filters': 64, 'pool_size': 1,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 2,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 5},
#     {'filter_height': 5, 'filter_width': 3,
#      'stride_height': 1, 'stride_width': 2,
#      'num_filters': 48, 'pool_size': 1,
#      'act_num_int_bits': 2, 'act_num_frac_bits': 5,
#      'weight_num_int_bits': 3, 'weight_num_frac_bits': 3},
#     {'filter_height': 1, 'filter_width': 5,
#      'stride_height': 1, 'stride_width': 2,
#      'num_filters': 64, 'pool_size': 1,
#      'act_num_int_bits': 2, 'act_num_frac_bits': 4,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 4},
#     {'filter_height': 7, 'filter_width': 5,
#      'stride_height': 1, 'stride_width': 1,
#      'num_filters': 48, 'pool_size': 1,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 3,
#      'weight_num_int_bits': 3, 'weight_num_frac_bits': 6},
#     {'filter_height': 5, 'filter_width': 3,
#      'stride_height': 3, 'stride_width': 1,
#      'num_filters': 24, 'pool_size': 1,
#      'act_num_int_bits': 1, 'act_num_frac_bits': 3,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 5},
#     {'filter_height': 3, 'filter_width': 5,
#      'stride_height': 1, 'stride_width': 3,
#      'num_filters': 24, 'pool_size': 1,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 1,
#      'weight_num_int_bits': 2, 'weight_num_frac_bits': 5}]

# 6 layers linear with stride and pooling 100000 LUT and 1000 throughput
# paras = [
#     {'filter_height': 5, 'filter_width': 7,
#      'stride_height': 1, 'stride_width': 3,
#      'num_filters': 36, 'pool_size': 1,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 2,
#      'weight_num_int_bits': 3, 'weight_num_frac_bits': 4},
#     {'filter_height': 3, 'filter_width': 5,
#      'stride_height': 1, 'stride_width': 1,
#      'num_filters': 36, 'pool_size': 1,
#      'act_num_int_bits': 2, 'act_num_frac_bits': 1,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 3},
#     {'filter_height': 1, 'filter_width': 5,
#      'stride_height': 2, 'stride_width': 2,
#      'num_filters': 36, 'pool_size': 1,
#      'act_num_int_bits': 2, 'act_num_frac_bits': 0,
#      'weight_num_int_bits': 3, 'weight_num_frac_bits': 3},
#     {'filter_height': 7, 'filter_width': 3,
#      'stride_height': 1, 'stride_width': 2,
#      'num_filters': 48, 'pool_size': 2,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 4,
#      'weight_num_int_bits': 3, 'weight_num_frac_bits': 5},
#     {'filter_height': 5, 'filter_width': 3,
#      'stride_height': 1, 'stride_width': 2,
#      'num_filters': 36, 'pool_size': 1,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 6,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 6},
#     {'filter_height': 1, 'filter_width': 1,
#      'stride_height': 2, 'stride_width': 3,
#      'num_filters': 36, 'pool_size': 2,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 2,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 5}]

# 6 layers linear with stride and pooling 300000 LUT and 500 throughput
# paras = [
#     {'filter_height': 3, 'filter_width': 7,
#      'stride_height': 1, 'stride_width': 2,
#      'num_filters': 48, 'pool_size': 2,
#      'act_num_int_bits': 1, 'act_num_frac_bits': 4,
#      'weight_num_int_bits': 3, 'weight_num_frac_bits': 4},
#     {'filter_height': 5, 'filter_width': 7,
#      'stride_height': 2, 'stride_width': 2,
#      'num_filters': 24, 'pool_size': 1,
#      'act_num_int_bits': 2, 'act_num_frac_bits': 1,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 5},
#     {'filter_height': 7, 'filter_width': 1,
#      'stride_height': 1, 'stride_width': 1,
#      'num_filters': 48, 'pool_size': 1,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 2,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 4},
#     {'filter_height': 3, 'filter_width': 5,
#      'stride_height': 1, 'stride_width': 3,
#      'num_filters': 36, 'pool_size': 2,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 4,
#      'weight_num_int_bits': 3, 'weight_num_frac_bits': 4},
#     {'filter_height': 1, 'filter_width': 1,
#      'stride_height': 1, 'stride_width': 1,
#      'num_filters': 24, 'pool_size': 1,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 6,
#      'weight_num_int_bits': 2, 'weight_num_frac_bits': 3},
#     {'filter_height': 1, 'filter_width': 5,
#      'stride_height': 1, 'stride_width': 3,
#      'num_filters': 64, 'pool_size': 1,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 3,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 6}]

# 6 layers linear with stride and pooling 300000 LUT and 1000 throughput
# paras = [
#     {'filter_height': 5, 'filter_width': 5,
#      'stride_height': 1, 'stride_width': 1,
#      'num_filters': 64, 'pool_size': 1,
#      'act_num_int_bits': 2, 'act_num_frac_bits': 5,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 3},
#     {'filter_height': 1, 'filter_width': 7,
#      'stride_height': 1, 'stride_width': 1,
#      'num_filters': 24, 'pool_size': 1,
#      'act_num_int_bits': 1, 'act_num_frac_bits': 4,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 5},
#     {'filter_height': 3, 'filter_width': 5,
#      'stride_height': 1, 'stride_width': 1,
#      'num_filters': 64, 'pool_size': 2,
#      'act_num_int_bits': 1, 'act_num_frac_bits': 6,
#      'weight_num_int_bits': 2, 'weight_num_frac_bits': 5},
#     {'filter_height': 7, 'filter_width': 1,
#      'stride_height': 1, 'stride_width': 1,
#      'num_filters': 36, 'pool_size': 1,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 3,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 4},
#     {'filter_height': 3, 'filter_width': 7,
#      'stride_height': 2, 'stride_width': 3,
#      'num_filters': 48, 'pool_size': 1,
#      'act_num_int_bits': 1, 'act_num_frac_bits': 6,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 6},
#     {'filter_height': 5, 'filter_width': 7,
#      'stride_height': 2, 'stride_width': 2,
#      'num_filters': 24, 'pool_size': 1,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 1,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 5}]

# ns skip 300000 500
# paras = [
#     {'anchor_point': [], 'filter_height': 5, 'filter_width': 3,
#      'num_filters': 36, 'pool_size': 2,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 2,
#      'weight_num_int_bits': 3, 'weight_num_frac_bits': 5},
#     {'anchor_point': [1], 'filter_height': 7, 'filter_width': 5,
#      'num_filters': 64, 'pool_size': 2,
#      'act_num_int_bits': 2, 'act_num_frac_bits': 3,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 6},
#     {'anchor_point': [0, 1], 'filter_height': 7, 'filter_width': 5,
#      'num_filters': 48, 'pool_size': 2,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 2,
#      'weight_num_int_bits': 2, 'weight_num_frac_bits': 5},
#     {'anchor_point': [0, 0, 1], 'filter_height': 1, 'filter_width': 3,
#      'num_filters': 64, 'pool_size': 2,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 0,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 4},
#     {'anchor_point': [0, 1, 0, 0], 'filter_height': 1, 'filter_width': 1,
#      'num_filters': 48, 'pool_size': 2, 'act_num_int_bits': 3,
#      'act_num_frac_bits': 5, 'weight_num_int_bits': 2,
#      'weight_num_frac_bits': 6},
#     {'anchor_point': [0, 1, 1, 1, 1], 'filter_height': 1, 'filter_width': 1,
#      'num_filters': 24, 'pool_size': 2,
#      'act_num_int_bits': 1, 'act_num_frac_bits': 2,
#      'weight_num_int_bits': 2, 'weight_num_frac_bits': 5}]

# ns skip 300000 1000
# paras = [
#     {'anchor_point': [], 'filter_height': 5, 'filter_width': 7,
#      'num_filters': 24, 'pool_size': 2,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 6,
#      'weight_num_int_bits': 3, 'weight_num_frac_bits': 5},
#     {'anchor_point': [1], 'filter_height': 3, 'filter_width': 5,
#      'num_filters': 64, 'pool_size': 2,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 5,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 6},
#     {'anchor_point': [1, 0], 'filter_height': 5, 'filter_width': 3,
#      'num_filters': 36, 'pool_size': 2,
#      'act_num_int_bits': 1, 'act_num_frac_bits': 5,
#      'weight_num_int_bits': 3, 'weight_num_frac_bits': 5},
#     {'anchor_point': [1, 0, 0], 'filter_height': 7, 'filter_width': 7,
#      'num_filters': 48, 'pool_size': 1,
#      'act_num_int_bits': 2, 'act_num_frac_bits': 3,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 6},
#     {'anchor_point': [0, 0, 1, 0], 'filter_height': 1, 'filter_width': 1,
#      'num_filters': 48, 'pool_size': 2,
#      'act_num_int_bits': 1, 'act_num_frac_bits': 1,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 4},
#     {'anchor_point': [0, 1, 0, 1, 1], 'filter_height': 1, 'filter_width': 1,
#      'num_filters': 36, 'pool_size': 1,
#      'act_num_int_bits': 2, 'act_num_frac_bits': 4,
#      'weight_num_int_bits': 2, 'weight_num_frac_bits': 6}]

# ns skip 100000 500
# paras = [
#     {'anchor_point': [], 'filter_height': 5, 'filter_width': 5,
#      'num_filters': 64, 'pool_size': 2,
#      'act_num_int_bits': 2, 'act_num_frac_bits': 4,
#      'weight_num_int_bits': 2, 'weight_num_frac_bits': 6},
#     {'anchor_point': [1], 'filter_height': 7, 'filter_width': 5,
#      'num_filters': 36, 'pool_size': 1,
#      'act_num_int_bits': 2, 'act_num_frac_bits': 1,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 5},
#     {'anchor_point': [0, 0], 'filter_height': 7, 'filter_width': 3,
#      'num_filters': 64, 'pool_size': 2,
#      'act_num_int_bits': 2, 'act_num_frac_bits': 5,
#      'weight_num_int_bits': 2, 'weight_num_frac_bits': 4},
#     {'anchor_point': [1, 0, 0], 'filter_height': 1, 'filter_width': 7,
#      'num_filters': 48, 'pool_size': 2,
#      'act_num_int_bits': 1, 'act_num_frac_bits': 3,
#      'weight_num_int_bits': 3, 'weight_num_frac_bits': 6},
#     {'anchor_point': [1, 0, 1, 0], 'filter_height': 1, 'filter_width': 1,
#      'num_filters': 64, 'pool_size': 2,
#      'act_num_int_bits': 2, 'act_num_frac_bits': 4,
#      'weight_num_int_bits': 2, 'weight_num_frac_bits': 4},
#     {'anchor_point': [1, 1, 1, 1, 1], 'filter_height': 1, 'filter_width': 1,
#      'num_filters': 24, 'pool_size': 2,
#      'act_num_int_bits': 2, 'act_num_frac_bits': 3,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 5}]

# ns skip 300000 500
# paras = [
#     {'filter_height': 3, 'filter_width': 3, 'num_filters': 36, 'pool_size': 1,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 1,
#      'weight_num_int_bits': 2, 'weight_num_frac_bits': 6},
#     {'filter_height': 1, 'filter_width': 1, 'num_filters': 36, 'pool_size': 1,
#      'act_num_int_bits': 2, 'act_num_frac_bits': 4,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 5},
#     {'filter_height': 3, 'filter_width': 7, 'num_filters': 64, 'pool_size': 2,
#      'act_num_int_bits': 1, 'act_num_frac_bits': 4,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 6},
#     {'filter_height': 7, 'filter_width': 5, 'num_filters': 64, 'pool_size': 1,
#      'act_num_int_bits': 1, 'act_num_frac_bits': 5,
#      'weight_num_int_bits': 2, 'weight_num_frac_bits': 6},
#     {'filter_height': 7, 'filter_width': 5, 'num_filters': 64, 'pool_size': 2,
#      'act_num_int_bits': 1, 'act_num_frac_bits': 5,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 5},
#     {'filter_height': 5, 'filter_width': 5, 'num_filters': 24, 'pool_size': 2,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 6,
#      'weight_num_int_bits': 3, 'weight_num_frac_bits': 6}]

# ns skip 300000 1000
# paras = [
#     {'filter_height': 1, 'filter_width': 5, 'num_filters': 64, 'pool_size': 1,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 4,
#      'weight_num_int_bits': 3, 'weight_num_frac_bits': 4},
#     {'filter_height': 1, 'filter_width': 7, 'num_filters': 36, 'pool_size': 1,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 1,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 4},
#     {'filter_height': 5, 'filter_width': 7, 'num_filters': 64, 'pool_size': 2,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 5,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 5},
#     {'filter_height': 5, 'filter_width': 3, 'num_filters': 48, 'pool_size': 2,
#      'act_num_int_bits': 2, 'act_num_frac_bits': 3,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 4},
#     {'filter_height': 7, 'filter_width': 7, 'num_filters': 48, 'pool_size': 1,
#      'act_num_int_bits': 2, 'act_num_frac_bits': 2,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 4},
#     {'filter_height': 1, 'filter_width': 5, 'num_filters': 36, 'pool_size': 1,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 4,
#      'weight_num_int_bits': 2, 'weight_num_frac_bits': 3}]

# nas with stride with pooling 6 layers
# paras = [
#     {'filter_height': 3, 'filter_width': 5,
#      'stride_height': 1, 'stride_width': 1,
#      'num_filters': 48, 'pool_size': 1},
#     {'filter_height': 5, 'filter_width': 5,
#      'stride_height': 1, 'stride_width': 1,
#      'num_filters': 64, 'pool_size': 2},
#     {'filter_height': 5, 'filter_width': 7,
#      'stride_height': 1, 'stride_width': 1,
#      'num_filters': 48, 'pool_size': 1},
#     {'filter_height': 7, 'filter_width': 5,
#      'stride_height': 2, 'stride_width': 2,
#      'num_filters': 64, 'pool_size': 2},
#     {'filter_height': 5, 'filter_width': 7,
#      'stride_height': 2, 'stride_width': 2,
#      'num_filters': 48, 'pool_size': 1},
#     {'filter_height': 3, 'filter_width': 3,
#      'stride_height': 3, 'stride_width': 3,
#      'num_filters': 36, 'pool_size': 1}]

# joint skip no stride 300000/1000
# paras = [
#     {'anchor_point': [], 'filter_height': 5, 'filter_width': 7,
#      'num_filters': 24, 'pool_size': 2,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 6,
#      'weight_num_int_bits': 3, 'weight_num_frac_bits': 5},
#     {'anchor_point': [], 'filter_height': 3, 'filter_width': 5,
#      'num_filters': 64, 'pool_size': 2,
#      'act_num_int_bits': 3, 'act_num_frac_bits': 5,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 6},
#     {'anchor_point': [1], 'filter_height': 5, 'filter_width': 3,
#      'num_filters': 36, 'pool_size': 2,
#      'act_num_int_bits': 1, 'act_num_frac_bits': 5,
#      'weight_num_int_bits': 3, 'weight_num_frac_bits': 5},
#     {'anchor_point': [1], 'filter_height': 7, 'filter_width': 7,
#      'num_filters': 48, 'pool_size': 1,
#      'act_num_int_bits': 2, 'act_num_frac_bits': 3,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 6},
#     {'anchor_point': [0, 0, 1], 'filter_height': 1, 'filter_width': 1,
#      'num_filters': 48, 'pool_size': 2,
#      'act_num_int_bits': 1, 'act_num_frac_bits': 1,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 4},
#     {'anchor_point': [0, 1, 0, 1], 'filter_height': 1,
#      'filter_width': 1, 'num_filters': 36, 'pool_size': 1,
#      'act_num_int_bits': 2, 'act_num_frac_bits': 4,
#      'weight_num_int_bits': 2, 'weight_num_frac_bits': 6}]

# network A quantization 30000/500
# quan_paras = [
#     {'act_num_int_bits': 1, 'act_num_frac_bits': 0,
#      'weight_num_int_bits': 2, 'weight_num_frac_bits': 5},
#     {'act_num_int_bits': 0, 'act_num_frac_bits': 3,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 1},
#     {'act_num_int_bits': 0, 'act_num_frac_bits': 2,
#      'weight_num_int_bits': 2, 'weight_num_frac_bits': 0},
#     {'act_num_int_bits': 3, 'act_num_frac_bits': 1,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 5},
#     {'act_num_int_bits': 2, 'act_num_frac_bits': 4,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 0},
#     {'act_num_int_bits': 3, 'act_num_frac_bits': 2,
#      'weight_num_int_bits': 3, 'weight_num_frac_bits': 5}]

# network A quantization 100000/500
# quan_paras = [
#     {'act_num_int_bits': 1, 'act_num_frac_bits': 4,
#      'weight_num_int_bits': 3, 'weight_num_frac_bits': 5},
#     {'act_num_int_bits': 2, 'act_num_frac_bits': 1,
#      'weight_num_int_bits': 2, 'weight_num_frac_bits': 5},
#     {'act_num_int_bits': 1, 'act_num_frac_bits': 4,
#      'weight_num_int_bits': 2, 'weight_num_frac_bits': 4},
#     {'act_num_int_bits': 0, 'act_num_frac_bits': 6,
#      'weight_num_int_bits': 2, 'weight_num_frac_bits': 5},
#     {'act_num_int_bits': 3, 'act_num_frac_bits': 1,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 5},
#     {'act_num_int_bits': 2, 'act_num_frac_bits': 3,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 3}]

# network A quantiation 100000/1000
# quan_paras = [
#     {'act_num_int_bits': 0, 'act_num_frac_bits': 1,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 2},
#     {'act_num_int_bits': 2, 'act_num_frac_bits': 0,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 3},
#     {'act_num_int_bits': 2, 'act_num_frac_bits': 0,
#      'weight_num_int_bits': 2, 'weight_num_frac_bits': 4},
#     {'act_num_int_bits': 0, 'act_num_frac_bits': 3,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 3},
#     {'act_num_int_bits': 3, 'act_num_frac_bits': 6,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 3},
#     {'act_num_int_bits': 2, 'act_num_frac_bits': 6,
#      'weight_num_int_bits': 2, 'weight_num_frac_bits': 6}]

# network A quantiation 300000/5000
# quan_paras = [
#     {'act_num_int_bits': 3, 'act_num_frac_bits': 2,
#      'weight_num_int_bits': 2, 'weight_num_frac_bits': 6},
#     {'act_num_int_bits': 2, 'act_num_frac_bits': 4,
#      'weight_num_int_bits': 2, 'weight_num_frac_bits': 6},
#     {'act_num_int_bits': 3, 'act_num_frac_bits': 6,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 6},
#     {'act_num_int_bits': 3, 'act_num_frac_bits': 3,
#      'weight_num_int_bits': 3, 'weight_num_frac_bits': 5},
#     {'act_num_int_bits': 3, 'act_num_frac_bits': 5,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 6},
#     {'act_num_int_bits': 2, 'act_num_frac_bits': 4,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 6}]

# network A1 quantization 300000/1000
quan_paras = [
    {'act_num_int_bits': 3, 'act_num_frac_bits': 3,
     'weight_num_int_bits': 2, 'weight_num_frac_bits': 6},
    {'act_num_int_bits': 2, 'act_num_frac_bits': 1,
     'weight_num_int_bits': 0, 'weight_num_frac_bits': 6},
    {'act_num_int_bits': 1, 'act_num_frac_bits': 4,
     'weight_num_int_bits': 0, 'weight_num_frac_bits': 5},
    {'act_num_int_bits': 2, 'act_num_frac_bits': 4,
     'weight_num_int_bits': 2, 'weight_num_frac_bits': 5},
    {'act_num_int_bits': 2, 'act_num_frac_bits': 2,
     'weight_num_int_bits': 0, 'weight_num_frac_bits': 6},
    {'act_num_int_bits': 3, 'act_num_frac_bits': 1,
     'weight_num_int_bits': 0, 'weight_num_frac_bits': 4}]


A1 = [
    {'filter_height': 3, 'filter_width': 3,
     'stride_height': 1, 'stride_width': 1,
     'num_filters': 64, 'pool_size': 1},
    {'filter_height': 7, 'filter_width': 5,
     'stride_height': 1, 'stride_width': 1,
     'num_filters': 48, 'pool_size': 1},
    {'filter_height': 5, 'filter_width': 5,
     'stride_height': 2, 'stride_width': 1,
     'num_filters': 48, 'pool_size': 1},
    {'filter_height': 3, 'filter_width': 5,
     'stride_height': 1, 'stride_width': 1,
     'num_filters': 64, 'pool_size': 1},
    {'filter_height': 5, 'filter_width': 7,
     'stride_height': 1, 'stride_width': 1,
     'num_filters': 36, 'pool_size': 1},
    {'filter_height': 3, 'filter_width': 1,
     'stride_height': 1, 'stride_width': 2,
     'num_filters': 64, 'pool_size': 2}]

A2 = [
    {'filter_height': 3, 'filter_width': 3, 'stride_height': 1,
     'stride_width': 1, 'num_filters': 24, 'pool_size': 1},
    {'filter_height': 5, 'filter_width': 5, 'stride_height': 1,
     'stride_width': 1, 'num_filters': 36, 'pool_size': 1},
    {'filter_height': 5, 'filter_width': 5, 'stride_height': 2,
     'stride_width': 1, 'num_filters': 64, 'pool_size': 1},
    {'filter_height': 5, 'filter_width': 5, 'stride_height': 1,
     'stride_width': 1, 'num_filters': 64, 'pool_size': 1},
    {'filter_height': 5, 'filter_width': 5, 'stride_height': 1,
     'stride_width': 2, 'num_filters': 24, 'pool_size': 1},
    {'filter_height': 3, 'filter_width': 3, 'stride_height': 1,
     'stride_width': 2, 'num_filters': 64, 'pool_size': 1}]

B1 = [
    {'filter_height': 3, 'filter_width': 3, 'num_filters': 64, 'pool_size': 1},
    {'filter_height': 3, 'filter_width': 5, 'num_filters': 64, 'pool_size': 1},
    {'filter_height': 3, 'filter_width': 3, 'num_filters': 64, 'pool_size': 2},
    {'filter_height': 5, 'filter_width': 5, 'num_filters': 64, 'pool_size': 2},
    {'filter_height': 5, 'filter_width': 3, 'num_filters': 64, 'pool_size': 1},
    {'filter_height': 7, 'filter_width': 7, 'num_filters': 64, 'pool_size': 1}]

B2 = [
    {'filter_height': 5, 'filter_width': 3, 'num_filters': 64, 'pool_size': 1},
    {'filter_height': 3, 'filter_width': 5, 'num_filters': 64, 'pool_size': 1},
    {'filter_height': 3, 'filter_width': 5, 'num_filters': 64, 'pool_size': 2},
    {'filter_height': 5, 'filter_width': 5, 'num_filters': 64, 'pool_size': 2},
    {'filter_height': 5, 'filter_width': 3, 'num_filters': 64, 'pool_size': 1},
    {'filter_height': 7, 'filter_width': 7, 'num_filters': 64, 'pool_size': 1}]


# A2 30000/500
# quan_paras = [
#     {'act_num_int_bits': 2, 'act_num_frac_bits': 2,
#      'weight_num_int_bits': 2, 'weight_num_frac_bits': 6},
#     {'act_num_int_bits': 0, 'act_num_frac_bits': 2,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 3},
#     {'act_num_int_bits': 0, 'act_num_frac_bits': 2,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 5},
#     {'act_num_int_bits': 1, 'act_num_frac_bits': 3,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 3},
#     {'act_num_int_bits': 2, 'act_num_frac_bits': 3,
#      'weight_num_int_bits': 2, 'weight_num_frac_bits': 4},
#     {'act_num_int_bits': 3, 'act_num_frac_bits': 2,
#     'weight_num_int_bits': 1, 'weight_num_frac_bits': 6}]

# A2 100000/500
# quan_paras = [
#     {'act_num_int_bits': 3, 'act_num_frac_bits': 4,
#      'weight_num_int_bits': 3, 'weight_num_frac_bits': 5},
#     {'act_num_int_bits': 3, 'act_num_frac_bits': 5,
#      'weight_num_int_bits': 3, 'weight_num_frac_bits': 6},
#     {'act_num_int_bits': 2, 'act_num_frac_bits': 5, 'weight_num_int_bits': 2, 'weight_num_frac_bits': 4},
#     {'act_num_int_bits': 2, 'act_num_frac_bits': 1,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 4},
#     {'act_num_int_bits': 3, 'act_num_frac_bits': 6,
#      'weight_num_int_bits': 2, 'weight_num_frac_bits': 5},
#     {'act_num_int_bits': 3, 'act_num_frac_bits': 4,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 6}]

# A2 100000/1000
# quan_paras = [
#     {'act_num_int_bits': 2, 'act_num_frac_bits': 6,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 6},
#     {'act_num_int_bits': 0, 'act_num_frac_bits': 4,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 6},
#     {'act_num_int_bits': 0, 'act_num_frac_bits': 6,
#      'weight_num_int_bits': 2, 'weight_num_frac_bits': 5},
#     {'act_num_int_bits': 0, 'act_num_frac_bits': 5,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 4},
#     {'act_num_int_bits': 3, 'act_num_frac_bits': 5,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 4},
#     {'act_num_int_bits': 2, 'act_num_frac_bits': 4,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 3}]

# A2 300000/500
# quan_paras = [
    # {'act_num_int_bits': 3, 'act_num_frac_bits': 4,
    #  'weight_num_int_bits': 3, 'weight_num_frac_bits': 5},
    # {'act_num_int_bits': 2, 'act_num_frac_bits': 3,
    #  'weight_num_int_bits': 3, 'weight_num_frac_bits': 6},
    # {'act_num_int_bits': 2, 'act_num_frac_bits': 1,
    #  'weight_num_int_bits': 3, 'weight_num_frac_bits': 4},
    # {'act_num_int_bits': 3, 'act_num_frac_bits': 6,
    #  'weight_num_int_bits': 0, 'weight_num_frac_bits': 6},
    # {'act_num_int_bits': 3, 'act_num_frac_bits': 6,
    #  'weight_num_int_bits': 2, 'weight_num_frac_bits': 4},
    # {'act_num_int_bits': 3, 'act_num_frac_bits': 6,
    #  'weight_num_int_bits': 0, 'weight_num_frac_bits': 5}]

# A2 300000/1000
# quan_paras = [
#     {'act_num_int_bits': 3, 'act_num_frac_bits': 2,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 6},
#     {'act_num_int_bits': 1, 'act_num_frac_bits': 5,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 6},
#     {'act_num_int_bits': 1, 'act_num_frac_bits': 5,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 4},
#     {'act_num_int_bits': 1, 'act_num_frac_bits': 4,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 4},
#     {'act_num_int_bits': 2, 'act_num_frac_bits': 6,
#      'weight_num_int_bits': 2, 'weight_num_frac_bits': 6},
#     {'act_num_int_bits': 3, 'act_num_frac_bits': 6,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 4}]

# B1 30000/500
# quan_paras = [
#     {'act_num_int_bits': 0, 'act_num_frac_bits': 0,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 5},
#     {'act_num_int_bits': 2, 'act_num_frac_bits': 2,
#      'weight_num_int_bits': 2, 'weight_num_frac_bits': 1},
#     {'act_num_int_bits': 0, 'act_num_frac_bits': 4,
#      'weight_num_int_bits': 1, 'weight_num_frac_bits': 1},
#     {'act_num_int_bits': 1, 'act_num_frac_bits': 2,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 0} ,
#     {'act_num_int_bits': 0, 'act_num_frac_bits': 1,
#      'weight_num_int_bits': 0, 'weight_num_frac_bits': 3} ,
#     {'act_num_int_bits': 2, 'act_num_frac_bits': 1,
#      'weight_num_int_bits': 2, 'weight_num_frac_bits': 6}]

# B1 100000/500
quan_paras = [
    {'act_num_int_bits': 1, 'act_num_frac_bits': 4,
     'weight_num_int_bits': 2, 'weight_num_frac_bits': 5},
    {'act_num_int_bits': 3, 'act_num_frac_bits': 3,
     'weight_num_int_bits': 1, 'weight_num_frac_bits': 6},
    {'act_num_int_bits': 1, 'act_num_frac_bits': 2,
     'weight_num_int_bits': 0, 'weight_num_frac_bits': 4},
    {'act_num_int_bits': 3, 'act_num_frac_bits': 1,
     'weight_num_int_bits': 3, 'weight_num_frac_bits': 5},
    {'act_num_int_bits': 2, 'act_num_frac_bits': 1,
     'weight_num_int_bits': 0, 'weight_num_frac_bits': 4},
    {'act_num_int_bits': 2, 'act_num_frac_bits': 3,
     'weight_num_int_bits': 3, 'weight_num_frac_bits': 6}]

paras = utility.join_paras(B1, quan_paras)