ARCH_SPACE = {
    "filter_height": (1, 3, 5),
    "filter_width": (1, 3, 5),
    'stride_height': (1, 2),
    'stride_width': (1, 2),
    "num_filters": (24, 32, 40, 48, 64),
    "pool_size": (1, 2)
    }

QUAN_SPACE = {
    "act_num_int_bits": (0, 1, 2, 3),
    "act_num_frac_bits": (0, 1, 2, 3, 4, 5, 6),
    "weight_num_int_bits": (0, 1, 2, 3),
    "weight_num_frac_bits": (0, 1, 2, 3, 4, 5, 6)
    }

CLOCK_FREQUENCY = 100e6

'''
    `MIN_CONV_FEATURE_SIZE`: The sampled size of feature maps of layers (conv layer)
        along channel dimmension are multiples of 'MIN_CONV_FEATURE_SIZE'.
    `MIN_FC_FEATURE_SIZE`: The sampled size of features of FC layers are 
        multiples of 'MIN_FC_FEATURE_SIZE'.
'''
MIN_CONV_FEATURE_SIZE = 8
MIN_FC_FEATRE_SIZE    = 64

'''
    `MEASURE_LATENCY_BATCH_SIZE`: the batch size of input data
        when running forward functions to measure latency.
    `MEASURE_LATENCY_SAMPLE_TIMES`: the number of times to run the forward function of 
        a layer in order to get its latency.
'''
MEASURE_LATENCY_BATCH_SIZE = 128
MEASURE_LATENCY_SAMPLE_TIMES = 500


if __name__ == '__main__':
    print("architecture space: ", ARCH_SPACE)
    print("quantization space: ", QUAN_SPACE)
