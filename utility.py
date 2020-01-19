import logging
from config import ARCH_SPACE, QUAN_SPACE
import torch
import re

def get_sample_input(device, input_shape):
    sample_input = list(input_shape)
    sample_input.insert(0, 1)
    sample_input = tuple(sample_input)
    return torch.randn(sample_input).to(device)

def get_net_param(model):
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    #  print(f'Total number of parameters: {num_params}')
    return num_params

def cleanText(readData):
    """
    get rid of special chacters in the text
    """
    text = re.sub('[=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', readData)
    return text

def get_logger(filepath=None):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    if filepath is not None:
        file_handler = logging.FileHandler(filepath+'.log', mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(file_handler)
    return logger


def split_paras(paras):
    num_layers = len(paras)
    arch_paras = []
    quan_paras = []
    for i in range(num_layers):
        para = paras[i]
        arch_para = {}
        quan_para = {}
        for name, _ in ARCH_SPACE.items():
            if name in para:
                arch_para[name] = para[name]
            if 'anchor_point' in para:
                arch_para['anchor_point'] = para['anchor_point']
        for name, _ in QUAN_SPACE.items():
            if name in para:
                quan_para[name] = para[name]
        if arch_para != {}:
            arch_paras.append(arch_para)
        if quan_para != {}:
            quan_paras.append(quan_para)
        if arch_paras == []:
            arch_paras = None
        if quan_paras == []:
            quan_paras = None
    return arch_paras, quan_paras

def join_paras(arch_paras, quan_paras):
    paras = []
    for a, q in zip(arch_paras, quan_paras):
        paras.append(dict(a.items(), **q))
    return paras


def combine_rollout(arch_rollout, quan_rollout, num_layers):
    arch_num_paras_per_layer = int(len(arch_rollout) / num_layers)
    quan_num_paras_per_layer = int(len(quan_rollout) / num_layers)
    result = []
    for i in range(num_layers):
        result += arch_rollout[i * arch_num_paras_per_layer:
                               arch_num_paras_per_layer * (i + 1)]
        result += quan_rollout[i * quan_num_paras_per_layer:
                               quan_num_paras_per_layer * (i + 1)]
    return result


class BestSamples(object):
    def __init__(self, length=5):
        self.length = length
        self.id_list = list(range(1, self.length+1))
        self.rollout_list = [[]] * self.length
        self.reward_list = [-1] * self.length

    def register(self, id, rollout, reward):
        for i in range(self.length):
            if reward > self.reward_list[i]:
                self.reward_list[i] = reward
                self.id_list[i] = id
                self.rollout_list[i] = rollout
                break

    def __repr__(self):
        return str(dict(zip(self.id_list, self.reward_list)))



if __name__ == '__main__':
    # paras = [
    #     {'filter_height': 3, 'filter_width': 3, 'num_filters': 36,  # 0
    #      'anchor_point': []},
    #     {'filter_height': 3, 'filter_width': 3, 'num_filters': 48,  # 1
    #      'anchor_point': [1]},
    #     {'filter_height': 3, 'filter_width': 3, 'num_filters': 36,  # 2
    #      'anchor_point': [1, 1]},
    #     {'filter_height': 5, 'filter_width': 5, 'num_filters': 36,  # 3
    #      'anchor_point': [1, 1, 1]},
    #     {'filter_height': 3, 'filter_width': 7, 'num_filters': 48,  # 4
    #      'anchor_point': [0, 0, 1, 1]},
    #     {'filter_height': 7, 'filter_width': 7, 'num_filters': 48,  # 5
    #      'anchor_point': [0, 1, 1, 1, 1]},
    #     {'filter_height': 7, 'filter_width': 7, 'num_filters': 48,  # 6
    #      'anchor_point': [0, 1, 1, 1, 1, 1]},
    #     {'filter_height': 7, 'filter_width': 3, 'num_filters': 36,  # 7
    #      'anchor_point': [1, 0, 0, 0, 0, 1, 1]},
    #     {'filter_height': 7, 'filter_width': 1, 'num_filters': 36,  # 8
    #      'anchor_point': [1, 0, 0, 0, 1, 1, 0, 1]},
    #     {'filter_height': 7, 'filter_width': 7, 'num_filters': 36,  # 9
    #      'anchor_point': [1, 0, 1, 1, 1, 1, 1, 1, 1]},
    #     {'filter_height': 5, 'filter_width': 7, 'num_filters': 36,  # 10
    #      'anchor_point': [1, 1, 0, 0, 1, 1, 1, 1, 1, 1]},
    #     {'filter_height': 7, 'filter_width': 7, 'num_filters': 48,  # 11
    #      'anchor_point': [1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1]},
    #     {'filter_height': 7, 'filter_width': 5, 'num_filters': 48,  # 12
    #      'anchor_point': [1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1]},
    #     {'filter_height': 7, 'filter_width': 5, 'num_filters': 48,  # 13
    #      'anchor_point': [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1]},
    #     {'filter_height': 7, 'filter_width': 5, 'num_filters': 48,  # 14
    #      'anchor_point': [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1]}]
    # arch_paras, quan_paras = split_paras(paras)
    # print(arch_paras)
    # print()
    # print(quan_paras)
    # arch_rollout = [3, 3, 0, 0, 0, 1, 2, 2, 1, 0, 2, 1, 2, 0, 1, 2, 2, 1, 3, 3, 1, 1, 3, 0, 1, 2, 0, 2, 0, 1, 3, 2, 2, 2, 1, 1]
    # quan_rollout = [2, 1, 3, 4, 0, 0, 0, 6, 3, 2, 0, 4, 2, 4, 1, 2, 0, 2, 1, 1, 3, 6, 1, 2]
    # combine_rollout(arch_rollout, quan_rollout, 6)
    arch_paras = [
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
    quan_paras = [
    {'act_num_int_bits': 1, 'act_num_frac_bits': 4,
     'weight_num_int_bits': 3, 'weight_num_frac_bits': 5},
    {'act_num_int_bits': 2, 'act_num_frac_bits': 1,
     'weight_num_int_bits': 2, 'weight_num_frac_bits': 5},
    {'act_num_int_bits': 1, 'act_num_frac_bits': 4,
     'weight_num_int_bits': 2, 'weight_num_frac_bits': 4},
    {'act_num_int_bits': 0, 'act_num_frac_bits': 6,
     'weight_num_int_bits': 2, 'weight_num_frac_bits': 5},
    {'act_num_int_bits': 3, 'act_num_frac_bits': 1,
     'weight_num_int_bits': 0, 'weight_num_frac_bits': 5},
    {'act_num_int_bits': 2, 'act_num_frac_bits': 3,
     'weight_num_int_bits': 1, 'weight_num_frac_bits': 3}]

    print(join_paras(arch_paras, quan_paras))
