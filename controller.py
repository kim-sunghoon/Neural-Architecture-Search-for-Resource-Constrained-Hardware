import torch

import controller_nl
import controller_linear


class Agent():
    def __init__(self, para_space, para_num_layers, batch_size=5,
                 device=torch.device('cpu'), skip=True):
        if skip:
            self.agent = controller_nl.Agent(
                para_space, para_num_layers,
                batch_size=batch_size,
                device=device
                )
        else:
            self.agent = controller_linear.Agent(
                para_space, para_num_layers,
                batch_size=batch_size,
                device=device
                )
        self.rollout = self.agent.rollout
        self.store_rollout = self.agent.store_rollout
