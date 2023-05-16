import os
import torch
import numpy as np
from dataset.base import Loader
class CASIALoader(Loader):
    def __init__(self, datadir, goal, batch_size, model):
        super(CASIALoader, self).__init__(batch_size, model)
        with open(os.path.join('../config', 'pairs_casia.txt')) as f:
            lines = f.readlines()
        suffix = '.jpg'
        self.pairs = []
        for line in lines:
            line = line.strip().split(',')
            if len(line) == 3 and goal == 'dodging':
                path_src = os.path.join(datadir, line[0], line[1] + suffix)
                path_dst = os.path.join(datadir, line[0], line[2] + suffix)
                self.pairs.append([path_src, path_dst])
            elif len(line) == 4 and goal == 'impersonate':
                path_src = os.path.join(datadir, line[0], line[1] + suffix)
                path_dst = os.path.join(datadir, line[2], line[3] + suffix)
                self.pairs.append([path_src, path_dst])