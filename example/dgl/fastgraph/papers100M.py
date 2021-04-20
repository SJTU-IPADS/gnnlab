import numpy as np
import os
import time
import torch


class Papers100M:
    def __init__(self, path):
        tic = time.time()

        self.num_node = 111059956
        self.num_edge = 1726745828
        self.feat_dim = 128
        self.num_class = 172
        self.num_train_set = 1207179
        self.num_valid_set = 125265
        self.num_test_set = 214338

        self.indptr = torch.from_numpy(np.memmap(os.path.join(
            path, 'indptr.bin'), dtype='int32', mode='r', shape=(self.num_node + 1,)))
        self.indices = torch.from_numpy(np.memmap(os.path.join(
            path, 'indices.bin'), dtype='int32', mode='r', shape=(self.num_edge,)))
        self.eids = torch.from_numpy(
            np.arange(0, self.num_edge, dtype='int32'))
        self.feat = torch.from_numpy(np.memmap(os.path.join(
            path, 'feat.bin'), dtype='float32', mode='r', shape=(self.num_node, self.feat_dim)))
        self.label = torch.from_numpy(np.memmap(os.path.join(
            path, 'label.bin'), dtype='long', mode='r', shape=(self.num_node,)))
        self.train_set = torch.from_numpy(np.memmap(os.path.join(
            path, 'train_set.bin'), dtype='int32', mode='r', shape=(self.num_train_set,)))
        self.valid_set = torch.from_numpy(np.memmap(os.path.join(
            path, 'valid_set.bin'), dtype='int32', mode='r', shape=(self.num_valid_set,)))
        self.test_set = torch.from_numpy(np.memmap(os.path.join(
            path, 'test_set.bin'), dtype='int32', mode='r', shape=(self.num_test_set,)
        ))

        toc = time.time()

        print('Loading papers100M uses {:4f} secs.'.format(toc-tic))

    def to_dgl_graph(self):
        import dgl

        g_idx = dgl.heterograph_index.create_unitgraph_from_csc(
            1, self.num_node, self.num_node, self.indptr, self.indices, self.eids, ['csr', 'csc'])
        g = dgl.DGLGraph(g_idx)
        g.ndata['feat'] = self.feat
        g.ndata['label'] = self.label

        return g
