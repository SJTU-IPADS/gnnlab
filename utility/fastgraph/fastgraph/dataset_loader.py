import numpy as np
import os
import time
import torch

from .meta_reader import MetaReader

INT_MAX = 2**31


class DatasetLoader:
    def __init__(self, dataset_path, force_load64=False):
        tic = time.time()

        meta_reader = MetaReader()
        meta = meta_reader.read(dataset_path)

        self.num_node = meta['NUM_NODE']
        self.num_edge = meta['NUM_EDGE']
        self.feat_dim = meta['FEAT_DIM']
        self.num_class = meta['NUM_CLASS']
        self.num_train_set = meta['NUM_TRAIN_SET']
        self.num_valid_set = meta['NUM_VALID_SET']
        self.num_test_set = meta['NUM_TEST_SET']

        if self.num_edge < INT_MAX or force_load64:
            self.load32(dataset_path, force_load64)
        else:
            self.load64(dataset_path)

        if os.path.isfile(os.path.join(
                dataset_path, 'feat.bin')):
            self.feat = torch.from_numpy(np.memmap(os.path.join(
                dataset_path, 'feat.bin'), dtype='float32', mode='r', shape=(self.num_node, self.feat_dim)))
        else:
            self.feat = torch.empty(
                (self.num_node, self.feat_dim), dtype=torch.float32)
        if os.path.isfile(os.path.join(
                dataset_path, 'label.bin')):
            self.label = torch.from_numpy(np.memmap(os.path.join(
                dataset_path, 'label.bin'), dtype='long', mode='r', shape=(self.num_node,)))
        else:
            self.label = torch.empty(
                (self.num_node, ), dtype=torch.long)

        toc = time.time()

        print('Loading {:s} uses {:4f} secs.'.format(dataset_path, toc-tic))

    def load32(self, dataset_path, force_load64):
        self.indptr = torch.from_numpy(np.memmap(os.path.join(
            dataset_path, 'indptr.bin'), dtype='int32', mode='r', shape=(self.num_node + 1,)))
        self.indices = torch.from_numpy(np.memmap(os.path.join(
            dataset_path, 'indices.bin'), dtype='int32', mode='r', shape=(self.num_edge,)))
        self.eids = torch.from_numpy(
            np.arange(0, self.num_edge, dtype='int32'))
        if os.path.isfile(os.path.join(
                dataset_path, 'feat.bin')):
            self.feat = torch.from_numpy(np.memmap(os.path.join(
                dataset_path, 'feat.bin'), dtype='float32', mode='r', shape=(self.num_node, self.feat_dim)))
        else:
            self.feat = torch.empty(
                (self.num_node, self.feat_dim), dtype=torch.float32)
        if os.path.isfile(os.path.join(
                dataset_path, 'label.bin')):
            self.label = torch.from_numpy(np.memmap(os.path.join(
                dataset_path, 'label.bin'), dtype='long', mode='r', shape=(self.num_node,)))
        else:
            self.label = torch.empty(
                (self.num_node, ), dtype=torch.long)

        self.train_set = torch.from_numpy(np.memmap(os.path.join(
            dataset_path, 'train_set.bin'), dtype='int32', mode='r', shape=(self.num_train_set,)))
        self.valid_set = torch.from_numpy(np.memmap(os.path.join(
            dataset_path, 'valid_set.bin'), dtype='int32', mode='r', shape=(self.num_valid_set,)))
        self.test_set = torch.from_numpy(np.memmap(os.path.join(
            dataset_path, 'test_set.bin'), dtype='int32', mode='r', shape=(self.num_test_set,)
        ))

        if force_load64:
            self.indptr = self.indptr.to(torch.int64)
            self.indices = self.indices.to(torch.int64)
            self.eids = self.eids.to(torch.int64)
            self.train_set = self.train_set.to(torch.int64)
            self.valid_set = self.valid_set.to(torch.int64)
            self.test_set = self.test_set.to(torch.int64)


    def load64(self, dataset_path):
        self.indptr = torch.from_numpy(np.memmap(os.path.join(
            dataset_path, 'indptr64.bin'), dtype='int64', mode='r', shape=(self.num_node + 1,)))
        self.indices = torch.from_numpy(np.memmap(os.path.join(
            dataset_path, 'indices64.bin'), dtype='int64', mode='r', shape=(self.num_edge,)))
        self.eids = torch.from_numpy(
            np.arange(0, self.num_edge, dtype='int64'))

        self.train_set = torch.from_numpy(np.memmap(os.path.join(
            dataset_path, 'train_set64.bin'), dtype='int64', mode='r', shape=(self.num_train_set,)))
        self.valid_set = torch.from_numpy(np.memmap(os.path.join(
            dataset_path, 'valid_set64.bin'), dtype='int64', mode='r', shape=(self.num_valid_set,)))
        self.test_set = torch.from_numpy(np.memmap(os.path.join(
            dataset_path, 'test_set64.bin'), dtype='int64', mode='r', shape=(self.num_test_set,)))

    def to_dgl_graph(self, g_format='csc'):
        import dgl

        if g_format == 'csc':
            g_idx = dgl.heterograph_index.create_unitgraph_from_csc(
                # 1, self.num_node, self.num_node, self.indptr, self.indices, self.eids, ['csc'])
                1, self.num_node, self.num_node, self.indptr, self.indices, torch.empty(0), ['csc'])
        elif g_format == 'csr':
            g_idx = dgl.heterograph_index.create_unitgraph_from_csr(
                # 1, self.num_node, self.num_node, self.indptr, self.indices, self.eids, ['csr'])
                1, self.num_node, self.num_node, self.indptr, self.indices, torch.empty(0), ['csr'])
        else:
            assert(False)

        g = dgl.DGLGraph(g_idx)

        return g

    def to_pyg_graph(self):
        from torch_sparse import SparseTensor

        # Sparse require the dtype to be torch.long
        # We has already sorted the indices tensor in the preprocessing step
        return SparseTensor(rowptr=self.indptr, col=self.indices, sparse_sizes=(self.num_node, self.num_node), is_sorted=True)
