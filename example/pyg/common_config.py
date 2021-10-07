import os
import torch
import math
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from typing import List, Optional, Tuple, NamedTuple, Union, Callable

import torch
from torch import Tensor
from torch_sparse import SparseTensor


def wait_and_join(processes):
    ret = os.waitpid(-1, 0)
    if os.WEXITSTATUS(ret[1]) != 0:
        print("Detect pid {:} error exit".format(ret[0]))
        for p in processes:
            p.kill()
        
    for p in processes:
            p.join()

def event_sync():
    event = torch.cuda.Event(blocking=True)
    event.record()
    event.synchronize()

def get_default_timeout():
    # In seconds
    return 10

def wait_and_join(processes):
    ret = os.waitpid(-1, 0)
    if os.WEXITSTATUS(ret[1]) != 0:
        print("Detect pid {:} error exit".format(ret[0]))
        for p in processes:
            p.kill()
        
    for p in processes:
            p.join()

class _ScalarDataBatcherIter:
    def __init__(self, dataset, batch_size, drop_last):
        self.dataset = dataset
        self.batch_size = batch_size
        self.index = 0
        self.drop_last = drop_last

    # Make this an iterator for PyTorch Lightning compatibility
    def __iter__(self):
        return self

    def __next__(self):
        num_items = self.dataset.shape[0]
        if self.index >= num_items:
            raise StopIteration
        end_idx = self.index + self.batch_size
        if end_idx > num_items:
            if self.drop_last:
                raise StopIteration
            end_idx = num_items
        batch = self.dataset[self.index:end_idx]
        self.index += self.batch_size

        return batch

class _ScalarDataBatcher(torch.utils.data.IterableDataset):
    """Custom Dataset wrapper to return mini-batches as tensors, rather than as
    lists. When the dataset is on the GPU, this significantly reduces
    the overhead. For the case of a batch size of 1024, instead of giving a
    list of 1024 tensors to the collator, a single tensor of 1024 dimensions
    is passed in.
    """
    def __init__(self, dataset, shuffle=False, batch_size=1,
                 drop_last=False, use_ddp=False, ddp_seed=0):
        super(_ScalarDataBatcher).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.use_ddp = use_ddp
        if use_ddp:
            self.rank = dist.get_rank()
            self.num_replicas = dist.get_world_size()
            self.seed = ddp_seed
            self.epoch = 0
            # The following code (and the idea of cross-process shuffling with the same seed)
            # comes from PyTorch.  See torch/utils/data/distributed.py for details.

            # If the dataset length is evenly divisible by # of replicas, then there
            # is no need to drop any sample, since the dataset will be split evenly.
            if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore
                # Split to nearest available length that is evenly divisible.
                # This is to ensure each rank receives the same amount of data when
                # using this Sampler.
                self.num_samples = math.ceil(
                    # `type:ignore` is required because Dataset cannot provide a default __len__
                    # see NOTE in pytorch/torch/utils/data/sampler.py
                    (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore
                )
            else:
                self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore
            self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        if self.use_ddp:
            return self._iter_ddp()
        else:
            return self._iter_non_ddp()

    def _divide_by_worker(self, dataset):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            # worker gets only a fraction of the dataset
            chunk_size = dataset.shape[0] // worker_info.num_workers
            left_over = dataset.shape[0] % worker_info.num_workers
            start = (chunk_size*worker_info.id) + min(left_over, worker_info.id)
            end = start + chunk_size + (worker_info.id < left_over)
            assert worker_info.id < worker_info.num_workers-1 or \
                end == dataset.shape[0]
            dataset = dataset[start:end]

        return dataset

    def _iter_non_ddp(self):
        dataset = self._divide_by_worker(self.dataset)

        if self.shuffle:
            # permute the dataset
            perm = torch.randperm(dataset.shape[0], device=dataset.device)
            dataset = dataset[perm]

        return _ScalarDataBatcherIter(dataset, self.batch_size, self.drop_last)

    def _iter_ddp(self):
        # The following code (and the idea of cross-process shuffling with the same seed)
        # comes from PyTorch.  See torch/utils/data/distributed.py for details.
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)
        else:
            indices = torch.arange(len(self.dataset))

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            indices = torch.cat([indices, indices[:(self.total_size - indices.shape[0])]])
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert indices.shape[0] == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert indices.shape[0] == self.num_samples

        # Dividing by worker is our own stuff.
        dataset = self._divide_by_worker(self.dataset[indices])
        return _ScalarDataBatcherIter(dataset, self.batch_size, self.drop_last)

    def __len__(self):
        num_samples = self.num_samples if self.use_ddp else self.dataset.shape[0]
        return (num_samples + (0 if self.drop_last else self.batch_size - 1)) // self.batch_size

    def set_epoch(self, epoch):
        """Set epoch number for distributed training."""
        self.epoch = epoch

class EdgeIndex(NamedTuple):
    edge_index: Tensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        edge_index = self.edge_index.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return EdgeIndex(edge_index, e_id, self.size)


class Adj(NamedTuple):
    adj_t: SparseTensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        adj_t = self.adj_t.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return Adj(adj_t, e_id, self.size)

class MyNeighborSampler(torch.utils.data.DataLoader):
    r"""
    Args:
        edge_index (Tensor or SparseTensor): A :obj:`torch.LongTensor` or a
            :obj:`torch_sparse.SparseTensor` that defines the underlying graph
            connectivity/message passing flow.
            :obj:`edge_index` holds the indices of a (sparse) symmetric
            adjacency matrix.
            If :obj:`edge_index` is of type :obj:`torch.LongTensor`, its shape
            must be defined as :obj:`[2, num_edges]`, where messages from nodes
            :obj:`edge_index[0]` are sent to nodes in :obj:`edge_index[1]`
            (in case :obj:`flow="source_to_target"`).
            If :obj:`edge_index` is of type :obj:`torch_sparse.SparseTensor`,
            its sparse indices :obj:`(row, col)` should relate to
            :obj:`row = edge_index[1]` and :obj:`col = edge_index[0]`.
            The major difference between both formats is that we need to input
            the *transposed* sparse adjacency matrix.
        sizes ([int]): The number of neighbors to sample for each node in each
            layer. If set to :obj:`sizes[l] = -1`, all neighbors are included
            in layer :obj:`l`.
        node_idx (LongTensor, optional): The nodes that should be considered
            for creating mini-batches. If set to :obj:`None`, all nodes will be
            considered.
        num_nodes (int, optional): The number of nodes in the graph.
            (default: :obj:`None`)
        return_e_id (bool, optional): If set to :obj:`False`, will not return
            original edge indices of sampled edges. This is only useful in case
            when operating on graphs without edge features to save memory.
            (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in
            a sampled mini-batch and returns a transformed version.
            (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size`,
            :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.
    """
    def __init__(self, edge_index: Union[Tensor, SparseTensor],
                 sizes: List[int], node_idx: Optional[Tensor] = None,
                 num_nodes: Optional[int] = None, return_e_id: bool = True,
                 transform: Callable = None, use_ddp : bool = False, ddp_seed=0, **kwargs):

        edge_index = edge_index.to('cpu')

        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        # Save for Pytorch Lightning...
        self.edge_index = edge_index
        self.node_idx = node_idx
        self.num_nodes = num_nodes

        self.sizes = sizes
        self.return_e_id = return_e_id
        self.transform = transform
        self.is_sparse_tensor = isinstance(edge_index, SparseTensor)
        self.__val__ = None

        # Obtain a *transposed* `SparseTensor` instance.
        if not self.is_sparse_tensor:
            if (num_nodes is None and node_idx is not None
                    and node_idx.dtype == torch.bool):
                num_nodes = node_idx.size(0)
            if (num_nodes is None and node_idx is not None
                    and node_idx.dtype == torch.long):
                num_nodes = max(int(edge_index.max()), int(node_idx.max())) + 1
            if num_nodes is None:
                num_nodes = int(edge_index.max()) + 1

            value = torch.arange(edge_index.size(1)) if return_e_id else None
            self.adj_t = SparseTensor(row=edge_index[0], col=edge_index[1],
                                      value=value,
                                      sparse_sizes=(num_nodes, num_nodes)).t()
        else:
            adj_t = edge_index
            if return_e_id:
                self.__val__ = adj_t.storage.value()
                value = torch.arange(adj_t.nnz())
                adj_t = adj_t.set_value(value, layout='coo')
            self.adj_t = adj_t

        self.adj_t.storage.rowptr()

        if node_idx is None:
            node_idx = torch.arange(self.adj_t.sparse_size(0))
        elif node_idx.dtype == torch.bool:
            node_idx = node_idx.nonzero(as_tuple=False).view(-1)
        
        if use_ddp:
            dataloader_kwargs = {}
            # dataset = _ScalarDataBatcher(node_idx,
            #                              batch_size=kwargs.get('batch_size', 1),
            #                              shuffle=kwargs.get('shuffle', False),
            #                              drop_last=kwargs.get('drop_last', False),
            #                              use_ddp=use_ddp,
            #                              ddp_seed=ddp_seed)
            dataloader_kwargs.update(kwargs)
            # dataloader_kwargs['batch_size'] = None
            dataloader_kwargs['shuffle'] = False
            dataloader_kwargs['drop_last'] = False
            self.use_ddp = use_ddp
            # self.scalar_batcher = dataset
            self.dist_sampler = DistributedSampler(node_idx, shuffle=kwargs['shuffle'], drop_last=kwargs['drop_last'])
            super(MyNeighborSampler, self).__init__(node_idx, sampler=self.dist_sampler, collate_fn=self.sample, **dataloader_kwargs)
        else:
            super(MyNeighborSampler, self).__init__(node_idx, collate_fn=self.sample, **kwargs)

    def sample(self, batch):
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)

        batch_size: int = len(batch)

        adjs = []
        n_id = batch
        for size in self.sizes:
            adj_t, n_id = self.adj_t.sample_adj(n_id, size, replace=False)
            e_id = adj_t.storage.value()
            size = adj_t.sparse_sizes()[::-1]
            if self.__val__ is not None:
                adj_t.set_value_(self.__val__[e_id], layout='coo')

            if self.is_sparse_tensor:
                adjs.append(Adj(adj_t, e_id, size))
            else:
                row, col, _ = adj_t.coo()
                edge_index = torch.stack([col, row], dim=0)
                adjs.append(EdgeIndex(edge_index, e_id, size))

        adjs = adjs[0] if len(adjs) == 1 else adjs[::-1]
        out = (batch_size, n_id, adjs)
        out = self.transform(*out) if self.transform is not None else out
        return out
    
    def set_epoch(self, epoch):
        assert(self.use_ddp)
        self.dist_sampler.set_epoch(epoch)

    def __repr__(self):
        return '{}(sizes={})'.format(self.__class__.__name__, self.sizes)

