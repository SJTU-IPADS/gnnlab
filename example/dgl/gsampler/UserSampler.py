import dgl
from dgl import utils
import dgl.backend as F
from dgl.base import DGLError, EID
from dgl.heterograph import DGLHeteroGraph
from dgl import ndarray as nd
import torch
from dgl.dataloading.dataloader import BlockSampler
from dgl.distributed.dist_graph import DistGraph
from dgl import distributed

def sample_neighbors(g, topo_g, nodes, fanout, edge_dir='in', prob=None, replace=False,
                     copy_ndata=True, copy_edata=True, _dist_training=False):
    if not isinstance(nodes, dict):
        if len(g.ntypes) > 1:
            raise DGLError("Must specify node type when the graph is not homogeneous.")
        nodes = {g.ntypes[0] : nodes}

    ctx = topo_g.ctx
    nodes = utils.prepare_tensor_dict(g, nodes, 'nodes')
    nodes_all_types = []
    for ntype in g.ntypes:
        if ntype in nodes:
            nodes_all_types.append(F.to_dgl_nd(nodes[ntype].to(F.to_backend_ctx(ctx))))
        else:
            nodes_all_types.append(nd.array([], ctx=ctx))

    if isinstance(fanout, nd.NDArray):
        fanout_array = fanout
    else:
        if not isinstance(fanout, dict):
            fanout_array = [int(fanout)] * len(g.etypes)
        else:
            if len(fanout) != len(g.etypes):
                raise DGLError('Fan-out must be specified for each edge type '
                               'if a dict is provided.')
            fanout_array = [None] * len(g.etypes)
            for etype, value in fanout.items():
                fanout_array[g.get_etype_id(etype)] = value
        fanout_array = F.to_dgl_nd(F.tensor(fanout_array, dtype=F.int64).to(F.to_backend_ctx(ctx)))

    if isinstance(prob, list) and len(prob) > 0 and \
            isinstance(prob[0], nd.NDArray):
        prob_arrays = prob
    elif prob is None:
        prob_arrays = [nd.array([], ctx=ctx)] * len(g.etypes)
    else:
        prob_arrays = []
        for etype in g.canonical_etypes:
            if prob in g.edges[etype].data:
                prob_arrays.append(F.to_dgl_nd(g.edges[etype].data[prob].to(F.to_backend_ctx(ctx))))
            else:
                prob_arrays.append(nd.array([], ctx=ctx))

    subgidx = dgl.sampling.neighbor._CAPI_DGLSampleNeighbors(topo_g, nodes_all_types,
                                                    fanout_array, edge_dir, prob_arrays, replace)
    induced_edges = subgidx.induced_edges
    ret = DGLHeteroGraph(subgidx.graph, g.ntypes, g.etypes)

    # handle features
    # (TODO) (BarclayII) DGL distributed fails with bus error, freezes, or other
    # incomprehensible errors with lazy feature copy.
    # So in distributed training context, we fall back to old behavior where we
    # only set the edge IDs.
    if not _dist_training:
        if copy_ndata:
            node_frames = utils.extract_node_subframes(g, None)
            utils.set_new_frames(ret, node_frames=node_frames)

        if copy_edata:
            edge_frames = utils.extract_edge_subframes(g, induced_edges)
            utils.set_new_frames(ret, edge_frames=edge_frames)
    else:
        for i, etype in enumerate(ret.canonical_etypes):
            ret.edges[etype].data[EID] = induced_edges[i]

    return ret

class UserSampler(BlockSampler):
    def __init__(self, fanouts, topo_g, replace=False, return_eids=False):
        super().__init__(len(fanouts), return_eids)

        self.topo_g = topo_g
        self.ctx = topo_g.ctx
        self.fanouts = fanouts
        self.replace = replace

        # used to cache computations and memory allocations
        # list[dgl.nd.NDArray]; each array stores the fan-outs of all edge types
        self.fanout_arrays = []
        self.prob_arrays = None

    def sample_frontier(self, block_id, g, seed_nodes):
        fanout = self.fanouts[block_id]
        if isinstance(g, distributed.DistGraph):
            if fanout is None:
                # TODO(zhengda) There is a bug in the distributed version of in_subgraph.
                # let's use sample_neighbors to replace in_subgraph for now.
                frontier = distributed.sample_neighbors(g, seed_nodes, -1, replace=False)
            else:
                frontier = distributed.sample_neighbors(g, seed_nodes, fanout, replace=self.replace)
        else:
            if fanout is None:
                frontier = subgraph.in_subgraph(g, seed_nodes)
            else:
                self._build_fanout(block_id, g)
                self._build_prob_arrays(g)

                frontier = sample_neighbors(
                    g, self.topo_g, seed_nodes, self.fanout_arrays[block_id],
                    replace=self.replace, prob=self.prob_arrays)
        return frontier

    def _build_prob_arrays(self, g):
        # build prob_arrays only once
        if self.prob_arrays is None:
            self.prob_arrays = [nd.array([], ctx=self.ctx)] * len(g.etypes)

    def _build_fanout(self, block_id, g):
        assert not self.fanouts is None, \
            "_build_fanout() should only be called when fanouts is not None"
        # build fanout_arrays only once for each layer
        while block_id >= len(self.fanout_arrays):
            for i in range(len(self.fanouts)):
                fanout = self.fanouts[i]
                if not isinstance(fanout, dict):
                    fanout_array = [int(fanout)] * len(g.etypes)
                else:
                    if len(fanout) != len(g.etypes):
                        raise DGLError('Fan-out must be specified for each edge type '
                                       'if a dict is provided.')
                    fanout_array = [None] * len(g.etypes)
                    for etype, value in fanout.items():
                        fanout_array[g.get_etype_id(etype)] = value
                self.fanout_arrays.append(
                    F.to_dgl_nd(F.tensor(fanout_array, dtype=F.int64).to(F.to_backend_ctx(self.ctx))))

def test():
    print("--- gpu sampling ---")
    g = dgl.graph(([0,0,2,1,1,2,2], [1,2,0,3,4,4,5]))
    context=dgl.ndarray.gpu(0)
    # sg = dgl.sampling.sample_neighbors(g, 1, 1, 'out')
    ctx = nd.gpu(1)
    sg = sample_neighbors(g, g._graph.copy_to(ctx), 1, 1, 'out')
    print("sg: ", sg.edges())



if __name__ == "__main__":
    test()
