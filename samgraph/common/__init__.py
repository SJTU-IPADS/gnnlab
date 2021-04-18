import ctypes
import os
import sysconfig
from enum import IntEnum


def get_ext_suffix():
    """Determine library extension for various versions of Python."""
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
    if ext_suffix:
        return ext_suffix

    ext_suffix = sysconfig.get_config_var('SO')
    if ext_suffix:
        return ext_suffix

    return '.so'


def get_extension_full_path(pkg_path, *args):
    assert len(args) >= 1
    dir_path = os.path.join(os.path.dirname(pkg_path), *args[:-1])
    full_path = os.path.join(dir_path, args[-1] + get_ext_suffix())
    return full_path


class Graph(object):
    def __init__(self, key, graph_id, num_row, num_col, num_edge):
        self.key = key
        self.graph_id = graph_id
        self.num_row = num_row
        self.num_col = num_col
        self.num_edge = num_edge


class GraphBatch(object):
    def __init__(self, C_LIB_CTYPES, key, num_graph):
        self.key = key
        self.graphs = []
        for graph_id in range(num_graph):
            graph_key = C_LIB_CTYPES.samgraph_get_graph_key(key, graph_id)
            num_row = C_LIB_CTYPES.samgraph_get_graph_num_row(graph_key)
            num_col = C_LIB_CTYPES.samgraph_get_graph_num_col(graph_key)
            num_edge = C_LIB_CTYPES.samgraph_get_graph_num_edge(graph_key)
            self.graphs.append(
                Graph(graph_key, graph_id, num_row, num_col, num_edge))


class SamGraphBasics(object):
    def __init__(self, pkg_path, *args):
        full_path = get_extension_full_path(pkg_path, *args)
        self.C_LIB_CTYPES = ctypes.CDLL(full_path, mode=ctypes.RTLD_GLOBAL)

        self.C_LIB_CTYPES.samgraph_num_epoch.restype = ctypes.c_int

        self.C_LIB_CTYPES.samgraph_num_step_per_epoch.restype = ctypes.c_size_t
        self.C_LIB_CTYPES.samgraph_dataset_num_class.restype = ctypes.c_size_t
        self.C_LIB_CTYPES.samgraph_dataset_num_feat_dim.restype = ctypes.c_size_t

        self.C_LIB_CTYPES.samgraph_get_next_batch.argtypes = (
            ctypes.c_int, ctypes.c_int)
        self.C_LIB_CTYPES.samgraph_get_next_batch.restype = ctypes.c_uint64

        self.C_LIB_CTYPES.samgraph_get_graph_key.argtypes = (
            ctypes.c_uint64, ctypes.c_int)
        self.C_LIB_CTYPES.samgraph_get_graph_key.restype = ctypes.c_uint64

        self.C_LIB_CTYPES.samgraph_get_graph_num_row.argtypes = (
            ctypes.c_uint64,)
        self.C_LIB_CTYPES.samgraph_get_graph_num_col.argtypes = (
            ctypes.c_uint64,)
        self.C_LIB_CTYPES.samgraph_get_graph_num_edge.argtypes = (
            ctypes.c_uint64,)
        self.C_LIB_CTYPES.samgraph_get_graph_num_row.restype = ctypes.c_size_t
        self.C_LIB_CTYPES.samgraph_get_graph_num_col.restype = ctypes.c_size_t
        self.C_LIB_CTYPES.samgraph_get_graph_num_edge.restype = ctypes.c_size_t
        self.C_LIB_CTYPES.samgraph_profiler_report.argtypes = (
            ctypes.c_int, ctypes.c_int)

    def init(self, path, sample_device, train_device, batch_size, fanout, num_epoch):
        num_fanout = len(fanout)

        return self.C_LIB_CTYPES.samgraph_init(ctypes.c_char_p(str.encode(path)),
                                               ctypes.c_int(sample_device),
                                               ctypes.c_int(train_device),
                                               ctypes.c_ulonglong(batch_size),
                                               (ctypes.c_int * num_fanout)(*fanout),
                                               ctypes.c_ulonglong(num_fanout),
                                               ctypes.c_int(num_epoch))

    def start(self):
        return self.C_LIB_CTYPES.samgraph_start()

    def dataset_num_class(self):
        return self.C_LIB_CTYPES.samgraph_dataset_num_class()

    def dataset_num_feat_dim(self):
        return self.C_LIB_CTYPES.samgraph_dataset_num_feat_dim()

    def num_epoch(self):
        return self.C_LIB_CTYPES.samgraph_num_epoch()

    def num_step_per_epoch(self):
        return self.C_LIB_CTYPES.samgraph_num_step_per_epoch()

    def get_next_batch(self, epoch, step, num_graph):
        batch_key = self.C_LIB_CTYPES.samgraph_get_next_batch(epoch, step)
        return GraphBatch(self.C_LIB_CTYPES, batch_key, num_graph)

    def shutdown(self):
        return self.C_LIB_CTYPES.samgraph_shutdown()

    def test_cusparse(self):
        return self.C_LIB_CTYPES.samgraph_test_cusparse()

    def sample(self):
        return self.C_LIB_CTYPES.samgraph_sample_once()

    def profiler_report(self, epoch, step):
        return self.C_LIB_CTYPES.samgraph_profiler_report(epoch, step)
