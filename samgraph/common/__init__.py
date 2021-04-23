import ctypes
import os
import sysconfig


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


class SamGraphContext(object):
    def __init__(self, device_type, device_id):
        self.device_type = device_type
        self.device_id = device_id


def cpu(device_id=0):
    kCPU = 0
    return SamGraphContext(kCPU, device_id)


def gpu(device_id=0):
    kGPU = 2
    return SamGraphContext(kGPU, device_id)


class SamGraphBasics(object):
    def __init__(self, pkg_path, *args):
        full_path = get_extension_full_path(pkg_path, *args)
        self.C_LIB_CTYPES = ctypes.CDLL(full_path, mode=ctypes.RTLD_GLOBAL)

        self.C_LIB_CTYPES.samgraph_get_next_batch.argtypes = (
            ctypes.c_uint64,
            ctypes.c_uint64)
        self.C_LIB_CTYPES.samgraph_get_graph_num_row.argtypes = (
            ctypes.c_uint64,)
        self.C_LIB_CTYPES.samgraph_get_graph_num_col.argtypes = (
            ctypes.c_uint64,)
        self.C_LIB_CTYPES.samgraph_get_graph_num_edge.argtypes = (
            ctypes.c_uint64,)
        self.C_LIB_CTYPES.samgraph_profiler_report.argtypes = (
            ctypes.c_uint64,
            ctypes.c_uint64)

        self.C_LIB_CTYPES.samgraph_steps_per_epoch.restype = ctypes.c_size_t
        self.C_LIB_CTYPES.samgraph_num_class.restype = ctypes.c_size_t
        self.C_LIB_CTYPES.samgraph_feat_dim.restype = ctypes.c_size_t
        self.C_LIB_CTYPES.samgraph_get_next_batch.restype = ctypes.c_uint64
        self.C_LIB_CTYPES.samgraph_num_epoch.restype = ctypes.c_size_t
        self.C_LIB_CTYPES.samgraph_get_graph_num_row.restype = ctypes.c_size_t
        self.C_LIB_CTYPES.samgraph_get_graph_num_col.restype = ctypes.c_size_t
        self.C_LIB_CTYPES.samgraph_get_graph_num_edge.restype = ctypes.c_size_t

    def config(self, run_config):
        return self.C_LIB_CTYPES.samgraph_config(
            ctypes.c_char_p(
                str.encode(run_config['dataset_path'])
            ),
            ctypes.c_int(
                run_config['sampler_ctx'].device_type
            ),
            ctypes.c_int(
                run_config['sampler_ctx'].device_id
            ),
            ctypes.c_int(
                run_config['trainer_ctx'].device_type
            ),
            ctypes.c_int(
                run_config['trainer_ctx'].device_id
            ),
            ctypes.c_size_t(
                run_config['batch_size']
            ),
            (ctypes.c_int * run_config['num_fanout'])(
                *run_config['fanout']
            ),
            ctypes.c_size_t(
                run_config['num_fanout']
            ),
            ctypes.c_size_t(
                run_config['num_epoch']
            ),
            ctypes.c_int(
                run_config['cpu_hashtable_type']
            ))

    def init(self):
        return self.C_LIB_CTYPES.samgraph_init()

    def start(self):
        return self.C_LIB_CTYPES.samgraph_start()

    def shutdown(self):
        return self.C_LIB_CTYPES.samgraph_shutdown()

    def num_class(self):
        return self.C_LIB_CTYPES.samgraph_num_class()

    def feat_dim(self):
        return self.C_LIB_CTYPES.samgraph_feat_dim()

    def num_epoch(self):
        return self.C_LIB_CTYPES.samgraph_num_epoch()

    def steps_per_epoch(self):
        return self.C_LIB_CTYPES.samgraph_steps_per_epoch()

    def get_next_batch(self, epoch, step):
        batch_key = self.C_LIB_CTYPES.samgraph_get_next_batch(epoch, step)
        return batch_key

    def get_graph_num_row(self, key, graph_id):
        return self.C_LIB_CTYPES.samgraph_get_graph_num_row(key, graph_id)

    def get_graph_num_col(self, key, graph_id):
        return self.C_LIB_CTYPES.samgraph_get_graph_num_col(key, graph_id)

    def sample(self):
        return self.C_LIB_CTYPES.samgraph_sample_once()

    def profiler_report(self, epoch, step):
        return self.C_LIB_CTYPES.samgraph_profiler_report(epoch, step)
