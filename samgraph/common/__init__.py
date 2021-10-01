import ctypes
import os
import sysconfig


def _get_ext_suffix():
    """Determine library extension for various versions of Python."""
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
    if ext_suffix:
        return ext_suffix

    ext_suffix = sysconfig.get_config_var('SO')
    if ext_suffix:
        return ext_suffix

    return '.so'


def _get_extension_full_path(pkg_path, *args):
    assert len(args) >= 1
    dir_path = os.path.join(os.path.dirname(pkg_path), *args[:-1])
    full_path = os.path.join(dir_path, args[-1] + _get_ext_suffix())
    return full_path


def _get_next_enum_val(next_val):
    res = next_val[0]
    next_val[0] += 1
    return res

kCPU  = 0
kMMAP = 1
kGPU  = 2

kKHop0                 = 0
kKHop1                 = 1
kWeightedKHop          = 2
kRandomWalk            = 3
kWeightedKHopPrefix    = 4
kKHop2                 = 5
kWeightedKHopHashDedup = 6

kArch0 = 0
kArch1 = 1
kArch2 = 2
kArch3 = 3
kArch4 = 4
kArch5 = 5

kCacheByDegree          = 0
kCacheByHeuristic       = 1
kCacheByPreSample       = 2
kCacheByDegreeHop       = 3
kCacheByPreSampleStatic = 4
kCacheByFakeOptimal     = 5
kDynamicCache           = 6


def cpu(device_id=0):
    return 'cpu:{:}'.format(device_id)


def gpu(device_id=0):
    return 'cuda:{:}'.format(device_id)


sample_types = {
    'khop0'                   : kKHop0,
    'khop1'                   : kKHop1,
    'khop2'                   : kKHop2,
    'random_walk'             : kRandomWalk,
    'weighted_khop'           : kWeightedKHop,
    'weighted_khop_hash_dedup': kWeightedKHopHashDedup
}


builtin_archs = {
    'arch0': {
        'arch'       : kArch0,
        'sampler_ctx': cpu(),
        'trainer_ctx': gpu(0)
    },
    'arch1': {
        'arch'       : kArch1,
        'sampler_ctx': gpu(0),
        'trainer_ctx': gpu(0)
    },
    'arch2': {
        'arch'       : kArch2,
        'sampler_ctx': gpu(0),
        'trainer_ctx': gpu(0)
    },
    'arch3': {
        'arch'       : kArch3,
        'sampler_ctx': gpu(0),
        'trainer_ctx': gpu(1)
    },
    'arch4': {
        'arch'       : kArch4,
        'sampler_ctx': gpu(1),
        'trainer_ctx': gpu(0)
    },
    'arch5': {
        'arch': kArch5
    }
}


cache_policies = {
    'degree'          : kCacheByDegree,
    'heuristic'       : kCacheByHeuristic,
    'pre_sample'      : kCacheByPreSample,
    'degree_hop'      : kCacheByDegreeHop,
    'presample_static': kCacheByPreSampleStatic,
    'fake_optimal'    : kCacheByFakeOptimal,
    'dynamic_cache'   : kDynamicCache
}


_step_log_val = [0]

# Step L1 Log
kLogL1NumSample        = _get_next_enum_val(_step_log_val)
kLogL1NumNode          = _get_next_enum_val(_step_log_val)
kLogL1SampleTime       = _get_next_enum_val(_step_log_val)
kLogL1SendTime         = _get_next_enum_val(_step_log_val)
kLogL1RecvTime         = _get_next_enum_val(_step_log_val)
kLogL1CopyTime         = _get_next_enum_val(_step_log_val)
kLogL1ConvertTime      = _get_next_enum_val(_step_log_val)
kLogL1TrainTime        = _get_next_enum_val(_step_log_val)
kLogL1FeatureBytes     = _get_next_enum_val(_step_log_val)
kLogL1LabelBytes       = _get_next_enum_val(_step_log_val)
kLogL1IdBytes          = _get_next_enum_val(_step_log_val)
kLogL1GraphBytes       = _get_next_enum_val(_step_log_val)
kLogL1MissBytes        = _get_next_enum_val(_step_log_val)
kLogL1PrefetchAdvanced = _get_next_enum_val(_step_log_val)
kLogL1GetNeighbourTime = _get_next_enum_val(_step_log_val)
# Step L2 Log
kLogL2ShuffleTime    = _get_next_enum_val(_step_log_val)
kLogL2LastLayerTime  = _get_next_enum_val(_step_log_val)
kLogL2LastLayerSize  = _get_next_enum_val(_step_log_val)
kLogL2CoreSampleTime = _get_next_enum_val(_step_log_val)
kLogL2IdRemapTime    = _get_next_enum_val(_step_log_val)
kLogL2GraphCopyTime  = _get_next_enum_val(_step_log_val)
kLogL2IdCopyTime     = _get_next_enum_val(_step_log_val)
kLogL2ExtractTime    = _get_next_enum_val(_step_log_val)
kLogL2FeatCopyTime   = _get_next_enum_val(_step_log_val)
kLogL2CacheCopyTime  = _get_next_enum_val(_step_log_val)
# Step L3 Log
kLogL3KHopSampleCooTime          = _get_next_enum_val(_step_log_val)
kLogL3KHopSampleSortCooTime      = _get_next_enum_val(_step_log_val)
kLogL3KHopSampleCountEdgeTime    = _get_next_enum_val(_step_log_val)
kLogL3KHopSampleCompactEdgesTime = _get_next_enum_val(_step_log_val)
kLogL3RandomWalkSampleCooTime    = _get_next_enum_val(_step_log_val)
kLogL3RandomWalkTopKTime         = _get_next_enum_val(_step_log_val)
kLogL3RandomWalkTopKStep1Time    = _get_next_enum_val(_step_log_val)
kLogL3RandomWalkTopKStep2Time    = _get_next_enum_val(_step_log_val)
kLogL3RandomWalkTopKStep3Time    = _get_next_enum_val(_step_log_val)
kLogL3RandomWalkTopKStep4Time    = _get_next_enum_val(_step_log_val)
kLogL3RandomWalkTopKStep5Time    = _get_next_enum_val(_step_log_val)
kLogL3RandomWalkTopKStep6Time    = _get_next_enum_val(_step_log_val)
kLogL3RandomWalkTopKStep7Time    = _get_next_enum_val(_step_log_val)
kLogL3RandomWalkTopKStep8Time    = _get_next_enum_val(_step_log_val)
kLogL3RandomWalkTopKStep9Time    = _get_next_enum_val(_step_log_val)
kLogL3RandomWalkTopKStep10Time   = _get_next_enum_val(_step_log_val)
kLogL3RandomWalkTopKStep11Time   = _get_next_enum_val(_step_log_val)
kLogL3RemapFillUniqueTime        = _get_next_enum_val(_step_log_val)
kLogL3RemapPopulateTime          = _get_next_enum_val(_step_log_val)
kLogL3RemapMapNodeTime           = _get_next_enum_val(_step_log_val)
kLogL3RemapMapEdgeTime           = _get_next_enum_val(_step_log_val)
kLogL3CacheGetIndexTime          = _get_next_enum_val(_step_log_val)
KLogL3CacheCopyIndexTime         = _get_next_enum_val(_step_log_val)
kLogL3CacheExtractMissTime       = _get_next_enum_val(_step_log_val)
kLogL3CacheCopyMissTime          = _get_next_enum_val(_step_log_val)
kLogL3CacheCombineMissTime       = _get_next_enum_val(_step_log_val)
kLogL3CacheCombineCacheTime      = _get_next_enum_val(_step_log_val)

# Epoch Log
_epoch_log_val = [0]
kLogEpochSampleTime                  = _get_next_enum_val(_epoch_log_val)
KLogEpochSampleGetCacheMissIndexTime = _get_next_enum_val(_epoch_log_val)
kLogEpochSampleSendTime              = _get_next_enum_val(_epoch_log_val)
kLogEpochSampleTotalTime             = _get_next_enum_val(_epoch_log_val)
kLogEpochCopyTime                    = _get_next_enum_val(_epoch_log_val)
kLogEpochConvertTime                 = _get_next_enum_val(_epoch_log_val)
kLogEpochTrainTime                   = _get_next_enum_val(_epoch_log_val)
kLogEpochTotalTime                   = _get_next_enum_val(_epoch_log_val)
kLogEpochFeatureBytes                = _get_next_enum_val(_epoch_log_val)
kLogEpochMissBytes                   = _get_next_enum_val(_epoch_log_val)

_step_event_val = [0]

kL0Event_Train_Step                  = _get_next_enum_val(_step_event_val)
kL1Event_Sample                      = _get_next_enum_val(_step_event_val)
kL2Event_Sample_Shuffle              = _get_next_enum_val(_step_event_val)
kL2Event_Sample_Core                 = _get_next_enum_val(_step_event_val)
kL2Event_Sample_IdRemap              = _get_next_enum_val(_step_event_val)
kL1Event_Copy                        = _get_next_enum_val(_step_event_val)
kL2Event_Copy_Id                     = _get_next_enum_val(_step_event_val)
kL2Event_Copy_Graph                  = _get_next_enum_val(_step_event_val)
kL2Event_Copy_Extract                = _get_next_enum_val(_step_event_val)
kL2Event_Copy_FeatCopy               = _get_next_enum_val(_step_event_val)
kL2Event_Copy_CacheCopy              = _get_next_enum_val(_step_event_val)
kL3Event_Copy_CacheCopy_GetIndex     = _get_next_enum_val(_step_event_val)
kL3Event_Copy_CacheCopy_CopyIndex    = _get_next_enum_val(_step_event_val)
kL3Event_Copy_CacheCopy_ExtractMiss  = _get_next_enum_val(_step_event_val)
kL3Event_Copy_CacheCopy_CopyMiss     = _get_next_enum_val(_step_event_val)
kL3Event_Copy_CacheCopy_CombineMiss  = _get_next_enum_val(_step_event_val)
kL3Event_Copy_CacheCopy_CombineCache = _get_next_enum_val(_step_event_val)
kL1Event_Convert                     = _get_next_enum_val(_step_event_val)
kL1Event_Train                       = _get_next_enum_val(_step_event_val)


class SamGraphBasics(object):
    def __init__(self, pkg_path, *args):
        full_path = _get_extension_full_path(pkg_path, *args)
        self.C_LIB_CTYPES = ctypes.CDLL(full_path, mode=ctypes.RTLD_GLOBAL)

        self.C_LIB_CTYPES.samgraph_get_graph_num_src.argtypes = (
            ctypes.c_uint64,)
        self.C_LIB_CTYPES.samgraph_get_graph_num_dst.argtypes = (
            ctypes.c_uint64,)
        self.C_LIB_CTYPES.samgraph_get_graph_num_edge.argtypes = (
            ctypes.c_uint64,)
        self.C_LIB_CTYPES.samgraph_log_step.argtypes = (
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.c_double
        )
        self.C_LIB_CTYPES.samgraph_log_step_add.argtypes = (
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.c_double
        )
        self.C_LIB_CTYPES.samgraph_log_epoch_add.argtypes = (
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.c_double
        )
        self.C_LIB_CTYPES.samgraph_get_log_step_value.argtypes = (
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_int
        )
        self.C_LIB_CTYPES.samgraph_get_log_epoch_value.argtypes = (
            ctypes.c_uint64,
            ctypes.c_int
        )
        self.C_LIB_CTYPES.samgraph_report_step.argtypes = (
            ctypes.c_uint64,
            ctypes.c_uint64)
        self.C_LIB_CTYPES.samgraph_report_step_average.argtypes = (
            ctypes.c_uint64,
            ctypes.c_uint64)
        self.C_LIB_CTYPES.samgraph_report_epoch.argtypes = (
            ctypes.c_uint64,)
        self.C_LIB_CTYPES.samgraph_report_epoch_average.argtypes = (
            ctypes.c_uint64,)

        self.C_LIB_CTYPES.samgraph_trace_step_begin.argtypes = (
            ctypes.c_uint64, ctypes.c_int, ctypes.c_uint64)
        self.C_LIB_CTYPES.samgraph_trace_step_end.argtypes = (
            ctypes.c_uint64, ctypes.c_int, ctypes.c_uint64)

        self.C_LIB_CTYPES.samgraph_trace_step_begin_now.argtypes = (
            ctypes.c_uint64, ctypes.c_int)
        self.C_LIB_CTYPES.samgraph_trace_step_end_now.argtypes = (
            ctypes.c_uint64, ctypes.c_int)

        self.C_LIB_CTYPES.samgraph_steps_per_epoch.restype = ctypes.c_size_t
        self.C_LIB_CTYPES.samgraph_num_class.restype = ctypes.c_size_t
        self.C_LIB_CTYPES.samgraph_feat_dim.restype = ctypes.c_size_t
        self.C_LIB_CTYPES.samgraph_get_next_batch.restype = ctypes.c_uint64
        self.C_LIB_CTYPES.samgraph_num_epoch.restype = ctypes.c_size_t
        self.C_LIB_CTYPES.samgraph_get_graph_num_src.restype = ctypes.c_size_t
        self.C_LIB_CTYPES.samgraph_get_graph_num_dst.restype = ctypes.c_size_t
        self.C_LIB_CTYPES.samgraph_get_graph_num_edge.restype = ctypes.c_size_t
        self.C_LIB_CTYPES.samgraph_get_log_step_value.restype = ctypes.c_double
        self.C_LIB_CTYPES.samgraph_get_log_epoch_value.restype = ctypes.c_double

    def config(self, run_config : dict):
        num_configs_items = len(run_config)
        config_keys = [str.encode(str(key)) for key in run_config.keys()]
        config_values = []
        for value in run_config.values():
            if isinstance(value, list):
                config_values.append(str.encode(
                    ' '.join([str(v) for v in value])))
            else:
                config_values.append(str.encode(str(value)))


        return self.C_LIB_CTYPES.samgraph_config(
            (ctypes.c_char_p * num_configs_items)(
                *config_keys
            ),
            (ctypes.c_char_p * num_configs_items) (
                *config_values
            ),
            ctypes.c_size_t(num_configs_items)
        )

    def init(self):
        return self.C_LIB_CTYPES.samgraph_init()

    '''
     for multi-GPUs train
    '''

    def data_init(self):
        return self.C_LIB_CTYPES.samgraph_data_init()

    '''
     for multi-GPUs train
    '''
    def sample_init(self, worker_id, ctx):
        return self.C_LIB_CTYPES.samgraph_sample_init(
            ctypes.c_int(worker_id),
            ctypes.c_char_p(str.encode(ctx))
        )

    '''
     for multi-GPUs train
    '''
    def train_init(self, worker_id, ctx):
        return self.C_LIB_CTYPES.samgraph_train_init(
            ctypes.c_int(worker_id),
            ctypes.c_char_p(str.encode(ctx))
        )

    # for dynamic switcher
    def switch_init(self, worker_id, ctx, cache_percentage):
        return self.C_LIB_CTYPES.samgraph_switch_init(
            ctypes.c_int(worker_id),
            ctypes.c_char_p(str.encode(ctx)),
            ctypes.c_double(cache_percentage)
        )


    def extract_start(self, count):
        return self.C_LIB_CTYPES.samgraph_extract_start(ctypes.c_int(count))

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

    def get_next_batch(self):
        batch_key = self.C_LIB_CTYPES.samgraph_get_next_batch()
        return batch_key

    def get_graph_num_src(self, key, graph_id):
        return self.C_LIB_CTYPES.samgraph_get_graph_num_src(key, graph_id)

    def get_graph_num_dst(self, key, graph_id):
        return self.C_LIB_CTYPES.samgraph_get_graph_num_dst(key, graph_id)

    def sample_once(self):
        return self.C_LIB_CTYPES.samgraph_sample_once()

    def log_step(self, epoch, step, item, val):
        return self.C_LIB_CTYPES.samgraph_log_step(epoch, step, item, val)

    def log_step_add(self, epoch, step, item, val):
        return self.C_LIB_CTYPES.samgraph_log_step_add(epoch, step, item, val)

    def log_epoch_add(self, epoch, item, val):
        return self.C_LIB_CTYPES.samgraph_log_epoch_add(epoch, item, val)

    def get_log_step_value(self, epoch, step, item):
        return self.C_LIB_CTYPES.samgraph_get_log_step_value(epoch, step, item)

    def get_log_epoch_value(self, epoch, item):
        return self.C_LIB_CTYPES.samgraph_get_log_epoch_value(epoch, item)

    def report_init(self):
        return self.C_LIB_CTYPES.samgraph_report_init()

    def report_step(self, epoch, step):
        return self.C_LIB_CTYPES.samgraph_report_step(epoch, step)

    def report_step_average(self, epoch, step):
        return self.C_LIB_CTYPES.samgraph_report_step_average(epoch, step)

    def report_epoch(self, epoch):
        return self.C_LIB_CTYPES.samgraph_report_epoch(epoch)

    def report_epoch_average(self, epoch):
        return self.C_LIB_CTYPES.samgraph_report_epoch_average(epoch)

    def report_node_access(self):
        return self.C_LIB_CTYPES.samgraph_report_node_access()

    def trace_step_begin(self, key, item, us):
        return self.C_LIB_CTYPES.samgraph_trace_step_begin(key, item, us)

    def trace_step_end(self, key, item, us):
        return self.C_LIB_CTYPES.samgraph_trace_step_end(key, item, us)

    def trace_step_begin_now(self, key, item):
        return self.C_LIB_CTYPES.samgraph_trace_step_begin_now(key, item)

    def trace_step_end_now(self, key, item):
        return self.C_LIB_CTYPES.samgraph_trace_step_end_now(key, item)

    def dump_trace(self):
        return self.C_LIB_CTYPES.samgraph_dump_trace()

    def forward_barrier(self):
        return self.C_LIB_CTYPES.samgraph_forward_barrier()
