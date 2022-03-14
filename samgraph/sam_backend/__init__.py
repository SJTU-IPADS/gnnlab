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

# Load all the necessary PyTorch C types.

from samgraph.common import *
from samgraph.common import _basics
C_LIB_CTYPES = ctypes.CDLL(os.path.join(__path__[0], "c_lib" + _get_ext_suffix()), mode=ctypes.RTLD_GLOBAL)

# from samgraph.sam_backend import c_lib

config               = _basics.config
init                 = _basics.init
start                = _basics.start
num_class            = _basics.num_class
feat_dim             = _basics.feat_dim
num_epoch            = _basics.num_epoch
steps_per_epoch      = _basics.steps_per_epoch
get_next_batch       = _basics.get_next_batch
get_graph_num_src    = _basics.get_graph_num_src
get_graph_num_dst    = _basics.get_graph_num_dst
shutdown             = _basics.shutdown
sample_once          = _basics.sample_once
log_step             = _basics.log_step
log_step_add         = _basics.log_step_add
log_epoch_add        = _basics.log_epoch_add
get_log_init_value   = _basics.get_log_init_value
get_log_step_value   = _basics.get_log_step_value
get_log_epoch_value  = _basics.get_log_epoch_value
report_init          = _basics.report_init
report_step          = _basics.report_step
report_step_average  = _basics.report_step_average
report_epoch         = _basics.report_epoch
report_epoch_average = _basics.report_epoch_average
report_node_access   = _basics.report_node_access
trace_step_begin     = _basics.trace_step_begin
trace_step_end       = _basics.trace_step_end
trace_step_begin_now = _basics.trace_step_begin_now
trace_step_end_now   = _basics.trace_step_end_now
dump_trace           = _basics.dump_trace
forward_barrier      = _basics.forward_barrier
wait_one_child       = _basics.wait_one_child

# switch APIs
switch_init   = _basics.switch_init

# multi-GPUs training APIs
data_init      = _basics.data_init
sample_init    = _basics.sample_init
train_init     = _basics.train_init
extract_start  = _basics.extract_start
num_local_step = _basics.num_local_step

def backend_init_model():
    return C_LIB_CTYPES.samgraph_backend_init_model();
def backend_train_current_batch():
    return C_LIB_CTYPES.samgraph_backend_train_current_batch();
