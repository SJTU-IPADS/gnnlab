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

class SamGraphBasics(object):
    def __init__(self, pkg_path, *args):
        full_path = get_extension_full_path(pkg_path, *args)
        self.C_LIB_CTYPES = ctypes.CDLL(full_path, mode=ctypes.RTLD_GLOBAL)

    def init(self, path, device, batch_size, fanout, num_epoch):
        # parameter conversion
        if isinstance(path, str):
            path = str.encode(path)
        
        num_fanout = len(fanout)
        fanout = (ctypes.c_int * num_fanout)(*fanout)

        return self.C_LIB_CTYPES.samgraph_init(path, device, batch_size, fanout, num_fanout, num_epoch)
    
    def start(self):
        return self.C_LIB_CTYPES.samgraph_start()

    def stop(self):
        return self.C_LIB_CTYPES.samgraph_stop()

    def shutdown(self):
        return self.C_LIB_CTYPES.samgraph_shutdown()
