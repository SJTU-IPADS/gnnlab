import os
import torch

def event_sync():
    event = torch.cuda.Event(blocking=True)
    event.record()
    event.synchronize()

def get_default_timeout():
    # In seconds
    return 600

def wait_and_join(processes):
    ret = os.waitpid(-1, 0)
    if os.WEXITSTATUS(ret[1]) != 0:
        print("Detect pid {:} error exit".format(ret[0]))
        for p in processes:
            p.kill()
        
    for p in processes:
            p.join()