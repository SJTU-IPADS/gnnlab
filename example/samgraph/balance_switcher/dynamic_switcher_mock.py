import multiprocessing as mp
import time
from enum import Enum

def run_sample(worker_id, config):
    global_barrier = run_config['global_barrier']
    mq_sem  = config['mq_sem']
    sampler_stop_event = config['sampler_stop_event']
    epoch = config['epoch']

    for e in range(epoch):
        global_barrier.wait()
        sampler_stop_event.clear()

        for i in range (32):
            time.sleep(0.01)
            print('send ', i)
            mq_sem.release()

        sampler_stop_event.set()
        global_barrier.wait()

class TrainerType(Enum):
    Trainer = 1
    Switcher = 2

def run_train(workder_id, config, trainer_type):
    global_barrier = run_config['global_barrier']
    mq_sem  = config['mq_sem']
    sampler_stop_event = config['sampler_stop_event']
    epoch = config['epoch']
    for e in range(epoch):
        global_barrier.wait()
        if (trainer_type == TrainerType.Switcher):
            sampler_stop_event.wait()
        i = 0
        while True:
            if (not mq_sem.acquire(timeout=0.01)):
                if (trainer_type == TrainerType.Switcher):
                    break
                elif (trainer_type == TrainerType.Trainer):
                    if (sampler_stop_event.is_set()):
                        break
                    else:
                        continue
                else:
                    assert(0)
            # bala bala
            time.sleep(0.02)
            i += 1
        print(f'Type {trainer_type.name} trainer count: ', i, flush=True)
        global_barrier.wait()

if '__main__' == __name__:

    num_sample_worker = 2;
    num_train_worker  = 3;

    run_config = {}
    run_config['global_barrier'] = mp.Barrier(2 * num_sample_worker + num_train_worker)
    run_config['mq_sem'] = mp.Semaphore(0)
    run_config['sampler_stop_event'] = mp.Event()
    run_config['epoch'] = 2

    workers = []
    # sample processes
    for worker_id in range(num_sample_worker):
        p = mp.Process(target=run_sample, args=(
            worker_id, run_config))
        p.start()
        workers.append(p)
        p = mp.Process(target=run_train, args=(
            worker_id, run_config, TrainerType.Switcher))
        p.start()
        workers.append(p)

    # train processes
    for worker_id in range(num_train_worker):
        p = mp.Process(target=run_train, args=(
            worker_id, run_config, TrainerType.Trainer))
        p.start()
        workers.append(p)

    for p in workers:
        p.join()
