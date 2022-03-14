import argparse
import time
import sys
import numpy as np

from common_config import *
import samgraph.sam_backend as sam

def parse_args(default_run_config):
    argparser = argparse.ArgumentParser("SamBack Training")

    add_common_arguments(argparser, default_run_config)

    argparser.add_argument('--fanout', nargs='+',
                           type=int, default=default_run_config['fanout'])
    argparser.add_argument('--lr', type=float, default=default_run_config['lr'])
    argparser.add_argument('--dropout', type=float,
                           default=default_run_config['dropout'])
    argparser.add_argument('--report-acc', type=int,
                           default=0)

    return vars(argparser.parse_args())


def get_run_config():
    run_config = {}

    run_config.update(get_default_common_config())
    run_config['arch'] = 'arch3'
    run_config['sample_type'] = 'khop2'

    run_config['fanout'] = [25, 10]
    run_config['lr'] = 0.003
    run_config['dropout'] = 0.5

    run_config.update(parse_args(run_config))

    process_common_config(run_config)
    assert(run_config['sample_type'] != 'random_walk')

    run_config['num_fanout'] = run_config['num_layer'] = len(
        run_config['fanout'])

    print_run_config(run_config)

    if run_config['validate_configs']:
        sys.exit()

    return run_config


def run():
    run_config = get_run_config()

    sam.config(run_config)
    sam.init()

    sam.backend_init_model()

    # sam.report_init()


    in_feat = sam.feat_dim()
    num_class = sam.num_class()
    num_layer = run_config['num_layer']

    num_epoch = sam.num_epoch()
    num_step = sam.steps_per_epoch()

    epoch_sample_times  = [0 for i in range(num_epoch)]
    epoch_copy_times    = [0 for i in range(num_epoch)]
    epoch_convert_times = [0 for i in range(num_epoch)]
    epoch_train_times   = [0 for i in range(num_epoch)]
    epoch_total_times   = [0 for i in range(num_epoch)]

    # sample_times  = [0 for i in range(num_epoch * num_step)]
    # copy_times    = [0 for i in range(num_epoch * num_step)]
    # convert_times = [0 for i in range(num_epoch * num_step)]
    # train_times   = [0 for i in range(num_epoch * num_step)]
    # total_times   = [0 for i in range(num_epoch * num_step)]
    # num_nodes     = [0 for i in range(num_epoch * num_step)]
    # num_samples   = [0 for i in range(num_epoch * num_step)]

    cur_step_key = 0
    global batch_key
    for epoch in range(num_epoch):
        for step in range(num_step):
            t0 = time.time()
            sam.trace_step_begin_now (epoch * num_step + step, sam.kL0Event_Train_Step)
            if not run_config['pipeline']:
                sam.sample_once()
            elif epoch + step == 0:
                sam.start()
            batch_key = sam.get_next_batch()
            t1 = time.time()
            t2 = time.time()
            sam.trace_step_begin_now (batch_key, sam.kL1Event_Train)
            sam.backend_train_current_batch()

            sam.trace_step_end_now   (batch_key, sam.kL1Event_Train)
            t3 = time.time()
            sam.trace_step_end_now (epoch * num_step + step, sam.kL0Event_Train_Step)

            # sample_time = sam.get_log_step_value(epoch, step, sam.kLogL1SampleTime)
            # copy_time = sam.get_log_step_value(epoch, step, sam.kLogL1CopyTime)
            train_time = t3 - t2
            total_time = t3 - t0

            # num_node = sam.get_log_step_value(epoch, step, sam.kLogL1NumNode)
            # num_sample = sam.get_log_step_value(epoch, step, sam.kLogL1NumSample)

            sam.log_step(epoch, step, sam.kLogL1TrainTime,   train_time)
            sam.log_epoch_add(epoch, sam.kLogEpochTrainTime,   train_time)
            sam.log_epoch_add(epoch, sam.kLogEpochTotalTime,   total_time)

            # sample_times  [cur_step_key] = sample_time
            # copy_times    [cur_step_key] = copy_time
            # convert_times [cur_step_key] = convert_time
            # train_times   [cur_step_key] = train_time
            # total_times   [cur_step_key] = total_time

            # num_nodes     [cur_step_key] = num_node
            # num_samples   [cur_step_key] = num_sample

            # print('Epoch {:05d} | Step {:05d} | Nodes {:.0f} | Samples {:.0f} | Time {:.4f} secs | Sample Time {:.4f} secs | Copy Time {:.4f} secs |  Train Time {:.4f} secs (Convert Time {:.4f} secs) | Loss {:.4f} '.format(
            #     epoch, step, num_node, num_sample, total_time,
            #         sample_time, copy_time, train_time, convert_time, loss
            # ))

            # sam.report_step_average(epoch, step)
            # sam.report_step(epoch, step)
            cur_step_key += 1
        # sam.report_epoch_average(epoch)

        epoch_sample_times  [epoch] =  sam.get_log_epoch_value(epoch, sam.kLogEpochSampleTime)
        epoch_copy_times    [epoch] =  sam.get_log_epoch_value(epoch, sam.kLogEpochCopyTime)
        epoch_convert_times [epoch] =  sam.get_log_epoch_value(epoch, sam.kLogEpochConvertTime)
        epoch_train_times   [epoch] =  sam.get_log_epoch_value(epoch, sam.kLogEpochTrainTime)
        epoch_total_times   [epoch] =  sam.get_log_epoch_value(epoch, sam.kLogEpochTotalTime)
        sam.forward_barrier()

    sam.report_step_average(num_epoch - 1, num_step - 1)
    print('Avg Epoch Time {:.4f} | Sample Time {:.4f} | Copy Time {:.4f} | Convert Time {:.4f} | Train Time {:.4f}'.format(
        np.mean(epoch_total_times[1:]),  np.mean(epoch_sample_times[1:]),  np.mean(epoch_copy_times[1:]), np.mean(epoch_convert_times[1:]), np.mean(epoch_train_times[1:])))

    sam.report_node_access()
    sam.dump_trace()
    sam.shutdown()


if __name__ == '__main__':
    run()
