import argparse
import time

import samgraph.torch as sam


def run(args):
    fanout_list = [int(fanout) for fanout in args.fan_out.split(',')]

    sam.init(args.dataset_path, args.sample_device, args.train_device,
             args.batch_size, fanout_list, args.num_epoch)
    num_epoch = sam.num_epoch()
    num_step = sam.num_step_per_epoch()

    # sam.start()
    num_graph = len(fanout_list)
    for epoch in range(num_epoch):

        tic_step = time.time()
        for step in range(num_step):
            sam.sample()
            sam.get_next_batch(epoch, step, num_graph)

            print('Epoch {:05d} | Step {:05d} | Time {:.4f} secs'.format(
                epoch, step, time.time() - tic_step
            ))
            tic_step = time.time()

    sam.shutdown()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("GraphSage Training")
    argparser.add_argument('--train-device', type=int, default=0,
                           help="")
    argparser.add_argument('--sample-device', type=int, default=1)
    argparser.add_argument('--dataset-path', type=str,
                           default='/graph-learning/samgraph/papers100M')
    argparser.add_argument('--num-epoch', type=int, default=2)
    argparser.add_argument('--num-hidden', type=int, default=256)
    argparser.add_argument('--fan-out', type=str, default='15,10,5')
    argparser.add_argument('--batch-size', type=int, default=8192)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)

    args = argparser.parse_args()
    run(args)
