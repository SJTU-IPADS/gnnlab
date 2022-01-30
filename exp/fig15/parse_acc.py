#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import argparse
import re

def parse_args():
    argparser = argparse.ArgumentParser('Acc Timeline Parser')
    argparser.add_argument('-f', '--file', type=str,
            help='the log file path to parse')
    argparser.add_argument('--system', choices=['dgl', 'fgnn'],
            type=str, help='the system name of this test, like dgl/fgnn')
    argparser.add_argument('--dataset', choices=['papers', 'products'],
            type=str, help='the dataset of this test')
    argparser.add_argument('--batch-size', type=int,
            help='the batch size of this test')
    ret = vars(argparser.parse_args())
    if (ret['file'] == None):
        argparser.error('Add --file argument')
    if (ret['system'] == None):
        argparser.error('Add --system argument')
    if (ret['dataset'] == None):
        argparser.error('Add --dataset argument')
    if (ret['batch_size'] == None):
        argparser.error('Add --batch-size argument')
    return ret

def parse_data(file_name, system, dataset, batch_size,
               pattern = r'^Valid Acc: (.+)\% \| .* \| Time Cost: (.+)'):
    with open(file_name, 'r') as file:
        for line in file:
            m = re.match(pattern, line)
            if m:
                # print('{} {}'.format(m.group(1), m.group(2)))
                # system(like dgl) dataset batch_size time acc
                print('{}\t{}\t{}\t{}\t{}'.format(
                    system, dataset, batch_size, m.group(2), m.group(1)))


if __name__ == '__main__':
    arguments   = parse_args()
    file_name   = arguments['file']
    system      = arguments['system']
    dataset     = arguments['dataset']
    batch_size  = arguments['batch_size']
    parse_data(file_name, system, dataset, batch_size)
