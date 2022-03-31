"""
  Copyright 2022 Institute of Parallel and Distributed Systems, Shanghai Jiao Tong University
  
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
  
      http://www.apache.org/licenses/LICENSE-2.0
  
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
"""

from typing import BinaryIO


TB = 1 * 1024 * 1024 * 1024 * 1024
GB = 1 * 1024 * 1024 * 1024
MB = 1 * 1024 * 1024
KB = 1 * 1024

BILLION = 1 * 10**9
MILLION = 1 * 10**6
THOUSAND = 1 * 10**3
INT_BYTES = 4
FLOAT_BYTES = 4
LONG_BYTES = 8


def sz_format(sz):
    if sz >= TB:
        return "{:<6.2f} TB".format(sz / TB)
    elif sz >= GB:
        return "{:<6.2f} GB".format(sz / GB)
    elif sz >= MB:
        return "{:<6.2f} MB".format(sz / MB)
    elif sz >= KB:
        return "{:<6.2f} KB".format(sz / KB)
    else:
        return "{:<6.2f} B".format(sz)


class Graph:
    def __init__(self, name, num_nodes, num_edges, feat_dim):
        self.name = name
        self.num_nodes = num_nodes * 1.0
        self.num_edges = num_edges * 1.0
        self.feat_dim = feat_dim * 1.0

    def __str__(self):
        graph_sz = sz_format(
            ((self.num_nodes + 1) + self.num_edges) * INT_BYTES)
        feat_sz = sz_format((self.num_nodes * self.feat_dim) * FLOAT_BYTES)

        return "{:15s} | topology: {:s} | feat: {:s}".format(self.name, graph_sz, feat_sz)


Reddit = Graph("Reddit", 232965, 114615892, 602)
Products = Graph("Products", 2449029, 123718152, 100)
Papers100M = Graph('Papers100M', 111059956, 1615685872, 128)
ComFriendster = Graph('com-friendster', 65608366, 3612134270, 140)
AlipayGraph = Graph('AlipayGraph', 4 * BILLION, 26 * BILLION, 128 * 4)
Amazon = Graph('Amazon', 65 * MILLION, 3.6 * MILLION, 300)
Mag240M_lsc = Graph('Mag240M_lsc', 121 * MILLION, 1.2 * BILLION, 768)
Twitter = Graph('Twitter', 41652230, 1468365182, 256)
Uk = Graph('UK-2006-06', 77741046, 2965197340, 256)

graph_list = [Reddit, Products, Papers100M, ComFriendster,
              AlipayGraph, Amazon, Mag240M_lsc, Twitter, Uk]


def run():
    for graph in graph_list:
        print(graph)


if __name__ == '__main__':
    run()
