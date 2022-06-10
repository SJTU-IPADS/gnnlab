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

import os

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
import torch, cupy

import efficient_converter

INPUT_DATA_DIR = '/graph-learning/samgraph/papers100M'
OUTPUT_DATA_DIR = '/graph-learning/samgraph/papers100M-undir'

def paper_symmetric_cpu():
    indptr = np.memmap(os.path.join(INPUT_DATA_DIR, 'indptr.bin'), dtype='int32', mode='r')
    indices = np.memmap(os.path.join(INPUT_DATA_DIR, 'indices.bin'), dtype='int32', mode='r')
    global num_nodes
    global num_edges
    num_nodes = indptr.shape[0] - 1
    num_edges = indices.shape[0]
    data = np.ones(num_edges)

    print('Converting data...')
    csr = csr_matrix((data, indices, indptr), shape=(num_nodes, num_nodes))
    csr += csr.transpose(copy=True)

    indptr = csr.indptr
    indices = csr.indices

    num_edges = indices.shape[0]

    print('Writing files...')
    indptr.astype('uint32').tofile(f'{OUTPUT_DATA_DIR}/indptr.bin')
    indices.astype('uint32').tofile(f'{OUTPUT_DATA_DIR}/indices.bin')

def paper_symmetric_helper_chunk():
    global num_nodes
    global num_edges
    # 1. csc/csr to coo
    indptr  = cupy.asarray(np.memmap(os.path.join(INPUT_DATA_DIR, 'indptr.bin'),  dtype='uint32', mode='r'))
    indices = cupy.asarray(np.memmap(os.path.join(INPUT_DATA_DIR, 'indices.bin'), dtype='uint32', mode='r'))
    num_rows = indptr.shape[0] - 1
    data = cupy.ones(indices.shape)
    coo = cupy.sparse.csr_matrix((data, indices, indptr)).tocoo()
    num_nodes = indptr.shape[0] - 1
    print(coo.row.dtype)
    coo1 = efficient_converter.TinyCOO(coo.row, coo.col)
    coo2 = efficient_converter.TinyCOO(coo.col, coo.row)

    del coo
    del indptr
    del indices
    del data
    efficient_converter.COOUtil.to_cpu(coo1)
    efficient_converter.COOUtil.to_cpu(coo2)
    efficient_converter.clean_cuda_cache()
    efficient_converter.COOUtil.sort_coo_row_only(coo2)
    efficient_converter.COOUtil.to_cpu(coo2)
    coo = efficient_converter.chunk_sparse_add(coo1, coo2, num_rows, chunk_size=1024*1024)

    indptr = efficient_converter.get_indptr_from_sorted_coo(coo.row, num_nodes)
    indices = coo.col
    num_edges = indices.shape[0]
    print('Writing files...')
    indptr.astype('uint32').tofile(f'{OUTPUT_DATA_DIR}/indptr.bin')
    indices.astype('uint32').tofile(f'{OUTPUT_DATA_DIR}/indices.bin')

def validate_eq(a, b):
    a = cupy.asarray(a)
    b = cupy.asarray(b)
    cupy.testing.assert_allclose(a, b)
    del a, b
    efficient_converter.clean_cuda_cache()

def write_meta():
    print('Writing meta file...')
    with open(f'{OUTPUT_DATA_DIR}/meta.txt', 'w') as f:
        f.write('{}\t{}\n'.format('NUM_NODE', num_nodes))
        f.write('{}\t{}\n'.format('NUM_EDGE', num_edges))
        f.write('{}\t{}\n'.format('FEAT_DIM', 128))
        f.write('{}\t{}\n'.format('NUM_CLASS', 172))
        f.write('{}\t{}\n'.format('NUM_TRAIN_SET', 1207179))
        f.write('{}\t{}\n'.format('NUM_VALID_SET', 125265))
        f.write('{}\t{}\n'.format('NUM_TEST_SET', 214338))


if __name__ == '__main__':
    assert(os.system(f'mkdir -p {OUTPUT_DATA_DIR}') == 0)
    # paper_symmetric_cpu()
    paper_symmetric_helper_chunk()
    write_meta()
