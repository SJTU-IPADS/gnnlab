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
import torch

import numpy as np
from scipy.sparse import coo_matrix
from torch_sparse import SparseTensor
import dgl
import efficient_converter

DOWNLOAD_URL = 'https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/mag240m_kddcup2021.zip'

ROOT_DIR = '/graph-learning'
RAW_DATA_DIR = f'{ROOT_DIR}/data-raw'
MAG240M_RAW_DATA_DIR = f'{RAW_DATA_DIR}/mag240m_kddcup2021'
OUTPUT_DATA_DIR = f'{ROOT_DIR}/samgraph/mag240m-homo'

meta_dict = {
    'NUM_NODE'      : None,
    'NUM_EDGE'      : None,
    'FEAT_DIM'      : None,
    'NUM_CLASS'     : None,
    'NUM_TRAIN_SET' : None,
    'NUM_VALID_SET' : None,
    'NUM_TEST_SET'  : None,
}

def download_data():
    print('Download data...')
    if not os.path.exists(f'{RAW_DATA_DIR}/mag240m_kddcup2021.zip'):
        print('Start downloading...')
        assert(os.system(f'wget {DOWNLOAD_URL} -O {RAW_DATA_DIR}/mag240m_kddcup2021.zip') == 0)
    else:
        print('Already downloaded.')

    print('Unzip data...')
    if not os.path.exists(f'{MAG240M_RAW_DATA_DIR}/unzipped'):
        print('Start unziping...')
        assert(os.system(f'cd {RAW_DATA_DIR}; unzip {RAW_DATA_DIR}/mag240m_kddcup2021.zip') == 0)
        assert(os.system(f'touch {MAG240M_RAW_DATA_DIR}/unzipped') == 0)
    else:
        print('Already unzipped...')

def convert():
    meta = torch.load(f'{MAG240M_RAW_DATA_DIR}/meta.pt')
    print(meta)
    num_paper = meta['paper']
    num_author = meta['author']
    num_institution = meta['institution']
    if os.path.exists(f'{OUTPUT_DATA_DIR}/indices.bin'):
        indices = np.memmap(f'{OUTPUT_DATA_DIR}/indices.bin', dtype='uint32', mode='r')
        global meta_dict
        meta_dict['NUM_EDGE'] = indices.shape[0]
        return

    print('Reading raw data...')
    paper_edge_index = np.load(f'{MAG240M_RAW_DATA_DIR}/processed/paper___cites___paper/edge_index.npy').astype('uint32')
    write_edge_index = np.load(f'{MAG240M_RAW_DATA_DIR}/processed/author___writes___paper/edge_index.npy').astype('uint32')
    affil_edge_index = np.load(f'{MAG240M_RAW_DATA_DIR}/processed/author___affiliated_with___institution/edge_index.npy').astype('uint32')

    p_c_p_coo = efficient_converter.TinyCOO(paper_edge_index[0], paper_edge_index[1])
    a_w_p_coo = efficient_converter.TinyCOO(write_edge_index[0], write_edge_index[1])
    a_a_i_coo = efficient_converter.TinyCOO(affil_edge_index[0], affil_edge_index[1])

    p_c_p_coo_T = efficient_converter.TinyCOO(p_c_p_coo.col, p_c_p_coo.row)
    a_w_p_coo_T = efficient_converter.TinyCOO(a_w_p_coo.col, a_w_p_coo.row)
    a_a_i_coo_T = efficient_converter.TinyCOO(a_a_i_coo.col, a_a_i_coo.row)

    print('sorting coo transpose')
    efficient_converter.COOUtil.sort_coo_row_only(p_c_p_coo_T)
    efficient_converter.COOUtil.to_cpu(p_c_p_coo_T)
    efficient_converter.COOUtil.sort_coo_row_only(a_w_p_coo_T)
    efficient_converter.COOUtil.to_cpu(a_w_p_coo_T)
    efficient_converter.COOUtil.sort_coo_row_only(a_a_i_coo_T)
    efficient_converter.COOUtil.to_cpu(a_a_i_coo_T)

    # line1: paper_cite_paper, paper_cite_paper_T, author_write_paper_T
    print('line1')
    print(f'orig cite has {p_c_p_coo.row.shape[0]} edges')
    a_w_p_coo_T.col += num_paper
    line1 = efficient_converter.chunk_sparse_add(p_c_p_coo, p_c_p_coo_T, num_paper, True, True, chunk_size=1024*1024*16)
    print(f'pcp symmetric has {line1.row.shape[0]} edges')
    line1 = efficient_converter.chunk_sparse_add(line1, a_w_p_coo_T, num_paper, True, False, chunk_size=1024*1024*16)

    # line2: authoer_write_paper, author_aff_insti
    print('line2')
    a_a_i_coo.col += num_paper+num_author
    line2 = efficient_converter.chunk_sparse_multi_add([a_w_p_coo, a_a_i_coo], num_author, True, False, chunk_size=1024*1024*16)
    line2.row += num_paper

    #line3: author_aff_insti_T
    print('line3')
    a_a_i_coo_T.row += num_paper + num_author
    a_a_i_coo_T.col += num_paper
    line3 = a_a_i_coo_T

    print('concating')
    final_coo = efficient_converter.concat_coo([line1, line2, line3])
    print(f'total edge: {final_coo.row.shape[0]}')
    meta_dict['NUM_EDGE'] = final_coo.row.shape[0]

    # final_coo.row.tofile(f'{OUTPUT_DATA_DIR}/row.bin')

    indptr = efficient_converter.get_indptr_from_sorted_coo(final_coo.row, num_paper + num_author + num_institution)
    indptr.astype('uint32').tofile(f'{OUTPUT_DATA_DIR}/indptr.bin')
    final_coo.col.tofile(f'{OUTPUT_DATA_DIR}/indices.bin')

def write_meta():
    print('Writing meta file...')
    meta = torch.load(f'{MAG240M_RAW_DATA_DIR}/meta.pt')
    print(meta)
    num_paper = meta['paper']
    num_author = meta['author']
    num_institution = meta['institution']

    split_dict = torch.load(f'{MAG240M_RAW_DATA_DIR}/split_dict.pt')
    train_idx = split_dict['train']
    valid_idx = split_dict['valid']
    test_idx = split_dict['test']
    paper_features = np.load(f'{MAG240M_RAW_DATA_DIR}/processed/paper/node_feat.npy', mmap_mode='r')

    global meta_dict
    meta_dict['NUM_NODE']      = num_paper + num_author + num_institution
    # meta_dict['NUM_EDGE']      = src.shape[0]
    meta_dict['FEAT_DIM']      = paper_features.shape[1]
    meta_dict['NUM_CLASS']     = meta['num_classes']
    meta_dict['NUM_TRAIN_SET'] = train_idx.shape[0]
    meta_dict['NUM_VALID_SET'] = valid_idx.shape[0]
    meta_dict['NUM_TEST_SET']  = test_idx.shape[0]
    meta_dict['FEAT_DATA_TYPE'] = 'F16'
    with open(f'{OUTPUT_DATA_DIR}/meta.txt', 'w') as f:
        f.write('{}\t{}\n'.format('NUM_NODE',       meta_dict['NUM_NODE']))
        f.write('{}\t{}\n'.format('NUM_EDGE',       meta_dict['NUM_EDGE']))
        f.write('{}\t{}\n'.format('FEAT_DIM',       meta_dict['FEAT_DIM']))
        f.write('{}\t{}\n'.format('NUM_CLASS',      meta_dict['NUM_CLASS']))
        f.write('{}\t{}\n'.format('NUM_TRAIN_SET',  meta_dict['NUM_TRAIN_SET']))
        f.write('{}\t{}\n'.format('NUM_VALID_SET',  meta_dict['NUM_VALID_SET']))
        f.write('{}\t{}\n'.format('NUM_TEST_SET',   meta_dict['NUM_TEST_SET']))
        f.write('{}\t{}\n'.format('FEAT_DATA_TYPE', meta_dict['FEAT_DATA_TYPE']))

def write_other():
    meta = torch.load(f'{MAG240M_RAW_DATA_DIR}/meta.pt')
    print(meta)
    num_paper = meta['paper']
    num_author = meta['author']
    num_institution = meta['institution']
    paper_label = np.load(f'{MAG240M_RAW_DATA_DIR}/processed/paper/node_label.npy')
    paper_label=np.pad(paper_label, (0, num_author + num_institution), 'constant', constant_values=(0,))
    split_dict = torch.load(f'{MAG240M_RAW_DATA_DIR}/split_dict.pt')
    train_idx = split_dict['train']
    valid_idx = split_dict['valid']
    test_idx = split_dict['test']
    train_idx.astype('uint32').tofile(f'{OUTPUT_DATA_DIR}/train_set.bin')
    valid_idx.astype('uint32').tofile(f'{OUTPUT_DATA_DIR}/valid_set.bin')
    test_idx.astype('uint32').tofile(f'{OUTPUT_DATA_DIR}/test_set.bin')
    paper_label.astype('uint64').tofile(f'{OUTPUT_DATA_DIR}/label.bin')


if __name__ == '__main__':
    assert(os.system(f'mkdir -p {MAG240M_RAW_DATA_DIR}') == 0)
    assert(os.system(f'mkdir -p {OUTPUT_DATA_DIR}') == 0)

    download_data()
    convert()
    write_meta()
    write_other()
