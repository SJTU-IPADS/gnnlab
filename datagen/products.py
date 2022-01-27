import os

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

DOWNLOAD_URL = 'http://snap.stanford.edu/ogb/data/nodeproppred/products.zip'

RAW_DATA_DIR = '/graph-learning/data-raw'
PRODUCTS_RAW_DATA_DIR = f'{RAW_DATA_DIR}/products'
OUTPUT_DATA_DIR = '/graph-learning/samgraph/products'


def download_data():
    print('Download data...')
    if not os.path.exists(f'{RAW_DATA_DIR}/products.zip'):
        assert(os.system(
            f'wget {DOWNLOAD_URL} -O {RAW_DATA_DIR}/products.zip') == 0)
    else:
        print('Already downloaded.')

    print('Unzip data...')
    if not os.path.exists(f'{PRODUCTS_RAW_DATA_DIR}/unzipped'):
        assert(os.system(f'cd {RAW_DATA_DIR}; unzip {RAW_DATA_DIR}/products.zip') == 0)
        assert(os.system(f'touch {PRODUCTS_RAW_DATA_DIR}/unzipped') == 0)
    else:
        print('Already unzipped...')


def convert_data():
    print('Reading raw data...')
    edges = pd.read_csv(f'{PRODUCTS_RAW_DATA_DIR}/raw/edge.csv.gz',
                        compression='gzip', header=None).values.T
    feature = pd.read_csv(f'{PRODUCTS_RAW_DATA_DIR}/raw/node-feat.csv.gz',
                          compression='gzip', header=None).values
    label = pd.read_csv(f'{PRODUCTS_RAW_DATA_DIR}/raw/node-label.csv.gz',
                        compression='gzip', header=None).values.T[0]

    train_idx = pd.read_csv(f'{PRODUCTS_RAW_DATA_DIR}/split/sales_ranking/train.csv.gz',
                            compression='gzip', header=None).values.T[0]
    valid_idx = pd.read_csv(f'{PRODUCTS_RAW_DATA_DIR}/split/sales_ranking/valid.csv.gz',
                            compression='gzip', header=None).values.T[0]
    test_idx = pd.read_csv(f'{PRODUCTS_RAW_DATA_DIR}/split/sales_ranking/test.csv.gz',
                           compression='gzip', header=None).values.T[0]

    print('Converting data...')
    # products is undirected, so we have to double its edges
    src = np.concatenate((edges[0], edges[1]))
    dst = np.concatenate((edges[1], edges[0]))
    data = np.zeros(src.shape)

    # products is undirected, so we don't have to swap src and dst
    coo = coo_matrix((data, (src, dst)), shape=(
        feature.shape[0], feature.shape[0]), dtype=np.uint32)
    csr = coo.tocsr()
    indptr = csr.indptr
    indices = csr.indices

    print('Writing files...')
    indptr.astype('uint32').tofile(f'{OUTPUT_DATA_DIR}/indptr.bin')
    indices.astype('uint32').tofile(f'{OUTPUT_DATA_DIR}/indices.bin')
    train_idx.astype('uint32').tofile(f'{OUTPUT_DATA_DIR}/train_set.bin')
    valid_idx.astype('uint32').tofile(f'{OUTPUT_DATA_DIR}/valid_set.bin')
    test_idx.astype('uint32').tofile(f'{OUTPUT_DATA_DIR}/test_set.bin')
    feature.astype('float32').tofile(f'{OUTPUT_DATA_DIR}/feat.bin')
    label.astype('uint64').tofile(f'{OUTPUT_DATA_DIR}/label.bin')


def write_meta():
    print('Writing meta file...')
    with open(f'{OUTPUT_DATA_DIR}/meta.txt', 'w') as f:
        f.write('{}\t{}\n'.format('NUM_NODE', 2449029))
        f.write('{}\t{}\n'.format('NUM_EDGE', 123718152))
        f.write('{}\t{}\n'.format('FEAT_DIM', 100))
        f.write('{}\t{}\n'.format('NUM_CLASS', 47))
        f.write('{}\t{}\n'.format('NUM_TRAIN_SET', 196615))
        f.write('{}\t{}\n'.format('NUM_VALID_SET', 39323))
        f.write('{}\t{}\n'.format('NUM_TEST_SET', 2213091))


if __name__ == '__main__':
    assert(os.system(f'mkdir -p {PRODUCTS_RAW_DATA_DIR}'))
    assert(os.system(f'mkdir -p {OUTPUT_DATA_DIR}') == 0)

    download_data()
    convert_data()
    write_meta()
