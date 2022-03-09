import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

if __name__ == '__main__':
    src_dir = '/graph-learning/data-raw/reddit/'
    output_dir = '/graph-learning/samgraph/reddit/'

    file0 = np.load(src_dir + 'reddit_graph.npz')
    file1 = np.load(src_dir + 'reddit_data.npz')

    feature = file1['feature']
    label = file1['label']
    node_ids = file1['node_types']

    row = file0['row']
    col = file0['col']
    data = file0['data']

    train_idx = np.where(node_ids == 1)[0]
    valid_idx = np.where(node_ids == 2)[0]
    test_idx = np.where(node_ids == 3)[0]

    # swap col and row to make csc graph
    coo = coo_matrix((data, (col, row)), shape=(feature.shape[0], feature.shape[0]),dtype=np.uint32)
    csr = coo.tocsr()

    indptr = csr.indptr
    indices = csr.indices


    indptr.astype('uint32').tofile(output_dir + 'indptr.bin')
    indices.astype('uint32').tofile(output_dir + 'indices.bin')


    train_idx.astype('uint32').tofile(output_dir + 'train_set.bin')
    valid_idx.astype('uint32').tofile(output_dir + 'valid_set.bin')
    test_idx.astype('uint32').tofile(output_dir + 'test_set.bin')

    feature.astype('float32').tofile(output_dir + 'feat.bin')
    label.astype('uint64').tofile(output_dir + 'label.bin')

    with open(f'{output_dir}meta.txt', 'w') as f:
        f.write('{}\t{}\n'.format('NUM_NODE', 232965))
        f.write('{}\t{}\n'.format('NUM_EDGE', 114615892))
        f.write('{}\t{}\n'.format('FEAT_DIM', 602))
        f.write('{}\t{}\n'.format('NUM_CLASS', 41))
        f.write('{}\t{}\n'.format('NUM_TRAIN_SET', 153431))
        f.write('{}\t{}\n'.format('NUM_VALID_SET', 23831))
        f.write('{}\t{}\n'.format('NUM_TEST_SET', 55703))
