import os


class MetaReader(object):
    def __init__(self):
        pass

    def read(self, folder):
        meta = {}
        with open(os.path.join(folder, 'meta.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split()
                assert len(line) == 2
                meta[line[0]] = int(line[1])

        meta_keys = meta.keys()

        assert('NUM_NODE' in meta_keys)
        assert('NUM_EDGE' in meta_keys)
        assert('FEAT_DIM' in meta_keys)
        assert('NUM_CLASS' in meta_keys)
        assert('NUM_TRAIN_SET' in meta_keys)
        assert('NUM_VALID_SET' in meta_keys)
        assert('NUM_TEST_SET' in meta_keys)

        return meta
