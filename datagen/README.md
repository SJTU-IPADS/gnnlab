# Datagen

This guide shows how to get the original datasets and convert them into the format that FGNN can read. We convert those data into binary format so that GNN systems can read the data very fast by using `MMAP`.



Default dataset path is `/graph-learning/samgraph/{dataset name}`.  `/graph-learning/samgraph/` is the default dataset root.

```sh
> tree -L 2 /graph-learning
/graph-learning
├── data-raw                    # original downloaded dataset  
│   ├── papers100M-bin
│   ├── papers100M-bin.zip
│   ├── products
│   ├── products.zip
│   ├── twitter
│   └── uk-2006-05
└── samgraph                   # The converted dataset
    ├── papers100M
    ├── products
    ├── twitter
    └── uk-2006-05
```



```sh
> tree /graph-learning/samgraph/papers100M
/graph-learning/samgraph/papers100M
├── cache_by_degree.bin      # vertexid sorted by cache rank(Higher rank, higher oppotunity to be cached)
├── feat.bin                 # vertex feature binary data
├── indices64.bin            # csr indices stored as uint64
├── indices.bin              # csr indices stored as uint32
├── indptr64.bin             # csr indptr stored as uint64
├── indptr.bin               # csr indptr stored as uint32
├── label.bin                # vertex label binary data
├── meta.txt                 # dataset meta data
├── test_set64.bin           # testset node id list as uint64
├── test_set.bin             # testset node id list as uint32
├── train_set64.bin          # trainset node id list as uint64
├── train_set.bin            # trainset node id list as uint32
├── valid_set64.bin          # trainset node id list as uint64 
└── valid_set.bin            # validset node id list as uint32

0 directories, 14 files
```



## Disk Space Requirement

To store all the four datasets, your disk should have at least **128GB** of free space.

```
> du -h --max-depth 1 /graph-learning/samgraph
35G     /graph-learning/samgraph/uk-2006-05
2.4G    /graph-learning/samgraph/products
74G     /graph-learning/samgraph/papers100M
18G     /graph-learning/samgraph/twitter
128G    /graph-learning/samgraph
```



## 1. Download And Convert

Create the dataset directory:

```sh
sudo mkdir -p /graph-learning/samgraph
sudo mkdir -p /graph-learning/data-raw
sudo chmod -R 777 /graph-learning
```



Download the dataset and convert them into binary format:

```sh
cd samgraph/datagen

python products.py
python papers100M.py
bash twitter.sh
bash uk-2006-05.sh
```



Now we have:

```sh
> tree /graph-learning/samgraph/papers100M
/graph-learning/samgraph/papers100M
├── feat.bin
├── indices.bin
├── indptr.bin
├── label.bin
├── meta.txt
├── test_set.bin
├── train_set.bin
└── valid_set.bin
```



## 2. Generate uint64 Vertex ID Format 

In step1, the vertex IDs are encoded as uint32. However, PyG requires the vertex ID to be uint64. We need to generate a uint64 version for every dataset.

```
cd samgraph/utility/data-process

mkdir build
cd build

cmake ..
make 32to64 -j

./32to64 -g products
./32to64 -g papers100M
./32to64 -g twitter
./32to64 -g uk-2006-05
```



Now we have:

```sh
> tree /graph-learning/samgraph/papers100M
/graph-learning/samgraph/papers100M
├── feat.bin
├── indices64.bin           # new added
├── indices.bin
├── indptr64.bin            # new added
├── indptr.bin
├── label.bin
├── meta.txt
├── test_set64.bin          # new added
├── test_set.bin
├── train_set64.bin         # new added
├── train_set.bin
└── valid_set64.bin         # new added
└── valid_set.bin
```



## 3. Generate Cache Rank Table

The degree-based cache policy uses the out-degree as cache rank. The ranking only needs to be preprocessed once. The cache rank table is a sorted vertex-id list by their out-degree.

```sh
cd samgraph/utility/data-process/build

make cache-by-degree cache-by-random  -j

# degree-based cache policy
./cache-by-degree -g products
./cache-by-degree -g papers100M
./cache-by-degree -g twitter
./cache-by-degree -g uk-2006-05

# random cache policy
./cache-by-random -g products
./cache-by-random -g papers100M
./cache-by-random -g twitter
./cache-by-random -g uk-2006-05
```



Now we have:

```sh
/graph-learning/samgraph/papers100M
├── cache_by_degree.bin   # new added
├── cache_by_random.bin   # new added
├── feat.bin
├── indices64.bin
├── indices.bin
├── indptr64.bin
├── indptr.bin
├── label.bin
├── meta.txt
├── test_set64.bin
├── test_set.bin
├── train_set64.bin
├── train_set.bin
└── valid_set64.bin
└── valid_set.bin
```



## 4. Generate prob-prefix-table for Weighted-Sampling

Since the original datasets have no edge weights, we need to manually generate the edge weights.


```sh
cd samgraph/utility/data-process/build

make create-prob-prefix-table -j

./create-prob-prefix-table -g products
./create-prob-prefix-table -g papers100M
./create-prob-prefix-table -g twitter
./create-prob-prefix-table -g uk-2006-05
```



Now we have:

```sh
/graph-learning/samgraph/papers100M
├── cache_by_degree.bin
├── feat.bin
├── indices64.bin
├── indices.bin
├── indptr64.bin
├── indptr.bin
├── label.bin
├── meta.txt
├── prob_prefix_table.bin   #new added
├── test_set64.bin
├── test_set.bin
├── train_set64.bin
├── train_set.bin
└── valid_set64.bin
└── valid_set.bin
```