# SamGraph

SamGraph is a high-performance GPU-based graph sampler for deep graph learning application

| dgl ||why(if ✘)|
|-|-|-|
| dgl/train_gat.py | ✘ | `'feat'` KeyError |
| dgl/train_gcn.py | ✔ ||
| dgl/train_graphsage.py | ✔ ||
| dgl/train_pinsage.py | ✔ ||


|sam||why & fix ✘(if ✔)|
|-|-|-|
|samgraph/train_gat.py|✘|`sam.simple_hashtable`|
|samgraph/train_gcd.py|✔||
|samgraph/train_graphsage.py|✔|train with 'cuda:0'|
|samgraph/train_pinsage.py|✔|train with 'cuda:0'|

|sam/sgnn||why & fix ✘(if ✔)|
|-|-|-|
|samgraph/sgnn/train_gcd.py|✔||
|samgraph/sgnn/train_graphsage.py|✔||
|samgraph/sgnn/train_pinsage.py|✔||

|sam/sgnn_dgl||why & fix ✘(if ✔)|
|-|-|-|
|samgraph/sgnn_dgl/train_gcd.py|✔||
|samgraph/sgnn_dgl/train_graphsage.py|✔||
|samgraph/sgnn_dgl/train_pinsage.py|✔||

|pyg||why & fix ✘(if ✔)|
|-|-|-|
|pyg/train_gcn.py|✔|train with 'cuda:0'|
|pyg/train_graphsage.py|✔|train with 'cuda:0'|

|auto_runner|fail_count|
|-|-|
|run_dgl.py|0/1|
|run_pyg.py|-|
|run_samgraph.py|34/36, !{0, 18}|
|rnn_sgnn_dgl.py|14/16, !{0, 8}|
|run_sgnn.py|7/8, !{0}|

multi-gpu： NCCL

# TODO

- [ ] samgraph
    - [x] run examples
    - [ ] understanding implemention (on doing)
      - [x] arch 3
    - [ ] ...
- [x] NextDoor paper    
- [ ] CUDA C++ Programming Guide
  - [x] chapter 1
  - [x] chapter 2
  - [ ] chapter 3
  - [x] chapter 4
  - [ ] chapter 5
- [ ] large graph sampling
  - [ ] create subgraph
    - [ ] ...
  - [ ] single-gpu
    - [ ] ...
  - [ ] multi-gpu


# What's going on

## comm

there are three `DeviceType`: `CPU`, `GPU`, `MMAP`  

> `CPU` load data to cpu, `GPU` load data to GPU, `MMAP` raw mmap ptr

`Context` describe devices.

`config` just copy config to `RunConfig` and check configs.

the difference between arch0-7

```
arch0: vanilla mode(CPU sampling + GPU training)
arch1: standalone mode (single GPU for both sampling and training)
arch2: offload mode (offload the feature extraction to CPU)
arch3: dedicated mode (dedicated GPU for sampling and training)
TODO:  is it a prefetch mode ?
arch4: prefetch mode
arch5: distributed mode (CPU/GPU sampling + multi-GPUs traning)
arch6: sgnn mode
arch7: sgnn mode but use pytorch extracting
```

`Engine` contains almost everything. `Engine` will be created firstly, this virtually does nothing.

then engine will be initialized. `init`, `data_init` are virtual functions.

`GPUEngine::init`, assuming only consider arch3 and ignore cache for simplicity temporarily.

```
ArchCheck -> LoadGraphDataset -> create streams, shuffler, hash table,random states, queue , graph pool -> ...
```

and a dataset looks like this, `LoadGraphDataset` will load bin to corresponding ctx(maintained by `ctx_map`), in the form of `Tensor`

```
(py38) wangjl@meepo3:~$ ls /graph-learning/samgraph/papers100M_256/
alias_table.bin            cache_by_sample_trace.bin  prob_prefix_table.bin
cache_by_degree.bin        indices.bin                prob_table.bin
cache_by_degree_hop.bin    indptr.bin                 test_set.bin
cache_by_fake_optimal.bin  label.bin                  train_set.bin
cache_by_heuristic.bin     meta.txt                   valid_set.bin
```

> indices.bin indptr.bin, csr graph format, vertex & edge


In `meta.txt`, parameters, such as `NUM_NODE`, are listed.

`GPUShuffler` is a bit like `DataLoader` :sweat_smile:

>  HashTable, eliminate duplication

`TaskQueue` is task queue literally.

`GraphPool`, the queue of `GraphBatch`, `GraphPool` is one of owners of `GraphBatch`, which is a `Task`. 

During training, if pipline isn't configured, call `RunSampleOnce` every mini-batch, else call `Start` at the beginning.

`RunSampleOnce` will call different functions depends on arch. For arch3, `RunSampleSubLoopOnce` and `RunDataCopySubLoopOnce`(if not use gpu cache else `RunCacheDataCopySubLoopOnce`) will be called eventually.

`RunSampleSubLoopOnce` will get a task by call `DoShuffle` and  make a sample(`DoGPUSample`) based on task, then push task to data copy TaskQueue.

`sam.get_next_batch` is simple, return the key of next batch from graph pool. this will set graph batch also.`get_dgl_blocks` will translate graph batch to dgl blocks. 

`Start` will call `GetArch{x}Loops` where x is arch id, then start background threads doing the similar staff. simply overlap sampling and training.

> :confused: `RunConfig::option_sanity_check`

`RunDataCopySubLoopOnce` will get a task from data copy TaskQueue, do copy and submit to `GraphPool`. 

> :confused: in `sample_khop0`, why `__syncwarp`, is it guarantee gpu schedules warps in that way ? where the *magic numbers* come from?

> :confused: mock eatra?

## engine

gpu engine have three cuda stream

- _sample_stream
- _sampler_copy_stream
- _trainer_copy_stream

## gcn

$$
h_i^{(l+1)} = \sigma(b^{(l)} + \sum_{j\in\mathcal{N}(i)}\frac{1}{c_{ji}}h_j^{(l)}W^{(l)})
$$


# UM 


`factor` is the estimation of $\frac{graph\_size}{gpu\_mem\_size}$

- if `factor <= 1`, the graph is stored in gpu
- otherwise, $\frac{(factor-1)}{factor} * graph\_size$ equal to #bytes in cpu, $\frac{1}{factor}*graph\_size$ equal to #bytes in gpu

## paperM100

khop0

|available gpu mem|factor|sampler mem usage|sample time|
|-|-|-|-|
|-|-|-|0.75(not use um, sample by gpu)|
|11727MiB|0.9|28.68 GB|0.77|
|8655MiB|1.38|29.93 GB|1.61|
|5583MiB|2.57|30.81 GB|2.59|
|2511MiB|18|31.56 GB|3.71|
|-|-|-|3.76(um in cpu)|
|-|-|-|14.42(sample by cpu)|


khop0, store graph in UM, advice by degree

|available gpu mem|factor|sampler mem usage|sample time|
|-|-|-|-|
|-|-|-|0.77(not use um, sample by gpu)|
|11727 MB|0.94|28.68 GB|0.76|
|8655 MB|1.38|29.96 GB|0.77|
|5583 MB|2.57|30.83 GB|1.14|
|2511 MB|18.0|31.59 GB|2.75|
|-|-|-|3.37(um in cpu)|
|-|-|-|14.42(sample by cpu)|

khop0, advice by trainset

|available gpu mem|factor|sampler mem usage|sample time|
|-|-|-|-|
|-|-|-|0.76(not use um, sample by gpu)|
|11727 MB|0.94|28.61 GB|0.76|
|8655 MB|1.38|29.89 GB|1.09|
|5583 MB|2.57|30.77 GB|1.38|
|2511 MB|18.0|31.58 GB|2.11|
|-|-|-|2.89(um in cpu)|

khop0, arch0

adivce by presample,14.03
advice by degree 14.37
advice by default 14.48
... random 14.50
... trainset 14.51

## friendster

khop0, default

|available gpu mem|factor|sampler mem usage|sample time|
|-|-|-|-|
|-|-|-|2.02(not use um, sample by gpu)|
|18895 MB|0.97|28.98 GB|2.01|
|14799 MB|1.28|29.98 GB|3.15|
|10703 MB|1.88|30.61 GB|7.08|
|6607 MB|3.56|31.12 GB|13.61|
|2511 MB|32.0|31.73 GB|20.50|
|-|-|-|21.37(um in cpu)|
|-|-|-|112.09(sample by cpu)|

khop0, advice by degree

|available gpu mem|factor|sampler mem usage|sample time|
|-|-|-|-|
|-|-|-|2.01(not use um, sample by gpu)|
|18895 MB|0.97|28.98 GB|2.01|
|14799 MB|1.28|29.98 GB|3.05|
|10703 MB|1.88|30.61 GB|5.29|
|6607 MB|3.56|31.12 GB|10.29|
|2511 MB|32.0|31.73 GB|19.31|
|-|-|-|20.82(um in cpu)|

khop0, advice by trainset

|available gpu mem|factor|sampler mem usage|sample time|
|-|-|-|-|
|-|-|-|2.01(not use um, sample by gpu)|
|18895 MB|0.97|28.98 GB|2.01|
|14799 MB|1.28|29.98 GB|3.69|
|10703 MB|1.88|30.61 GB|8.70|
|6607 MB|3.56|31.12 GB|14.82|
|2511 MB|32.0|31.73 GB|20.49|
|-|-|-|21.29(um in cpu)|