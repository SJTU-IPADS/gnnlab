# Figure 4a: Impact of cache ratio

The goal of this experiment is to show how cache ratio affects feature extraction(i.e. cache hit rate, extraction time).
This proves that by enabling GPU-based sampling, the reduced cache ratio leads to significant slowdown in feature extraction.

`runner.py` runs all necessary tests and redirect logs to directory `run-logs`.
`parser.py` parses results from log files and generate `data.dat`.
`plot.plt` plots corresponding figure to `fig4a.eps`.

## Hardware Requirements

- Paper's configurations: Two 16GB NVIDIA V100 GPUs
- For other hardware configurations, you may need to modify the cache percentage
  -  Modify `L29` in `runner.py`. `percent_gen(0, 30, 1)` means run test from cache ratio 0% to 30% with step=1%.

## Run Command

```sh
> python runner.py
> python parser.py
> gnuplot plot.plt
```

There are serveral command line arguments for `runner.py`:

- `-m`, `--mock`: Show the run command for each test case but not actually run it
- `-i`, `--interactive`: run these tests with output printed to terminal, rather than redirec to log directory.

The number of epochs to run is set to 3 for fast reproduce. You may change line containing `.override('epoch', [3])` to change the numer of epochs.


## Output Example

`python runner.py` will redirect all logs to `run-logs` directory.
```sh
> cat data.dat
cache_policy	cache_percentage	dataset_short	sample_type	app	hit_percent	batch_copy_time	batch_train_time
degree	0.0	PA	kKHop2	gcn	0.000	0.0411	0.0279
degree	5.0	PA	kKHop2	gcn	28.920	0.0309	0.0297
degree	10.0	PA	kKHop2	gcn	49.570	0.0229	0.0295
degree	15.0	PA	kKHop2	gcn	63.960	0.0171	0.0293
degree	20.0	PA	kKHop2	gcn	75.120	0.0126	0.0293
degree	25.0	PA	kKHop2	gcn	82.890	0.0102	0.0299
degree	30.0	PA	kKHop2	gcn	88.410	0.0075	0.0301
```

## FAQ