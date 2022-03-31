# Figure 11b: Case study of presample

The goal of this experiment is to show that the performance of presample is close to optimal under another case.
This test is similar to figure 5a, while adds extra tests of presample and random-based policy.

`runner.py` runs all necessary tests and redirect logs to directory `run-logs`.
`parser.py` parses results from log files and generate `data.dat`.
`plot.plt` plots corresponding figure to `fig11b.eps`.

The results of optimal is produced by an extra test and is calculated by profiling each batch's access(corresponding log file looks like `run-logs/report_optimal_..._optimal_cache_hit.txt`). This is already included in `runner.py`.

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

`python runner.py` will redirect all logs to `run-logs` directory. An short example of the `data.dat` looks like this:
```sh
> cat data.dat
cache_policy	cache_percentage	dataset_short	sample_type	app	hit_percent	optimal_hit_percent
random	0.0	PA	kKHop2	gcn	0.000	0.000
random	5.0	PA	kKHop2	gcn	5.000	97.164
random	10.0	PA	kKHop2	gcn	9.960	100.000
random	15.0	PA	kKHop2	gcn	14.940	100.000
random	20.0	PA	kKHop2	gcn	19.960	100.000
random	25.0	PA	kKHop2	gcn	24.960	100.000
random	30.0	PA	kKHop2	gcn	29.960	100.000
degree	0.0	PA	kKHop2	gcn	0.000	0.000
degree	5.0	PA	kKHop2	gcn	28.930	97.164
degree	10.0	PA	kKHop2	gcn	49.570	100.000
degree	15.0	PA	kKHop2	gcn	63.960	100.000
degree	20.0	PA	kKHop2	gcn	75.110	100.000
degree	25.0	PA	kKHop2	gcn	82.900	100.000
degree	30.0	PA	kKHop2	gcn	88.400	100.000
presample_1	0.0	PA	kKHop2	gcn	0.000	0.000
presample_1	5.0	PA	kKHop2	gcn	96.020	97.164
presample_1	10.0	PA	kKHop2	gcn	98.290	100.000
presample_1	15.0	PA	kKHop2	gcn	98.510	100.000
presample_1	20.0	PA	kKHop2	gcn	98.520	100.000
presample_1	25.0	PA	kKHop2	gcn	98.530	100.000
presample_1	30.0	PA	kKHop2	gcn	98.690	100.000

```

## FAQ