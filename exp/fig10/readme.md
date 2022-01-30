# Figure 10: Hit rate of presample under various applicaion

The goal of this experiment is to show presample's robustness under various application with fixed cache ratio=10%.

`runner.py` runs all necessary tests and redirect logs to directory `run-logs`.
`parser.py` parses results from log files and generate `data.dat`.
`plot.plt` plots corresponding figure to `fig10.eps`.

The results of optimal is produced by an extra test and is calculated by profiling each batch's access(corresponding log file looks like `run-logs/report_optimal_..._optimal_cache_hit.txt`). This is already included in `runner.py`.

## Hardware Requirements

- Paper's configurations: Two 16GB NVIDIA V100 GPUs. However, UK-2006-05 with weighted sampling requires more than 16GB memory, so this configuration is separatedly executed on another machine with less but larger GPUs. Since only hit rate is reported, the results should be comparable.
- For other hardware configurations, you may need to modify the cache percentage
  -  Modify `L41` in `runner.py`. `percent_gen(10, 10, 1)` means run test at cache ratio 10%.

## Run Command

```sh
> python runner.py
> python parser.py
> gnuplot plot.plt
```

There are serveral command line arguments for `runner.py`:

- `-m`, `--mock`: Show the run command for each test case but not actually run it
- `-i`, `--interactive`: run these tests with output printed to terminal, rather than redirec to log directory.

## Output Example

`python runner.py` will redirect all logs to `run-logs` directory.
```sh
> cat data.dat
cache_policy	cache_percentage	dataset_short	sample_type	app	hit_percent	optimal_hit_percent
random	10.0	PR	kKHop2	gcn	10.010	20.669
random	10.0	TW	kKHop2	gcn	9.930	79.827
random	10.0	PA	kKHop2	gcn	9.960	100.000
random	10.0	UK	kKHop2	gcn	9.840	65.230
random	10.0	PR	kRandomWalk	pinsage	10.000	31.517
random	10.0	TW	kRandomWalk	pinsage	10.000	84.368
random	10.0	PA	kRandomWalk	pinsage	10.000	100.000
random	10.0	UK	kRandomWalk	pinsage	10.000	58.963
random	10.0	PR	kWeightedKHopPrefix	gcn	10.010	24.947
random	10.0	TW	kWeightedKHopPrefix	gcn	9.870	75.405
random	10.0	PA	kWeightedKHopPrefix	gcn	9.950	100.000
random	10.0	UK	kWeightedKHopPrefix	gcn	9.790	66.866
degree	10.0	PR	kKHop2	gcn	19.120	20.669
degree	10.0	TW	kKHop2	gcn	72.720	79.827
degree	10.0	PA	kKHop2	gcn	49.570	100.000
degree	10.0	UK	kKHop2	gcn	28.010	65.230
degree	10.0	PR	kRandomWalk	pinsage	26.000	31.517
degree	10.0	TW	kRandomWalk	pinsage	75.000	84.368
degree	10.0	PA	kRandomWalk	pinsage	52.000	100.000
degree	10.0	UK	kRandomWalk	pinsage	28.000	58.963
degree	10.0	PR	kWeightedKHopPrefix	gcn	21.260	24.947
degree	10.0	TW	kWeightedKHopPrefix	gcn	48.400	75.405
degree	10.0	PA	kWeightedKHopPrefix	gcn	52.170	100.000
degree	10.0	UK	kWeightedKHopPrefix	gcn	32.020	66.866
presample_1	10.0	PR	kKHop2	gcn	20.370	20.669
presample_1	10.0	TW	kKHop2	gcn	74.870	79.827
presample_1	10.0	PA	kKHop2	gcn	98.290	100.000
presample_1	10.0	UK	kKHop2	gcn	61.290	65.230
presample_1	10.0	PR	kRandomWalk	pinsage	30.000	31.517
presample_1	10.0	TW	kRandomWalk	pinsage	75.000	84.368
presample_1	10.0	PA	kRandomWalk	pinsage	97.000	100.000
presample_1	10.0	UK	kRandomWalk	pinsage	50.000	58.963
presample_1	10.0	PR	kWeightedKHopPrefix	gcn	24.370	24.947
presample_1	10.0	TW	kWeightedKHopPrefix	gcn	64.020	75.405
presample_1	10.0	PA	kWeightedKHopPrefix	gcn	97.940	100.000
presample_1	10.0	UK	kWeightedKHopPrefix	gcn	60.810	66.866
```

## FAQ