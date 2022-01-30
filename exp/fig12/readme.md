# Figure 12: Impact of policy on extract time

The goal of this experiment is to show how cache ratio affects the time of feature extraction.

This test is similar to fig10, while cache percentage is updated to the max availiable value under 16GB gpu.

`runner.py` runs all necessary tests and redirect logs to directory `run-logs`.
`parser.py` parses results from log files and generate `data.dat`.
`plot.plt` plots corresponding figure to `fig12.eps`.

The results of optimal is produced by an extra test and is calculated by profiling each batch's access(corresponding log file looks like `run-logs/report_optimal_..._optimal_cache_hit.txt`). This is already included in `runner.py`.

## Hardware Requirements

- Paper's configurations: Two 16GB NVIDIA V100 GPUs
- For other hardware configurations, you may need to modify the cache percentage
  -  Modify `L29`~`L41` in `runner.py`.

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

```sh
> cat data.dat
cache_policy	cache_percentage	dataset_short	sample_type	app	pipeline	train_process_time	hit_percent	epoch_time:train_total	epoch_time:copy_time
random	25.0	TW	kKHop2	gcn	False	5.58	24.810	1.6976	3.8787
degree	25.0	TW	kKHop2	gcn	False	2.71	85.460	1.7318	0.9733
presample_1	25.0	TW	kKHop2	gcn	False	2.47	89.430	1.7113	0.7567
random	21.0	PA	kKHop2	gcn	False	9.87	20.940	4.5263	5.3443
degree	21.0	PA	kKHop2	gcn	False	6.37	76.810	4.4623	1.908
presample_1	21.0	PA	kKHop2	gcn	False	5.11	98.520	4.507	0.6048
random	14.0	UK	kKHop2	gcn	False	11.21	13.780	3.574	7.6358
degree	14.0	UK	kKHop2	gcn	False	9.23	35.950	3.4587	5.7692
presample_1	14.0	UK	kKHop2	gcn	False	6.72	69.530	3.6477	3.0724
random	32.0	TW	kKHop2	graphsage	False	2.26	32.000	0.51	1.75
degree	32.0	TW	kKHop2	graphsage	False	0.96	87.000	0.52	0.44
presample_1	32.0	TW	kKHop2	graphsage	False	0.98	89.000	0.52	0.46
random	25.0	PA	kKHop2	graphsage	False	4.38	25.000	1.55	2.83
degree	25.0	PA	kKHop2	graphsage	False	2.35	84.000	1.51	0.84
presample_1	25.0	PA	kKHop2	graphsage	False	1.84	99.000	1.47	0.37
random	18.0	UK	kKHop2	graphsage	False	5.01	17.000	1.31	3.7
degree	18.0	UK	kKHop2	graphsage	False	4.00	44.000	1.37	2.63
presample_1	18.0	UK	kKHop2	graphsage	False	2.64	73.000	1.27	1.37
random	26.0	TW	kRandomWalk	pinsage	False	4.91	26.000	2.85	2.06
degree	26.0	TW	kRandomWalk	pinsage	False	3.43	87.000	2.93	0.5
presample_1	26.0	TW	kRandomWalk	pinsage	False	3.46	86.000	2.91	0.55
random	22.0	PA	kRandomWalk	pinsage	False	9.24	22.000	6.75	2.49
degree	22.0	PA	kRandomWalk	pinsage	False	7.67	79.000	6.83	0.84
presample_1	22.0	PA	kRandomWalk	pinsage	False	7.25	97.000	6.81	0.44
random	13.0	UK	kRandomWalk	pinsage	False	14.16	13.000	7.71	6.45
degree	13.0	UK	kRandomWalk	pinsage	False	12.88	33.000	7.96	4.92
presample_1	13.0	UK	kRandomWalk	pinsage	False	11.28	57.000	7.94	3.34
random	25.0	TW	kWeightedKHopPrefix	gcn	False	3.01	24.600	1.1453	1.8684
degree	25.0	TW	kWeightedKHopPrefix	gcn	False	2.37	53.590	1.1557	1.213
presample_1	25.0	TW	kWeightedKHopPrefix	gcn	False	1.75	81.710	1.1519	0.5987
random	21.0	PA	kWeightedKHopPrefix	gcn	False	6.11	20.950	3.2277	2.8779
degree	21.0	PA	kWeightedKHopPrefix	gcn	False	4.43	74.740	3.3373	1.0932
presample_1	21.0	PA	kWeightedKHopPrefix	gcn	False	3.70	98.210	3.3127	0.3859

```

## FAQ