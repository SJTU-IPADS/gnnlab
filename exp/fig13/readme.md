# Figure 13: Impact of policy on end-to-end time

The goal of this experiment is to show how cache ratio affects the end-to-end time.

This test is similar to table4 with same cache rate.

`runner.py` runs all necessary tests and redirect logs to directory `run-logs`.
`parser.py` parses results from log files and generate `data.dat`.
`plot.plt` plots corresponding figure to `fig13.eps`.

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
- `-i`, `--interactive`: run these tests with output printed to terminal, rather than redirect to log directory.

The number of epochs to run is set to 3 for fast reproduce. You may change line containing `.override('epoch', [3])` to change the numer of epochs.


## Output Example

```sh
> cat data.dat
cache_policy	cache_percentage	dataset_short	sample_type	app	pipeline	train_process_time	hit_percent	epoch_time:train_total	epoch_time:copy_time
presample_1	22.0	TW	kKHop2	gcn	True	0.48	87.330	0.3996	0.4061
presample_1	20.0	PA	kKHop2	gcn	True	0.85	98.530	0.7862	0.3441
presample_1	11.0	UK	kKHop2	gcn	True	1.59	63.380	0.8447	1.5472
presample_1	31.0	TW	kKHop2	graphsage	True	0.20	88.000	0.14	0.18
presample_1	24.0	PA	kKHop2	graphsage	True	0.30	99.000	0.25	0.15
presample_1	16.0	UK	kKHop2	graphsage	True	0.65	69.000	0.46	0.62
presample_1	25.0	TW	kRandomWalk	pinsage	True	0.65	86.000	0.59	0.27
presample_1	21.0	PA	kRandomWalk	pinsage	True	1.07	97.000	1.0	0.22
presample_1	9.0	UK	kRandomWalk	pinsage	True	1.97	48.000	1.45	1.78
presample_1	24.0	TW	kWeightedKHopPrefix	gcn	True	0.32	81.520	0.2502	0.2692
presample_1	21.0	PA	kWeightedKHopPrefix	gcn	True	0.58	98.200	0.5301	0.1949
degree	22.0	TW	kKHop2	gcn	True	0.62	83.980	0.4901	0.4752
degree	20.0	PA	kKHop2	gcn	True	0.99	75.150	0.8959	0.8957
degree	11.0	UK	kKHop2	gcn	True	2.71	30.070	1.3676	2.6305
degree	31.0	TW	kKHop2	graphsage	True	0.21	87.000	0.12	0.19
degree	24.0	PA	kKHop2	graphsage	True	0.39	83.000	0.33	0.36
degree	16.0	UK	kKHop2	graphsage	True	1.19	41.000	0.5	1.17
degree	25.0	TW	kRandomWalk	pinsage	True	0.65	86.000	0.6	0.25
degree	21.0	PA	kRandomWalk	pinsage	True	1.10	77.000	1.01	0.42
degree	9.0	UK	kRandomWalk	pinsage	True	2.52	26.000	1.82	2.32
degree	24.0	TW	kWeightedKHopPrefix	gcn	True	0.58	53.390	0.3244	0.5426
degree	21.0	PA	kWeightedKHopPrefix	gcn	True	0.63	74.700	0.5738	0.4774
random	21.0	TW	kKHop2	gcn	True	1.85	20.860	0.8306	1.7778
random	18.0	PA	kKHop2	gcn	True	2.22	17.940	1.4008	2.1809
random	10.0	UK	kKHop2	gcn	True	3.40	9.840	1.1519	3.3289
random	29.0	TW	kKHop2	graphsage	True	0.84	29.000	0.44	0.76
random	24.0	PA	kKHop2	graphsage	True	1.13	24.000	0.85	1.11
random	15.0	UK	kKHop2	graphsage	True	1.59	15.000	0.59	1.57
random	20.0	TW	kRandomWalk	pinsage	True	1.08	20.000	0.68	0.99
random	20.0	PA	kRandomWalk	pinsage	True	1.21	20.000	1.11	1.11
random	7.0	UK	kRandomWalk	pinsage	True	2.98	7.000	1.94	2.84
random	24.0	TW	kWeightedKHopPrefix	gcn	True	0.83	23.600	0.3614	0.8079
random	21.0	PA	kWeightedKHopPrefix	gcn	True	1.17	20.940	0.7532	1.1441
```

## FAQ