# Figure 4b: Impact of feature dimension

The goal of this experiment is to show that, under fixed cache space, how the dimension of feature affects feature extraction(i.e. cache hit rate, extraction time).

`runner.py` runs all necessary tests and redirect logs to directory `run-logs`.
`parser.py` parses results from log files and generate `data.dat`.
`plot.plt` plots corresponding figure to `fig4b.eps`.

This test is an simulation.
By rerunning tests in figure 4a, we have the relationship between cache ratio and hit rate.
Now we can calculate that, given a 5GB(5120MB) cache, if we could only cache X% features of papers100M dataset, what is the dimension of it.
Since the original dimension is 128 and the original feature is 54228MB, we have this equation:

$$
\frac{54228}{128} * new_dimension * \frac{X}{100} = 5120

new_dimension = 5120 * \frac{128}{54228} * \frac{100}{X}
$$

And the transfer size in a batch is 
$$
new_miss_size = \frac{100 - hit_percent}{100} * original_miss_size * \frac{new_dimension}{128}
$$

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
cache_policy	cache_percentage	dataset_short	sample_type	app	hit_percent	batch_feat_nbytes	batch_miss_nbytes	dim	new_copy_GB
degree	0.0	PA	kKHop2	gcn	0.0	290906671.4702	290906671.4702	inf	inf
degree	5.0	PA	kKHop2	gcn	28.93	290975260.8212	206806790.7815	241.7053920483883	0.36367954993908674
degree	10.0	PA	kKHop2	gcn	49.57	290975589.7219	146735831.3113	120.85269602419415	0.12903039562581198
degree	15.0	PA	kKHop2	gcn	63.96	290926724.2384	104853159.8411	80.56846401612943	0.061464399626513115
degree	20.0	PA	kKHop2	gcn	75.12	290952203.8675	72392237.7748	60.42634801209707	0.03182647463469112
degree	25.0	PA	kKHop2	gcn	82.9	290943272.6887	49754539.2318	48.341078409677664	0.017498907082961285
degree	30.0	PA	kKHop2	gcn	88.4	290989422.1987	33745753.8543	40.28423200806471	0.009893738794973135

```

## FAQ