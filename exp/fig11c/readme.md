# Figure 11c: Impact of feature dimension

The goal of this experiment is to show that, under fixed cache space, how the dimension of feature affects feature extraction(i.e. cache hit rate, extraction time).
This test is similar to fig4b.

`runner.py` runs all necessary tests and redirect logs to directory `run-logs`.
`parser.py` parses results from log files and generate `data.dat`.
`plot.plt` plots corresponding figure to `fig11c.eps`.

This test is an simulation. Detailed explaination please refer to `fig4b/readme.md`.

## Hardware Requirements

- Paper's configurations: Two 16GB NVIDIA V100 GPUs
- For other hardware configurations, you may need to modify the cache percentage
  -  Modify `L30` in `runner.py`. `percent_gen(0, 30, 1)` means run test from cache ratio 0% to 30% with step=1%.

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
cache_policy	cache_percentage	dataset_short	sample_type	app	hit_percent	optimal_hit_percent	batch_feat_nbytes	batch_miss_nbytes	dim	new_batch_feat_GB	new_batch_miss_GB
degree	0.0	PA	kKHop2	gcn	0.0	0.0	290930409.9603	290930409.9603	inf	inf	inf
degree	5.0	PA	kKHop2	gcn	28.93	97.167	290985073.5894	206814767.4702	241.7053920483883	0.5117374624544394	0.36369181456637
degree	10.0	PA	kKHop2	gcn	49.57	100.0	290927220.9801	146712786.2252	120.85269602419415	0.2558178603232342	0.129008946961007
degree	15.0	PA	kKHop2	gcn	63.96	100.0	290982869.6159	104873343.1523	80.56846401612943	0.1705778621541592	0.06147626152035898
degree	20.0	PA	kKHop2	gcn	75.11	100.0	290924911.894	72396701.6689	60.42634801209707	0.12790791494986858	0.03183628003102229
degree	25.0	PA	kKHop2	gcn	82.89	100.0	290943391.3642	49780578.3311	48.341078409677664	0.10233283170029119	0.017509147503919822
degree	30.0	PA	kKHop2	gcn	88.4	100.0	290979363.6026	33745292.7152	40.28423200806471	0.08528790344226697	0.009893396799302963
random	0.0	PA	kKHop2	gcn	0.0	0.0	290944605.245	290944605.245	inf	inf	inf
random	5.0	PA	kKHop2	gcn	5.0	97.167	290914534.5695	276381621.404	241.7053920483883	0.511613409840323	0.4860327393483068
random	10.0	PA	kKHop2	gcn	9.96	100.0	290937152.4238	261963075.8146	120.85269602419415	0.25582659323130896	0.23034626454547055
random	15.0	PA	kKHop2	gcn	14.94	100.0	290978461.6689	247513172.7682	80.56846401612943	0.1705752781595181	0.14509133160248608
random	20.0	PA	kKHop2	gcn	19.96	100.0	290959975.4172	232871151.0464	60.42634801209707	0.127923330962592	0.10238983410245862
random	25.0	PA	kKHop2	gcn	24.96	100.0	290938181.5099	218333952.0	48.341078409677664	0.10233099925054635	0.07678918183760998
random	30.0	PA	kKHop2	gcn	29.96	100.0	290972627.9205	203786412.9272	40.28423200806471	0.08528592917097347	0.05973426479134981
presample_1	0.0	PA	kKHop2	gcn	0.0	0.0	290986911.3642	290986911.3642	inf	inf	inf
presample_1	5.0	PA	kKHop2	gcn	96.02	97.167	290995223.7351	11595759.0464	241.7053920483883	0.5117553128882777	0.02036786145295347
presample_1	10.0	PA	kKHop2	gcn	98.29	100.0	291006423.3113	4969217.6954	120.85269602419415	0.25588750444533437	0.004375676326015202
presample_1	15.0	PA	kKHop2	gcn	98.51	100.0	290970951.2053	4336706.1192	80.56846401612943	0.1705708754301534	0.0025415060439092766
presample_1	20.0	PA	kKHop2	gcn	98.52	100.0	290958508.9272	4300801.6954	60.42634801209707	0.12792268620626304	0.0018932557558526983
presample_1	25.0	PA	kKHop2	gcn	98.53	100.0	290927590.5695	4280255.5762	48.341078409677664	0.10232727412410023	0.0015042109296242723
presample_1	30.0	PA	kKHop2	gcn	98.7	100.0	290926931.0728	3787803.1258	40.28423200806471	0.08527253513406995	0.0011085429567429068
```

## FAQ