# Optimal extraction for multi gpu with different frequency

## workflow

1. run `runner.py` to generate statistics binary files for each test case.
  - `xxxx_freq_bin.bin`: a list of double. The average frequency(in an epoch) for each node.
2. run `solver.py` to calculate ideal performance improvement of each case.
  - `gen_density_matrix_fine_grain` is called to 
    - split nodes into blocks using `exp` method.
      - the method of split:
        - `num`: deprecated
          - for each device, slice nodes into 100 slots by the sorting of frequency.
          - so each node have a "percentage rank" for each device.
          - group nodes into blocks: nodes having same slot rank for each device is grouped together
        - `exp`: For reducing time complexity. This is similar to num, except the slicing method is complex:
          - slice frequency into S slots according to the top 1% frequency F:
            - [F, +inf), [F/a, F), [F/(a^2), F/a), [F/(a^3), F/(a^2)), ...
        - `bin`: For calculating naive solution.
          - limit to 2 slot, use a `percent` parameter as the slice point.
    - for each block, calculate:
      - size of each block
      - average frequency of each block
      note that we have S^(num of device) blocks.
    - output these data to disk
      - `density.bin`
      - `block_freq.bin`
  - call `solver_gurobi` to build a Linear Programming problem and solve it
    - the problem is built by:
      - assign each block in pair with each device a variable, indicating whether the block is stored in the device.
      - constraint each gpu's cache size
      - goal is to min(max(time of each gpu))
    - solve the problem using gurobi, a third-party solver.
    - once solved, `solver_gurobi` outputs the ideal time of each gpu.
      - the detailed result of each variable is currently not stored.
  - calculate the time a naive method:
    - by using the `bin` with expected cache rate`
  - compare the improvement in time.

## Example outputs:

- `part8.out` and `part4.out`: example output for 4 or 8 devices under different hardware configuration.
- `trend-part4.out` and `trend-part4.out.small-range`: example output for 4 device with fine-grained changes in hardware configuration:
  - cache size
  - nvlink's speed

## Other files and their story

- `gen_density_matrix`: deprecated
  - The result frequency is not calculated per block. On the contrary, the frequency is average per slot per device. This may leads to inaccuracy when the slice is too discrete.
- `solver_fine_grain` and `solver_continious`: deprecated.
  - The initial idea is to put entire block into block or not, so the variable is binary, and the linear programming prblem is a Mixed Integer Programming problem. However, MIP is too slow, so the problem is relaxed into a continious form: each block is allowed to be cached at a rate between 0 and 1. This is the transition from `solver_fine_grain` to `solver_continious`. Now these script is merged into one script `solver` with an option.
- `gen_lp` and `gen_lp_continious`:
  - Originally, the LP problem is built in python using Pulp package, which generates an `.lp` text file describing the problem, and solved using `gurobi` command-line tool. The python part is too slow. Since the struct of `.lp` file is simple, so I wrote an cpp script to directly generate the `.lp` file. Now the generation is fast, however, reading the `.lp` file is also too slow. The final solution is to directly use `gurobi`'s cpp interface to build and solve the problem, which is described in the first chapter.