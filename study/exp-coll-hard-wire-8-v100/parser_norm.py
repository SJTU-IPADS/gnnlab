import pandas

df = pandas.read_csv('data.dat', sep='\t')
# df = df.pivot_table(["train_process_time", "Step(average) L1 train total"], columns="cache_policy_short", index=["sample_type_short", "unsupervised", "batch_size", "dataset_short", "cache_percentage"])
# df["CliqDeg_2"] = df["train_process_time", "CliqDeg_2"] / df["train_process_time", "Coll_2"]
# df["Cliq_2"]    = df["train_process_time", "Cliq_2"]    / df["train_process_time", "Coll_2"]
# df["Rep_2"]     = df["train_process_time", "Rep_2"]     / df["train_process_time", "Coll_2"]
# df["Coll_2"] = 1
df = df.pivot_table("train_process_time", columns="cache_policy_short", index=["sample_type_short", "unsupervised", "batch_size", "dataset_short", "cache_percentage"])
df["CliqDeg_2"] = df["CliqDeg_2"] / df["Coll_2"]
df["Cliq_2"]    = df["Cliq_2"]    / df["Coll_2"]
df["Rep_2"]     = df["Rep_2"]     / df["Coll_2"]
df["Coll_2"] = 1
df.sort_index()

with open(f'pv.dat', 'w') as f:
  print(df.to_csv(
    # columns=[
    #   'dataset', 'sample_type', 'cache_policy', 'cache_percent', 'z',
    #   'optimal_local_rate', 'optimal_remote_rate', 'optimal_cpu_rate',
    #   'unsup', 'gpu', 'num_trainer', 'norm_z'
    # ], 
    sep='\t', index=True), file=f)