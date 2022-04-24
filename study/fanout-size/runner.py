import re, os, subprocess

def parse_output(output):
  out_list = []
  for line in output.split('\n'):
    m = re.match(r'On hop ([0-9]*):.*\(([0-9\.]*)%\).*', line)
    if not m:
      continue
    # print(m.group(1), m.group(2))
    out_list.append(float(m.group(2)))
  return out_list

def run(dataset, percent):
  cmd=f'../../utility/data-process/build/train-graph-size -g {dataset} --percent {percent} --max-hop 20' 
  return subprocess.check_output(cmd, shell=True).decode('utf-8')

os.system('mkdir -p plot')
for percent in [0.99,0.3,0.1,0.03,0.01,0.003]:
  with open(f"plot/percent-{percent}.dat", 'w') as f:
    for dataset in ['products','twitter','uk-2006-05']:
      output = run(dataset, percent)
      output = parse_output(output)
      for hop in range(len(output)):
        print(f"{dataset},{hop},{output[hop]}",file=f)
  