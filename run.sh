log_path=./run_logs/
dataset=com-friendster
# dataset=reddit

log_file=${log_path}gcn_${dataset}_`date +%m-%d#%H-%M-%S`.log

cmd="python example/samgraph/train_gcn.py \
    --empty-feat 4 -ll debug \
    --sample-type khop0 \
    --dataset  ${dataset} \
    --override-device \
    --override-train-device cuda:0 \
    --override-sample-device cuda:1"
    # --unified-memory
    

if [ $# -ge 1 ] && [ $1 = "-log" ]; then
    echo $cmd > $log_file
    cmd="$cmd >>${log_file} 2>&1"
fi

echo $cmd
eval $cmd
    