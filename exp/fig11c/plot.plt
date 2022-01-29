outputfname = "fig11c.eps"
dat_file='data.dat'

# col numbers
col_cache_policy=1
col_cache_percent=2
col_dataset=3
col_hit_percent=6
col_optimal_hit_percent=7
col_dim=10
col_feat_GB=11
col_miss_GB=12

# cache_policy=1
# cache_percentage=2
# dataset_short=3
# sample_type=4
# app=5
# hit_percent=6
# optimal_hit_percent=7
# batch_feat_nbytes=8
# batch_miss_nbytes=9
# dim=10
# new_batch_feat_GB=11
# new_batch_miss_GB=12

set fit logfile '/dev/null'
set fit quiet
set datafile sep '\t'

set terminal postscript "Helvetica,16" eps enhance color dl 2
set pointsize 1
set size 0.4,0.5
set zeroaxis

set tics font ",14" scale 0.5

set rmargin 2
set lmargin 5.5
set tmargin 1.5
set bmargin 2.5

set output outputfname

#### magic to filter expected data entry from dat file
format_str="<awk -F'\\t' '{ if($".col_cache_policy." == \"%s\"      ) { print }}' ".dat_file." "
cmd_filter_dat_by_policy(policy)=sprintf(format_str, policy)
##########################################################################################

### Key
set key inside left Left reverse invert top enhanced nobox 
set key samplen 1 spacing 1.5 height 0.2 font ',13' #at graph 0.03, graph 0.98  noopaque


### X-axis
set xrange [50:950]
set xtics 100,200,900
set xlabel "Feature Dim" offset 0,0.5
set xtics nomirror offset -0.2,0.3

## Y-axis
set yrange [0:2]
set ytics 0.5
set ylabel "Transfer Size (GB)" offset 2,0
set ytics nomirror offset 0.5,0 #format "%.1f"

plot cmd_filter_dat_by_policy("degree")      u col_dim:((100-column(col_optimal_hit_percent))*column(col_feat_GB)/100)          w l lw 10 lc "#00ffff" title 'Optimal' \
    ,cmd_filter_dat_by_policy("presample_1") u col_dim:(column(col_miss_GB))   w l lw 3 lc "#000000" title "PreSC#1" \
    ,cmd_filter_dat_by_policy("degree")      u col_dim:(column(col_miss_GB))   w l lw 3 lc "#c00000" title "Degree" \
    ,cmd_filter_dat_by_policy("degree")      u col_dim:((100-column(col_cache_percent))*column(col_feat_GB)/100)   w l lw 3 lc "#ff9900" title "Random" \
