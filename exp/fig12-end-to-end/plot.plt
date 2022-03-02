# feat_size_MB = 54228.0
# max_cache_rate = 16130/feat_size_MB

outputfname = "fig12-end-to-end.eps"
dat_file='data.dat'

# col numbers
col_cache_policy = 1
col_cache_percent = 2
col_dataset = 3
col_sample_type = 4
col_app = 5
col_pipeline = 6
col_epoch_time = 7
col_hit_percent = 8
col_train_time = 9
col_copy_time = 10

set fit logfile '/dev/null'
set fit quiet
set datafile sep '\t'

set terminal postscript "Helvetica,16" eps enhance color dl 2
set size 0.9,0.5
# set zeroaxis

set style data histogram

set style histogram clustered gap 2
set style fill solid border -2

set pointsize 1
set boxwidth 0.5 relative

set tics font ",14" scale 0.5

set rmargin 1 #0
set lmargin 5 #5
set tmargin 0.5 #0.5
set bmargin 4 #1

set output outputfname

#### magic to filter expected data entry from dat file
format_str="<awk -F'\\t' '{ if($".col_cache_policy." ~ \"%s\"      ) { print }}' ".dat_file." "
cmd_filter_dat_by_policy(policy)=sprintf(format_str, policy)
##########################################################################################

### Key
set key inside right Right top enhanced nobox 
set key samplen 1.5 spacing 1.5 height 0.2 width 0 font ',13' maxrows 1 #at graph 0.02, graph 0.975  noopaque


set arrow from 0-0.3,graph -0.15 to  2+0.3,graph -0.15 nohead lt 2 lw 1 lc "#000000" front
set arrow from 3-0.3,graph -0.15 to  5+0.3,graph -0.15 nohead lt 2 lw 1 lc "#000000" front
set arrow from 6-0.3,graph -0.15 to  8+0.3,graph -0.15 nohead lt 2 lw 1 lc "#000000" front
set arrow from 9-0.3,graph -0.15 to 10+0.3,graph -0.15 nohead lt 2 lw 1 lc "#000000" front

set label "GCN"       center at   1, graph -0.22 font ",14" tc rgb "#000000" front
set label "GraphSAGE" center at   4, graph -0.22 font ",14" tc rgb "#000000" front
set label "PinSAGE"   center at   7, graph -0.22 font ",14" tc rgb "#000000" front
set label "GCN (W.)"  center at 9.5, graph -0.22 font ",14" tc rgb "#000000" front

#set arrow from  graph 0, first 8 to graph 1, first 8 nohead lt 1 lw 2 lc "#000000" front

### X-axis
set xrange [-0.5:10.5]
set xtics nomirror offset 0,0.3

## Y-axis
set yrange [0:4]
set ytics 0,1,4
set ylabel "Runtime (sec)" offset 1.,0
set ytics offset 0.5,0 #format "%.1f"


plot cmd_filter_dat_by_policy('random')      using col_epoch_time:xticlabels(col_dataset) lc "#ff9900" title 'Random' \
    ,cmd_filter_dat_by_policy('degree')      using col_epoch_time:xticlabels(col_dataset) lc "#c00000" title 'Degree' \
    ,cmd_filter_dat_by_policy('presample_1') using col_epoch_time:xticlabels(col_dataset) lc "#000000" title 'PreSC#1' \
