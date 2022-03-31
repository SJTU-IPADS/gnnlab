outputfname = "fig5a.eps"
dat_file='data.dat'

# col numbers
col_cache_policy=1
col_cache_percent=2
col_dataset=3
col_sample_type=4
col_app=5
col_hit_percent=6
col_optimal_hit_percent=7
col_miss_nbytes=8
col_feat_nbytes=9

# col_sample_second=4
# col_copy_second=5
# col_train_second=6

set fit logfile '/dev/null'
set fit quiet
set datafile sep '\t'

### Key
set key inside right Right top enhanced nobox 
#set key samplen 1 spacing 1.2 height 0.2 font ',13' at 28, 285 noopaque
set key samplen 1 spacing 1.2 height 0.2 font ',13' at graph 0.95, graph 0.95 noopaque
#maxrows 3 width 0.5 height 0.5 

set terminal postscript "Helvetica,16" eps enhance color dl 2
set pointsize 1
set size 0.4,0.4
set zeroaxis

set tics font ",14" scale 0.5

set rmargin 1.5
set lmargin 5.5
set tmargin 1.5
set bmargin 2.5

set output outputfname

### X-axis
set xlabel "Cache Ratio (%)" offset 0,0.7
set xrange [0:30]
set xtics 0, 5, 30  offset -0.2,0.3
set xtics nomirror

### Y-axis
set ylabel "Transfer Size (MB)" offset 1.7,0
set yrange [0:300]
set ytics 0,60,300 offset 0.5,0 #format "%.1f"
set ytics nomirror

# GPU sampling, max cache 7%, which is 53G * 0.07 = 3.7G
set arrow   1 from 7, 0 to 7, graph 1 nohead lt 2 lw 3 lc rgb "#000000" back
set label   2 "3.7GB"  center at 7,  graph 1.08 font ",14" tc rgb "#000000"  front

set label   3 "GPU-based" left at    5, graph 0.86 font ",12" tc rgb "#000000"  front
set label   4 "Sampling"  left at    5, graph 0.77 font ",12" tc rgb "#000000"  front
set object  5 rect from 6, graph 0.72 to 8, graph 0.92 \
    fc 'white' fs solid 1.0 noborder front

# CPU sampling, max cache 21%, which is 53G * 0.21 = 11.1G
set arrow  11 from 21, 0 to 21, graph 1 nohead lt 2 lw 3 lc rgb "#000000" back
set label  12 "11.1GB" center at 21, graph 1.08 font ",14" tc rgb "#000000"  front

set label  13 "CPU-based" left at 18.5, graph 0.51 font ",12" tc rgb "#000000" front 
set label  14 "Sampling"  left at 18.5, graph 0.42 font ",12" tc rgb "#000000" front 
set object 15 rect from 20, graph 0.37 to 22, graph 0.57 \
    fc 'white' fs solid 1.0 noborder front

# white box for key
#set arrow  21 from  6.8, graph 0.06 to  6.8, graph 0.29 nohead lt 2 lw 10 lc rgb "white" noborder back
set arrow  21 from  21, graph 0.69 to  21, graph 0.94 nohead lt 2 lw 10 lc rgb "white" noborder back
# set arrow  22 from 20.6, 6 to 20.6, 18 nohead lt 2 lw 10 lc rgb "white" noborder back

NonZero(t)=(t == 0 ? NaN : t)
to_copy_MB(hit_percent, feat_nbytes)=(100-hit_percent)/100*feat_nbytes/1024/1024

plot dat_file    using 2:(column(col_miss_nbytes)/1024/1024)                w l lw 3 lc "#c00000" title "Degree" \
    ,dat_file    using 2:(to_copy_MB(column(col_optimal_hit_percent),column(col_feat_nbytes)))  w l lw 3 lc "#0000ee" title 'Optimal'  \
