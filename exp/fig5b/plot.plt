outputfname = "fig5b.eps"
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
set xrange [0:35]
set xtics 0, 5, 35  offset -0.2,0.3
set xtics nomirror

### Y-axis
# set ylabel "Cache Hit Rate (%)" offset 2,0
# set yrange [0:100]
# set ytics 0,20,100 offset 0.5,0 #format "%.1f"
# set ytics nomirror
set ylabel "Transfer Size (MB)" offset 1.7,0
set yrange [0:400]
set ytics 0,100,400 offset 0.5,0 #format "%.1f"
set ytics nomirror

# GPU sampling, max cache 1%, which is 40G * 0.01 = 0.4G
set arrow   1 from 1, 0 to 1, graph 1 nohead lt 2 lw 3 lc rgb "#000000" back
set label   2 "0.4GB"  center at 4,  graph 1.08 font ",14" tc rgb "#000000"  front

set label   3 "GPU-based" right at    10.7, graph 0.85 font ",12" tc rgb "#000000"  front
set label   4 "Sampling"  right at    10.7, graph 0.76 font ",12" tc rgb "#000000"  front
set object  5 rect from 0.5, graph 0.79 to 1.5, graph 0.91 \
    fc 'white' fs solid 1.0 noborder front
# set label   3 "GPU-based" left at    1, graph 0.20 font ",12" tc rgb "#000000"  front
# set label   4 "Sampling"  left at    1, graph 0.11 font ",12" tc rgb "#000000"  front
# set object  5 rect from 0.8, graph 0.06 to 10, graph 0.25 \
#     fc 'white' fs solid 1.0 noborder front

# CPU sampling, max cache 30%, which is 40G * 0.30 = 12.2G
set arrow  11 from 30, 0 to 30, graph 1 nohead lt 2 lw 3 lc rgb "#000000" back
set label  12 "12.2GB" center at 30, graph 1.08 font ",14" tc rgb "#000000"  front

# set label  13 "CPU-based" left at 23, graph 0.61 font ",12" tc rgb "#000000" front 
# set label  14 "Sampling"  left at 23, graph 0.52 font ",12" tc rgb "#000000" front 
# set object 15 rect from 29.7, graph 0.47 to 31.7, graph 0.67 \
#     fc 'white' fs solid 1.0 noborder front
set label  13 "CPU-based" right at 33, graph 0.56 font ",12" tc rgb "#000000" front 
set label  14 "Sampling"  right at 33, graph 0.47 font ",12" tc rgb "#000000" front 
set object 15 rect from 29, graph 0.42 to 31, graph 0.62 \
    fc 'white' fs solid 1.0 noborder front

# white box for key
#set arrow  21 from  6.8, graph 0.06 to  6.8, graph 0.29 nohead lt 2 lw 10 lc rgb "white" noborder back
set arrow  21 from  30, graph 0.72 to  30, graph 0.97 nohead lt 2 lw 10 lc rgb "white" noborder back
# set arrow  22 from 20.6, 6 to 20.6, 18 nohead lt 2 lw 10 lc rgb "white" noborder back
NonZero(t)=(t == 0 ? NaN : t)
to_copy_MB(hit_percent, feat_nbytes)=(100-hit_percent)/100*feat_nbytes/1024/1024

plot dat_file    using 2:(column(col_miss_nbytes)/1024/1024)                w l lw 3 lc "#c00000" title "Degree" \
    ,dat_file    using 2:(to_copy_MB(column(col_optimal_hit_percent),column(col_feat_nbytes)))  w l lw 3 lc "#0000ee" title 'Optimal'  \
