outputfname = "fig4a.eps"
dat_file='data.dat'

# col numbers
col_cache_policy=1
col_cache_percent=2
col_dataset=3
col_sample_type=4
col_app=5
col_hit_percent=6
col_copy_second=7
col_train_second=8

set datafile sep '\t'

set terminal postscript "Helvetica,16" eps enhance color dl 2
set pointsize 1
set size 0.4,0.4
set zeroaxis

set tics font ",14" scale 0.5

set rmargin 5.5
set lmargin 5.5
set tmargin 1.5
set bmargin 2.5

set output outputfname

### Key
set key inside left Left reverse top enhanced nobox 
set key samplen 1 spacing 1.2 height 0.2 font ',13' at 4.7, 30 noopaque
#maxrows 3 width 0.5 height 0.5 

### Y-axis
set ylabel "Cache Hit Rate (%)" offset 1.7,0
set yrange [0:100]
set ytics 0,20,100 offset 0.5,0 #format "%.1f"
set ytics nomirror

set y2label "Extracting Time (ms)"offset -1.7,0
set y2range [0:50]
set y2tics 0,10,50 offset -0.5,0 #format "%.1f"
set y2tics nomirror

### X-axis
set xlabel "Cache Ratio (%)" offset 0,0.7
set xrange [0:30]
set xtics 0, 5, 30  offset -0.2,0.3
set xtics nomirror

# GPU sampling, max cache 7%, which is 53G * 0.07 = 3.7G
set arrow   1 from 7, 0 to 7, 100 nohead lt 2 lw 3 lc rgb "#000000" back
set label   2 "3.7GB"  center at 7,  108 font ",14" tc rgb "#000000"  front

set label   3 "GPU-based" left at    5, 86 font ",12" tc rgb "#000000"  front
set label   4 "Sampling"  left at    5, 77 font ",12" tc rgb "#000000"  front
set object  5 rect from 5.8,72 to 7.8,92 fc 'white' fs solid 1.0 noborder front


# CPU sampling, max cache 21%, which is 53G * 0.21 = 11.1G
set arrow  11 from 21, 0 to 21, 100 nohead lt 2 lw 3 lc rgb "#000000" back
set label  12 "11.1GB" center at 21, 108 font ",14" tc rgb "#000000"  front

set label  13 "CPU-based" left at 18.0, 61 font ",12" tc rgb "#000000" front 
set label  14 "Sampling"  left at 18.0, 52 font ",12" tc rgb "#000000" front 
set object 15 rect from 20,47 to 22,67 fc 'white' fs solid 1.0 noborder front


# white box for key
set arrow  21 from  7, 6 to  7, 29 nohead lt 2 lw 10 lc rgb "white" noborder back
set arrow  22 from 21, 6 to 21, 18.3 nohead lt 2 lw 10 lc rgb "white" noborder back

# training time, this requires manual making from the original dat file
#set object 31 circle center 30,second 24.445 size 0.5 noclip front fillstyle border lc "#0000ee"  lw 6
set label  31 "+" center at 29.8, second 24.445 font ",24" tc rgb "#000000"  front rotate by 45
set label  32 "Training" right at 28.5, second 18 font ",11" tc rgb "#000000"  front
set label  33 "    Time" right at 28.5, second 14 font ",11" tc rgb "#000000"  front
set arrow  34 from 28.0, second 19.5 to 29.3, second 22.2 nohead lt 1 lw 1 lc rgb "#000000" back

NonZero(t)=(t == 0 ? NaN : t)

plot dat_file using 2:(column(col_hit_percent))      title "Hit Rate"      with l lt 1 lw 3 lc rgb '#c00000' axis x1y1, \
     dat_file using 2:(column(col_copy_second)*1000) title "Extracting Time"  with l lt 1 lw 3 lc rgb '#0000ee' axis x1y2, \

##########################################################################################
