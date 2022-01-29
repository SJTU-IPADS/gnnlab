outputfname = "fig4b.eps"
src_file='data.dat'

set datafile sep '\t'

set terminal postscript "Helvetica,16 " eps enhance color dl 2
set size 0.4,0.4
set zeroaxis

set output outputfname


set key inside left Left reverse top enhanced nobox 
set key samplen 1 spacing 1.2 height 0.2 font ',13'  at 120, 95 noopaque

set rmargin 5.5
set lmargin 5.5
set tmargin 1.5
set bmargin 2.5
set tics font ",14" scale 0.5

##########################################################################################

set xlabel "Feature Dim" offset 0,0.7
set xrange [50:950]
set xtics 100,200,900 offset 0,0.3
set xtics nomirror

set ylabel "Cache Hit Rate (%)"  offset 1.7,0
set yrange [0:100]
set ytics 20 offset 0.5,0
set ytics nomirror

set y2label "Transfer Size (GB)" offset -1.7,0
set y2range[0:2]
set y2tics 0.5 format "%.1f" offset -0.5,0
set y2tics nomirror

set arrow  1 from 128, 0 to 128, 100 nohead lt 2 lw 3 lc rgb "#000000" back
set arrow  2 from 600, 0 to 602, 100 nohead lt 2 lw 3 lc rgb "#000000" back

set label 21 "OGB-Papers"  left at 50, 108 font ",14" tc rgb "#000000"  front
set label 22 "Reddit[22]=602"      left at 500, 108 font ",14" tc rgb "#000000"  front

NonZero(t)=(t == 0 ? NaN : t)

plot src_file using 9:6         w l lw 3 lc "#c00000" title "Hit Rate"  \
    ,src_file using 9:10 w l lw 3 lc "#0000ee" title "Data Size"   axis x1y2\
    # ,src_file using 1:2 w l lw 5 title "Cache Rate" smooth bezier \
    # ,src_file using 1:2 w l lc 0 title "Train Time"      axis x1y2 \

