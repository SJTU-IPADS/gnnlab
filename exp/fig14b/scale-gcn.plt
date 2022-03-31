#!/usr/bin/env gnuplot

reset
set output outfile

set terminal postscript "Helvetica,16" eps enhance color dl 2
set pointsize 1
set size 0.4,0.4
set zeroaxis

set tics font ",14" scale 0.5

# set rmargin 2 #2
# set lmargin 5 #5.5
# set tmargin 0.5 #1.5
# set bmargin 1 #2.5

set rmargin 2
set lmargin 5.5
set tmargin 1.5
set bmargin 2.5

### Key
set key inside right Right top enhanced nobox 
set key samplen 1.5 spacing 1.5 height 0.2 font ',11' noopaque #maxrow 3 #at 7.8, 15


### X-axis
set xrange [1:8]
set xtics 1,1,8
set xlabel "Number of GPUs" offset 0,0.7
set xtics nomirror offset -0.2,0.3

## Y-axis
set yrange [0:16]
set ytics 0,4,16
set ylabel "Epoch Time (sec)" offset 1.5,0
set ytics offset 0.5,0 #format "%.1f"  #nomirror


plot resfile u ($1):($2) t "DGL"     w lp lt 1 lw 3 pt  4 ps 1.5 lc rgb '#c00000', \
     resfile u ($1):($3) t "T_{SOTA}"  w lp lt 1 lw 3 pt  3 ps 1.5 lc rgb '#ff9900', \
     resfile u ($1):($4) t "FGNN/1S" w lp lt 1 lw 3 pt  6 ps 1.5 lc rgb '#008800', \
     resfile u ($1):($5) t "FGNN/2S" w lp lt 1 lw 3 pt  8 ps 1.5 lc rgb '#00bb00', \
     resfile u ($1):($6) t "FGNN/3S" w lp lt 1 lw 3 pt  2 ps 1.5 lc rgb '#00dd00'
