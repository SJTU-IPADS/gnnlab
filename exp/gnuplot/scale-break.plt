#!/usr/bin/env gnuplot

reset
set output "scale-break.eps"
set terminal postscript "Helvetica,16" eps enhance color dl 2

set pointsize 1
set size 0.8,0.4
set zeroaxis

set tics font ",14" scale 0.5

set rmargin 0 #2
set lmargin 5 #5.5
set tmargin 0.5 #1.5
set bmargin 1 #2.5


set style data histogram
set style histogram clustered gap 2
set style fill solid border -1
set boxwidth 0.6 relative


### Key
set key inside right Right top enhanced nobox 
set key samplen 1.5 spacing 1.5 height 0.2 width 0 font ',13' #maxrows 1 at graph 0.02, graph 0.975  noopaque


## Y-axis
set ylabel "Runtime (sec)" offset 1.,0
set yrange [0:5]
set ytics 0,1,5 
set ytics offset 0.5,0 #format "%.1f" #nomirror


### X-axis
#set xlabel "Number of GPUs" offset 0,0.7
set xrange [0:21]
#set xtics 1,1,8 
set xtics nomirror offset -0.2,0.3 rotate by -90

set arrow from   0, graph -0.3 to  0, graph 0.0 nohead lt 1 lw 2 lc "#000000" front
set arrow from   8, graph -0.3 to  8, graph 1.0 nohead lt 1 lw 2 lc "#000000" front
set arrow from  15, graph -0.3 to 15, graph 1.0 nohead lt 1 lw 2 lc "#000000" front
set arrow from  21, graph -0.3 to 21, graph 0.0 nohead lt 1 lw 2 lc "#000000" front

set datafile missing "-"


plot "scale-break.dat" using ($2):xticlabels(1) t "Sample"  w histogram lc rgb "#ff9900",\
     "scale-break.dat" using ($3):xticlabels(1) t "Extract" w histogram lc rgb "#c00000", \
     "scale-break.dat" using ($4):xticlabels(1) t "Train"   w histogram lc rgb "#0000ee", \
	 "scale-break.dat" using ($5):xticlabels(1) t "FGNN"    w lp lt 1 lw 3 pt 6 ps 1.5 lc rgb '#000000', \

##008800
