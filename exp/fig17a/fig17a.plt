#!/usr/bin/env gnuplot

reset
set output "fig17a.eps"
set terminal postscript "Helvetica,16" eps enhance color dl 2

set pointsize 1
set size 0.4,0.4
set nozeroaxis

set tics font ",14" scale 0.5

set rmargin 1 #2
set lmargin 5 #5.5
set tmargin 0.5 #1.5
set bmargin 1 #2.5


set style data histogram
set style histogram clustered gap 2
set style fill solid border -2
set boxwidth 0.6 relative


### Key
set key inside right Right top enhanced nobox 
set key samplen 1.5 spacing 1.5 height 0.2 width 0 autotitles columnhead font ',13' #maxrows 1 at graph 0.02, graph 0.975  noopaque


## Y-axis
set ylabel "Epoch Time (sec)" offset 1.,0
set yrange [0:8]
set ytics 0,2,8 
set ytics offset 0.5,0 #format "%.1f" #nomirror


### X-axis
#set xlabel "Number of GPUs" offset 0,0.7
set xrange [-0.5:6.5]
#set xtics 1,1,8 
set xtics nomirror offset -0.2,0.3 rotate by -90

# set arrow from   0, graph -0.3 to  0, graph 0.0 nohead lt 1 lw 2 lc "#000000" front
# set arrow from   8, graph -0.3 to  8, graph 1.0 nohead lt 1 lw 2 lc "#000000" front
# set arrow from  15, graph -0.3 to 15, graph 1.0 nohead lt 1 lw 2 lc "#000000" front
# set arrow from  21, graph -0.3 to 21, graph 0.0 nohead lt 1 lw 2 lc "#000000" front

set datafile missing "-"


plot "fig17a.dat" using ($2):xticlabels(1) t "w/o DS"  w histogram lc rgb "#c00000",\
     "fig17a.dat" using ($3):xticlabels(1) t "w/  DS" w histogram lc rgb "#008800", \

##008800
