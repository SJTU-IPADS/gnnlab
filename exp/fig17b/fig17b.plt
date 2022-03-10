outputfname = "fig17b.eps"
dat_file='fig17b.dat'

set datafile sep '\t'

set terminal postscript "Helvetica,16" eps enhance color dl 2
set style data histogram

set style histogram clustered gap 2
set style fill solid border -2
set pointsize 1
set size 0.4,0.4
set boxwidth 0.6 relative
# set no zeroaxis

set tics font ",14" scale 0.5

set rmargin 0
set lmargin 5
set tmargin 0.5
set bmargin 1

set output outputfname

### Key
set key inside right Right  top enhanced nobox autotitles columnhead
set key samplen 1.5 spacing 1.5 height 0.2 width 0.5 font ',13'  #at graph 1, graph 0.975 noopaque

set xrange [-.5:8.5]
set xtics nomirror offset -0.2,0.3

set arrow from 0-0.4,graph -0.11 to  2.3,graph -0.11 nohead lt 2 lw 1 lc "#000000" front
set arrow from 3-0.4,graph -0.11 to  5.3,graph -0.11 nohead lt 2 lw 1 lc "#000000" front
set arrow from 6-0.4,graph -0.11 to  8.3,graph -0.11 nohead lt 2 lw 1 lc "#000000" front
set label "GCN"          center at 1, graph -0.18 font ",14" tc rgb "#000000" front
set label "GraphSAGE"    center at 4, graph -0.18 font ",14" tc rgb "#000000" front
set label "PinSAGE"      center at 7, graph -0.18 font ",14" tc rgb "#000000" front

set arrow from  graph 0, first 100 to graph 1, first 100 nohead lt 1 lw 1 lc "#000000" front

## Y-axis
set ylabel "Epoch Time (sec)" offset 1.2,0
set yrange [0:20]
set ytics 5
set ytics offset 0.5,0 #format "%.1f" #nomirror

# ^((?!PR).)*$
plot "fig17b.dat"      using 2:xticlabels(1)       lc "#c00000" title "DGL" \
    ,"fig17b.dat"      using 3:xticlabels(1)       lc "#008800" title "FGNN" \
