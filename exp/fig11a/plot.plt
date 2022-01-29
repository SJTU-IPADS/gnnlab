outputfname = "fig11a.eps"
dat_file='data.dat'
fit_policy='degree'

# col numbers
col_cache_policy=1
col_cache_percent=2
col_dataset=3
col_sample_type=4
col_app=5
col_hit_percent=6
col_optimal_hit_percent=7

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
format_str="<awk -F'\\t' '{ if($".col_app."          == \"%s\"     && ". \
                              "$".col_sample_type."  == \"%s\"     && ". \
                              "$".col_cache_policy." == \"%s\"      ) { print }}' ".dat_file." "
cmd_filter_dat_by_policy(app, sample_type, policy)=sprintf(format_str, app, sample_type, policy)
##########################################################################################

### Key
set key inside right Right bottom invert enhanced nobox 
set key samplen 1 spacing 1.5 height 0.2 font ',13' #at graph 0.97, graph 0.63  noopaque

### X-axis
set xrange [0:35]
set xtics 0,5,35
set xlabel "Cache Ratio (%)" offset 0,0.5
set xtics nomirror offset -0.2,0.3

## Y-axis
# set yrange [0:100]
set ytics 20
set ylabel "Cache Hit Rate (%)" offset 2,0
set ytics nomirror offset 0.5,0 #format "%.1f"

plot cmd_filter_dat_by_policy("gcn", "kWeightedKHopPrefix", "degree")      using 2:col_optimal_hit_percent  w l lw 10 lc "#00ffff" title 'Optimal' \
    ,cmd_filter_dat_by_policy("gcn", "kWeightedKHopPrefix", "presample_1") using 2:col_hit_percent   w l lw 3 lc "#000000" title "PreSC#1" \
    ,cmd_filter_dat_by_policy("gcn", "kWeightedKHopPrefix", "presample_2") using 2:col_hit_percent   w l lw 3 lc "#444444" title "PreSC#2" \
    ,cmd_filter_dat_by_policy("gcn", "kWeightedKHopPrefix", "degree")      using 2:col_hit_percent   w l lw 3 lc "#c00000" title "Degree" \
    # ,cmd_filter_dat_by_policy("gcn", "kWeightedKHop", "presample_4") using 2:col_hit_percent   w l lw 3 lc "#888888" title "PreSC#4" \
