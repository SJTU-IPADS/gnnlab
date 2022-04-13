outputfname = "optimal.svg"
dat_file='data.dat'

# col numbers
col_cache_policy = 1
col_cache_percent = 2
col_dataset = 3
col_sample_type = 4
col_app = 5
col_hit_percent = 6
col_optimal_hit_percent= 7
col_train_set_percent=8

end_of_1=6
end_of_2=12

set datafile sep '\t'
set output outputfname

# set terminal svg "Helvetica,16" enhance color dl 2 background rgb "white"
set terminal svg size 800,600 font "Helvetica,16" enhanced background rgb "white" dl 2
# set style data histogram
set multiplot layout 3,3

# set style histogram clustered gap 2
set style fill solid border -2
set pointsize 1
set boxwidth 0.5 relative
# set no zeroaxis

set tics font ",12" scale 0.5

set rmargin 1.6
set lmargin 5.5
set tmargin 1.5
set bmargin 2

#### magic to filter expected data entry from dat file
format_str="<awk -F'\\t' '{ if(". \
                              "$".col_dataset."      ~ \"%s\"     && ". \
                              "$".col_sample_type."  ~ \"%s\"     && ". \
                              "$".col_cache_policy." ~ \"%s\"     ". \
                              ") { print }}' ".dat_file." "
cmd_filter_dat_by_policy(dataset,sample_type, policy)=sprintf(format_str, dataset, sample_type, policy)
##########################################################################################

### Key
set key outside left Left reverse top enhanced box 
set key samplen 1 spacing 1.2 height 0.2 width 0.5 font ',13' maxrows 1 center at graph 0.5, graph 1.2 noopaque

set xlabel "Percent of Train Set" offset 0,1.9 font ",13"
set xrange [-.5:10.5]
set xtics rotate by -45
set xtics nomirror offset -0.2,0.3

## Y-axis
set ylabel "Optimal Hit Rate (%)" offset 3.5,0 font ",13"
set yrange [0:110]
set ytics 0,20,100 
set ytics offset 0.5,0 #format "%.1f" #nomirror

set arrow from  graph 0.5,graph 0 to graph 0.5,graph 1 nohead dashtype(3,2) lw 1 lc "#000000" front

unset key

set title "3-Hop Random" offset 0,-1
plot cmd_filter_dat_by_policy("PR", "kKHop2", "Deg")      using col_optimal_hit_percent:xticlabels(col_train_set_percent)       w l lw 2 ps 0.8 lc "#ff9900" title "PR" \
    ,cmd_filter_dat_by_policy("TW", "kKHop2", "Deg")      using col_optimal_hit_percent:xticlabels(col_train_set_percent)       w l lw 2 ps 0.8 lc "#c00000" title "TW" \
    ,cmd_filter_dat_by_policy("PA", "kKHop2", "Deg")      using col_optimal_hit_percent:xticlabels(col_train_set_percent)       w l lw 2 ps 0.8 lc "#000000" title "PA" \
    ,cmd_filter_dat_by_policy("UK", "kKHop2", "Deg")      using col_optimal_hit_percent:xticlabels(col_train_set_percent)       w l lw 2 ps 0.8 lc "#0044ee" title "UK" \

unset ylabel
set title "Random Walk" offset 0,-1
plot cmd_filter_dat_by_policy("PR", "kRandomWalk", "Deg")      using col_optimal_hit_percent:xticlabels(col_train_set_percent)       w l lw 2 ps 0.8  lc "#ff9900" title "PR" \
    ,cmd_filter_dat_by_policy("TW", "kRandomWalk", "Deg")      using col_optimal_hit_percent:xticlabels(col_train_set_percent)       w l lw 2 ps 0.8  lc "#c00000" title "TW" \
    ,cmd_filter_dat_by_policy("PA", "kRandomWalk", "Deg")      using col_optimal_hit_percent:xticlabels(col_train_set_percent)       w l lw 2 ps 0.8  lc "#000000" title "PA" \
    ,cmd_filter_dat_by_policy("UK", "kRandomWalk", "Deg")      using col_optimal_hit_percent:xticlabels(col_train_set_percent)       w l lw 2 ps 0.8  lc "#0044ee" title "UK" \

set title "3-Hop Weighted" offset 0,-1
plot cmd_filter_dat_by_policy("PR", "kWeightedKHopPrefix", "Deg")      using col_optimal_hit_percent:xticlabels(col_train_set_percent)       w l lw 2 ps 0.8  lc "#ff9900" title "PR" \
    ,cmd_filter_dat_by_policy("TW", "kWeightedKHopPrefix", "Deg")      using col_optimal_hit_percent:xticlabels(col_train_set_percent)       w l lw 2 ps 0.8  lc "#c00000" title "TW" \
    ,cmd_filter_dat_by_policy("PA", "kWeightedKHopPrefix", "Deg")      using col_optimal_hit_percent:xticlabels(col_train_set_percent)       w l lw 2 ps 0.8  lc "#000000" title "PA" \
    ,cmd_filter_dat_by_policy("UK", "kWeightedKHopPrefix", "Deg")      using col_optimal_hit_percent:xticlabels(col_train_set_percent)       w l lw 2 ps 0.8  lc "#0044ee" title "UK" \

set ylabel "PreSC#1 Hit Rate (%)" offset 3.5,0 font ",13"
unset title
plot cmd_filter_dat_by_policy("PR", "kKHop2", "PreS_1")      using col_hit_percent:xticlabels(col_train_set_percent)       w l lw 2 ps 0.8 lc "#ff9900" title "PR" \
    ,cmd_filter_dat_by_policy("TW", "kKHop2", "PreS_1")      using col_hit_percent:xticlabels(col_train_set_percent)       w l lw 2 ps 0.8 lc "#c00000" title "TW" \
    ,cmd_filter_dat_by_policy("PA", "kKHop2", "PreS_1")      using col_hit_percent:xticlabels(col_train_set_percent)       w l lw 2 ps 0.8 lc "#000000" title "PA" \
    ,cmd_filter_dat_by_policy("UK", "kKHop2", "PreS_1")      using col_hit_percent:xticlabels(col_train_set_percent)       w l lw 2 ps 0.8 lc "#0044ee" title "UK" \

set key
unset ylabel
plot cmd_filter_dat_by_policy("PR", "kRandomWalk", "PreS_1")      using col_hit_percent:xticlabels(col_train_set_percent)       w l lw 2 ps 0.8  lc "#ff9900" title "PR" \
    ,cmd_filter_dat_by_policy("TW", "kRandomWalk", "PreS_1")      using col_hit_percent:xticlabels(col_train_set_percent)       w l lw 2 ps 0.8  lc "#c00000" title "TW" \
    ,cmd_filter_dat_by_policy("PA", "kRandomWalk", "PreS_1")      using col_hit_percent:xticlabels(col_train_set_percent)       w l lw 2 ps 0.8  lc "#000000" title "PA" \
    ,cmd_filter_dat_by_policy("UK", "kRandomWalk", "PreS_1")      using col_hit_percent:xticlabels(col_train_set_percent)       w l lw 2 ps 0.8  lc "#0044ee" title "UK" \

unset key
plot cmd_filter_dat_by_policy("PR", "kWeightedKHopPrefix", "PreS_1")      using col_hit_percent:xticlabels(col_train_set_percent)       w l lw 2 ps 0.8  lc "#ff9900" title "PR" \
    ,cmd_filter_dat_by_policy("TW", "kWeightedKHopPrefix", "PreS_1")      using col_hit_percent:xticlabels(col_train_set_percent)       w l lw 2 ps 0.8  lc "#c00000" title "TW" \
    ,cmd_filter_dat_by_policy("PA", "kWeightedKHopPrefix", "PreS_1")      using col_hit_percent:xticlabels(col_train_set_percent)       w l lw 2 ps 0.8  lc "#000000" title "PA" \
    ,cmd_filter_dat_by_policy("UK", "kWeightedKHopPrefix", "PreS_1")      using col_hit_percent:xticlabels(col_train_set_percent)       w l lw 2 ps 0.8  lc "#0044ee" title "UK" \

set ylabel "Degree Hit Rate (%)" offset 3.5,0 font ",13"
unset title
plot cmd_filter_dat_by_policy("PR", "kKHop2", "Deg")      using col_hit_percent:xticlabels(col_train_set_percent)       w l lw 2 ps 0.8 lc "#ff9900" title "PR" \
    ,cmd_filter_dat_by_policy("TW", "kKHop2", "Deg")      using col_hit_percent:xticlabels(col_train_set_percent)       w l lw 2 ps 0.8 lc "#c00000" title "TW" \
    ,cmd_filter_dat_by_policy("PA", "kKHop2", "Deg")      using col_hit_percent:xticlabels(col_train_set_percent)       w l lw 2 ps 0.8 lc "#000000" title "PA" \
    ,cmd_filter_dat_by_policy("UK", "kKHop2", "Deg")      using col_hit_percent:xticlabels(col_train_set_percent)       w l lw 2 ps 0.8 lc "#0044ee" title "UK" \

set key
unset ylabel
plot cmd_filter_dat_by_policy("PR", "kRandomWalk", "Deg")      using col_hit_percent:xticlabels(col_train_set_percent)       w l lw 2 ps 0.8  lc "#ff9900" title "PR" \
    ,cmd_filter_dat_by_policy("TW", "kRandomWalk", "Deg")      using col_hit_percent:xticlabels(col_train_set_percent)       w l lw 2 ps 0.8  lc "#c00000" title "TW" \
    ,cmd_filter_dat_by_policy("PA", "kRandomWalk", "Deg")      using col_hit_percent:xticlabels(col_train_set_percent)       w l lw 2 ps 0.8  lc "#000000" title "PA" \
    ,cmd_filter_dat_by_policy("UK", "kRandomWalk", "Deg")      using col_hit_percent:xticlabels(col_train_set_percent)       w l lw 2 ps 0.8  lc "#0044ee" title "UK" \

unset key
plot cmd_filter_dat_by_policy("PR", "kWeightedKHopPrefix", "Deg")      using col_hit_percent:xticlabels(col_train_set_percent)       w l lw 2 ps 0.8  lc "#ff9900" title "PR" \
    ,cmd_filter_dat_by_policy("TW", "kWeightedKHopPrefix", "Deg")      using col_hit_percent:xticlabels(col_train_set_percent)       w l lw 2 ps 0.8  lc "#c00000" title "TW" \
    ,cmd_filter_dat_by_policy("PA", "kWeightedKHopPrefix", "Deg")      using col_hit_percent:xticlabels(col_train_set_percent)       w l lw 2 ps 0.8  lc "#000000" title "PA" \
    ,cmd_filter_dat_by_policy("UK", "kWeightedKHopPrefix", "Deg")      using col_hit_percent:xticlabels(col_train_set_percent)       w l lw 2 ps 0.8  lc "#0044ee" title "UK" \
