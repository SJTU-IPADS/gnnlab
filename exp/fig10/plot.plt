outputfname = "fig10.eps"
dat_file='data.dat'
fit_policy='degree'

# col numbers
col_cache_policy = 1
col_cache_percent = 2
col_dataset = 3
col_sample_type = 4
col_app = 5
col_hit_percent = 6
col_optimal_hit_percent= 7

set datafile sep '\t'

set terminal postscript "Helvetica,16" eps enhance color dl 2
set style data histogram

set style histogram clustered gap 2
set style fill solid border -2
set pointsize 1
set size 0.8,0.5
set boxwidth 0.5 relative
# set no zeroaxis



set tics font ",14" scale 0.5

set rmargin 2
set lmargin 5.5
set tmargin 1.5
set bmargin 2.5

set output outputfname

#### magic to filter expected data entry from dat file
format_str="<awk -F'\\t' '{ if($".col_dataset."      ~ \"%s\"     && ". \
                              "$".col_cache_policy." ~ \"%s\"      ) { print }}' ".dat_file." "
cmd_filter_dat_by_policy(dataset, policy)=sprintf(format_str, dataset, policy)
##########################################################################################

### Key
set key inside left Left reverse top enhanced nobox 
set key samplen 1 spacing 1.2 height 0.2 width 0.5 font ',13' maxrows 1 at graph 0.02, graph 0.975 noopaque


set xrange [-.5:11.5]
set xtics nomirror offset -0.2,0.3

set arrow from 0-0.3,graph -0.11 to  3.3,graph -0.11 nohead lt 2 lw 1 lc "#000000" front
set arrow from 4-0.3,graph -0.11 to  7.3,graph -0.11 nohead lt 2 lw 1 lc "#000000" front
set arrow from 8-0.3,graph -0.11 to 11.3,graph -0.11 nohead lt 2 lw 1 lc "#000000" front
set label "3-hop random"   center at 1.5, graph -0.17 font ",16" tc rgb "#000000" front
set label "Random walk"    center at 5.5, graph -0.17 font ",16" tc rgb "#000000" front
set label "3-hop weighted" center at 9.5, graph -0.17 font ",16" tc rgb "#000000" front

set arrow from  graph 0, first 100 to graph 1, first 100 nohead lt 1 lw 1 lc "#000000" front

## Y-axis
set ylabel "Cache Hit Rate (%)" offset 1.7,0
set yrange [0:118]
set ytics 0,20,100 
set ytics offset 0.5,0 #format "%.1f" #nomirror

# ^((?!PR).)*$
plot cmd_filter_dat_by_policy(".*", "degree")      using col_cache_percent:xticlabels(col_dataset)       lc "#ff9900" title "Random" \
    ,cmd_filter_dat_by_policy(".*", "degree")      using col_hit_percent:xticlabels(col_dataset)         lc "#c00000" title "Degree" \
    ,cmd_filter_dat_by_policy(".*", "presample_1") using col_hit_percent:xticlabels(col_dataset)         lc "#000000" title "PreSC#1" \
    ,cmd_filter_dat_by_policy(".*", "degree")      using col_optimal_hit_percent:xticlabels(col_dataset) lc "#0044ee" title "Optimal" \
