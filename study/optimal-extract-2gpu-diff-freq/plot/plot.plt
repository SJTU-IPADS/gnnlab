outputfname = "data.eps"
dat_file='data.dat'

col_boost=1
col_cache_size=2
col_model=3
col_sample_type=4
col_dataset=5

set datafile sep '\t'

set terminal postscript "Helvetica,16" eps enhance color dl 2
set style data histogram

set style histogram clustered gap 2
set style fill solid border -2
set pointsize 1
set size 0.8,0.5
set boxwidth 1 relative
# set no zeroaxis



set tics font ",14" scale 0.5

set rmargin 2
set lmargin 5.5
set tmargin 1.5
set bmargin 2.5

set output outputfname

#### magic to filter expected data entry from dat file
format_str="<awk -F'\\t' '{ if($".col_dataset."      ~ \"%s\"     && ". \
                              "$".col_cache_size."   ~ \"%s\"      ) { print }}' ".dat_file." "
cmd_filter_dat_by_cache_size(dataset, cache_size)=sprintf(format_str, dataset, cache_size)
##########################################################################################

### Key
set key inside left Left reverse top enhanced nobox 
set key samplen 1 spacing 1.2 height 0.2 width 0.5 font ',13' maxrows 1 at graph 0.02, graph 0.975 noopaque


set xrange [-.5:7.5]
set xtics nomirror offset -0.2,0.3

set arrow from 0-0.3,graph -0.11 to  3.3,graph -0.11 nohead lt 2 lw 1 lc "#000000" front
set arrow from 4-0.3,graph -0.11 to  7.3,graph -0.11 nohead lt 2 lw 1 lc "#000000" front
set label "Twitter"   center at 1.5, graph -0.17 font ",16" tc rgb "#000000" front
set label "UK-2006-05"    center at 5.5, graph -0.17 font ",16" tc rgb "#000000" front

set arrow from  graph 0, first 60 to graph 1, first 60 nohead lt 1 lw 1 lc "#000000" front

## Y-axis
set ylabel "Perf. Boost (%)" offset 1.7,0
set yrange [0:70]
set ytics 0,10,60
set ytics offset 0.5,0 #format "%.1f" #nomirror

# ^((?!PR).)*$
plot cmd_filter_dat_by_cache_size(".*", "11GB")      using (100-column(col_boost)):xticlabels(col_sample_type)  lc "#ff9900" title "16G V100" \
    ,cmd_filter_dat_by_cache_size(".*", "27GB")      using (100-column(col_boost)):xticlabels(col_sample_type)  lc "#c00000" title "32G V100" \
    ,cmd_filter_dat_by_cache_size(".*", "35GB")      using (100-column(col_boost)):xticlabels(col_sample_type)  lc "#000000" title "40G A100" \
    ,cmd_filter_dat_by_cache_size(".*", "75GB")      using (100-column(col_boost)):xticlabels(col_sample_type)  lc "#0044ee" title "80G A100" \
