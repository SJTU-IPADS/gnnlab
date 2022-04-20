outputfname = "figure.svg"
dat_file='data.dat'

# col numbers
col_dataset = 1
col_hop = 2
col_percent = 3

set datafile sep ','
set output outputfname

# set terminal svg "Helvetica,16" enhance color dl 2 background rgb "white"
set terminal svg size 800,400 font "Helvetica,16" enhanced background rgb "white" dl 2
# set style data histogram
set multiplot layout 2,3

# set style histogram clustered gap 2
set style fill solid border -2
set pointsize 1
set boxwidth 0.5 relative
# set no zeroaxis

set tics font ",14" scale 0.5

set rmargin 2
set lmargin 5.5
set tmargin 1.5
set bmargin 2

#### magic to filter expected data entry from dat file
format_str="<awk -F'\\t' '{ if(". \
                              "$".col_dataset." ~ \"%s\"     ". \
                              ") { print }}' \"%s\" "
cmd_filter_dat_by_policy(dataset, filename)=sprintf(format_str, dataset, filename)
##########################################################################################

### Key
set key outside left Left reverse top enhanced box 
set key samplen 1 spacing 1.2 height 0.2 width 0.5 font ',13' maxrows 1 center at graph 0.5, graph 1.2 noopaque

set xlabel "Hop" offset 0,0.7
set xrange [:10]
set xtics rotate by -15
set xtics nomirror offset -0.2,0.1

## Y-axis
set ylabel "Touched Nodes(%)" offset 3.5,0
set yrange [0:110]
set ytics 0,20,100 
set ytics offset 0.5,0 #format "%.1f" #nomirror

# unset key

plot cmd_filter_dat_by_policy("products",  "percent-0.99.dat")      using col_hop:col_percent       w l lw 2 ps 0.8 lc "#ff9900" title "PR" \
    ,cmd_filter_dat_by_policy("twitter",   "percent-0.99.dat")      using col_hop:col_percent       w l lw 2 ps 0.8 lc "#c00000" title "TW" \
    ,cmd_filter_dat_by_policy("uk-2006-05","percent-0.99.dat")      using col_hop:col_percent       w l lw 2 ps 0.8 lc "#0044ee" title "UK" \

plot cmd_filter_dat_by_policy("products",  "percent-0.3.dat")      using col_hop:col_percent       w l lw 2 ps 0.8 lc "#ff9900" title "PR" \
    ,cmd_filter_dat_by_policy("twitter",   "percent-0.3.dat")      using col_hop:col_percent       w l lw 2 ps 0.8 lc "#c00000" title "TW" \
    ,cmd_filter_dat_by_policy("uk-2006-05","percent-0.3.dat")      using col_hop:col_percent       w l lw 2 ps 0.8 lc "#0044ee" title "UK" \

plot cmd_filter_dat_by_policy("products",  "percent-0.1.dat")      using col_hop:col_percent       w l lw 2 ps 0.8 lc "#ff9900" title "PR" \
    ,cmd_filter_dat_by_policy("twitter",   "percent-0.1.dat")      using col_hop:col_percent       w l lw 2 ps 0.8 lc "#c00000" title "TW" \
    ,cmd_filter_dat_by_policy("uk-2006-05","percent-0.1.dat")      using col_hop:col_percent       w l lw 2 ps 0.8 lc "#0044ee" title "UK" \

plot cmd_filter_dat_by_policy("products",  "percent-0.01.dat")      using col_hop:col_percent       w l lw 2 ps 0.8 lc "#ff9900" title "PR" \
    ,cmd_filter_dat_by_policy("twitter",   "percent-0.01.dat")      using col_hop:col_percent       w l lw 2 ps 0.8 lc "#c00000" title "TW" \
    ,cmd_filter_dat_by_policy("uk-2006-05","percent-0.01.dat")      using col_hop:col_percent       w l lw 2 ps 0.8 lc "#0044ee" title "UK" \

plot cmd_filter_dat_by_policy("products",  "percent-0.003.dat")      using col_hop:col_percent       w l lw 2 ps 0.8 lc "#ff9900" title "PR" \
    ,cmd_filter_dat_by_policy("twitter",   "percent-0.003.dat")      using col_hop:col_percent       w l lw 2 ps 0.8 lc "#c00000" title "TW" \
    ,cmd_filter_dat_by_policy("uk-2006-05","percent-0.003.dat")      using col_hop:col_percent       w l lw 2 ps 0.8 lc "#0044ee" title "UK" \
