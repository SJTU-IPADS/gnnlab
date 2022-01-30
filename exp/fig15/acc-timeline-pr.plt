outputfname = "fig15a.eps"
dat_file='acc_one.res'

# col numbers
col_system=1
col_dataset=2
col_batch_size=3
col_time=4
col_acc=5

# split location
split_location=20
end_location=140

set fit logfile '/dev/null'
set fit quiet
set datafile sep '\t'

set terminal postscript "Helvetica,16" eps enhance color dl 2
set pointsize 1
set size 0.4,0.4
set zeroaxis

set tics font ",14" scale 0.5

set rmargin 2
set lmargin 5.5
set tmargin 1.5
set bmargin 2.5

set output outputfname

set multiplot layout 1,2

#### magic to filter expected data entry from dat file
format_str="<awk -F'\\t' '{ if($".col_system."      == \"%s\"     && ". \
                              "$".col_dataset." == \"%s\"      ) { print }}' ".dat_file." "
cmd_filter_dat_by_policy(system, dataset)=sprintf(format_str, system, dataset)
##########################################################################################

set size 0.27, 0.4
set origin 0,0
set rmargin 0.5
set lmargin 5.5
set tmargin 1.5
set bmargin 1
### Key
set key inside left Left reverse bottom invert enhanced nobox 
set key samplen 1 spacing 1.5 height 0.2 width 5 font ',13'  at graph 0.15, graph 0.1 opaque


### X-axis
set xrange [0:split_location]
set xtics 0, 5, split_location-1
set xlabel "Time (sec)" offset 2.8,0.5
set xtics nomirror offset -0.2,0.3

## Y-axis
set yrange [0:100]
set ytics 0,20,100
set ylabel "Accuracy (%)" offset 2,0
set ytics offset 0.5,0 #format "%.1f"

# 91% acc
set arrow   1   from 0, 91 to split_location,91      nohead lt 3 lw 2 dashtype(3,2) lc rgb "#000000" back

set arrow   2   from 12, 91 to 12,100     nohead lt 2 lw 3 lc rgb "#000000" back
set label   3   "12s"     center at 12,  105 font ",13" tc rgb    "#000000"  front

plot cmd_filter_dat_by_policy("dgl",  "products")    using col_time:col_acc            w l lw 3  lc "#c00000" title "DGL" \
    ,cmd_filter_dat_by_policy("fgnn", "products")    using col_time:col_acc            w l lw 3  lc "#008800" title "FGNN/4S" \


# for second plot
set size 0.13, 0.4
set origin 0.27,0
set rmargin 3
set lmargin 0
set tmargin 1.5
set bmargin 1

unset xlabel
unset ylabel
unset key
unset label
unset arrow

set xrange[split_location:end_location]
set xtics split_location,60,end_location
set ytics format ""


set arrow   1   from split_location, 91 to end_location,91      nohead lt 3 lw 3 dashtype(3,2) lc rgb "#000000" back
set label   2   "91%" left at  end_location+10,91       font ",13" tc rgb   "#000000"  front

set arrow   3   from 120, 91 to 120,100   nohead lt 2 lw 3 lc rgb "#000000" back
set label   4   "120s"    center at 120,  105 font ",13" tc rgb   "#000000"  front

plot cmd_filter_dat_by_policy("dgl",  "products")    using col_time:col_acc            w l lw 3  lc "#c00000" title "DGL" \
    ,cmd_filter_dat_by_policy("fgnn", "products")    using col_time:col_acc            w l lw 3  lc "#008800" title "FGNN (4S 4T)" \
