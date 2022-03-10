outputfname = "fig16a.eps"
dat_file='acc_one.res'

# col numbers
col_system=1
col_dataset=2
col_batch_size=3
col_time=4
col_acc=5

# split location
split_location=50
end_location=400

set fit logfile '/dev/null'
set fit quiet
set datafile sep '\t'

set terminal postscript "Helvetica,16" eps enhance color dl 2
set pointsize 1
set size 0.51,0.4
set zeroaxis

set tics font ",14" scale 0.5

set output outputfname

set multiplot layout 1,2

#### magic to filter expected data entry from dat file
format_str="<awk -F'\\t' '{ if($".col_system."      == \"%s\"     && ". \
                              "$".col_dataset." == \"%s\"      ) { print }}' ".dat_file." "
cmd_filter_dat_by_policy(system, dataset)=sprintf(format_str, system, dataset)
##########################################################################################

set size 0.30, 0.4
set origin 0,0
set rmargin 0.5
set lmargin 5.5
set tmargin 1.5
set bmargin 3
### Key
set key inside left Left reverse bottom invert enhanced nobox 
set key samplen 1 spacing 1.5 height 0.2 font ',13' at graph 0.18, graph 0.1 opaque

### X-axis
set xrange [0:split_location]
set xtics 0, 10, split_location-1
set xlabel "Time (sec)" offset 6.,0.5
set xtics nomirror offset -0.2,0.3

## Y-axis
set yrange [0:60]
set ytics 0,10,60
set ylabel "Accuracy (%)" offset 1.,0
set ytics offset 0.5,0 #format "%.1f"

# 56% acc
set arrow   1   from 0, 56 to split_location,56   nohead lt 3 lw 2 dashtype(3,2) lc rgb "#000000" back

set arrow   2   from 33, 56  to 33, 60 nohead lt 2 lw 3 lc rgb "#000000" back
set label   3   "33s"   center at 33,  63 font ",14" tc rgb   "#000000"  front

plot cmd_filter_dat_by_policy("dgl",  "papers") using col_time:col_acc  w l lw 3  lc "#c00000" title "DGL" \
    ,cmd_filter_dat_by_policy("sgnn", "papers") using col_time:col_acc  w l lw 3  lc "#ff9900" title "T_{SOTA}" \
    ,cmd_filter_dat_by_policy("fgnn", "papers") using col_time:col_acc  w l lw 3  lc "#008800" title "GNNLab/2S" \


# for second plot
set size 0.1+0.06, 0.4
set origin 0.30,0
# set rmargin 3
set lmargin 0
set tmargin 1.5
set bmargin 3

unset xlabel
unset ylabel
unset key
unset label
unset arrow

set xrange[split_location:end_location]
set xtics split_location,150,500
set ytics format ""

set arrow   1   from split_location, 56 to end_location,56   nohead lt 2 lw 3 dashtype(3,2) lc rgb "#000000" back
set label   2   "56%" left at end_location+20,56               font ",14" tc rgb   "#000000"  front

set arrow   3   from 117, 56 to 117,60 nohead lt 2 lw 3 lc rgb "#000000" back
set label   4   "117s"  center at 117,  63 font ",14" tc rgb   "#000000"  front

set arrow   5   from 335, 56 to 335,60 nohead lt 2 lw 3 lc rgb "#000000" back
set label   6   "335s"  center at 335,  63 font ",14" tc rgb   "#000000"  front


plot cmd_filter_dat_by_policy("dgl",  "papers")  using col_time:col_acc  w l lw 3  lc "#c00000" title "DGL" \
    ,cmd_filter_dat_by_policy("sgnn", "papers")  using col_time:col_acc  w l lw 3  lc "#ff9900" title "T_{SOTA}" \
    ,cmd_filter_dat_by_policy("fgnn", "papers")  using col_time:col_acc  w l lw 3  lc "#008800" title "GNNLab/2S" \
