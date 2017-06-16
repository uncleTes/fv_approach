set term x11 persist
set xtic 1
set grid
set logscale y 2
set format y "%8.6e"
set size square
set style line 1 lw 3 lc rgb "blue"
set style line 2 lw 3 lc rgb "green"
set xlabel "refs"
set ylabel "norms"
plot filename using 1:2:ytic(2) title "inf" with linespoints ls 1, filename using 1:3:ytic(3) title "L2" with linespoints ls 2
# http://gnuplot.sourceforge.net/demo_5.0/lines_arrows.html
