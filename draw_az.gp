#! /usr/bin/gnuplot -persist
set terminal postscript
set output "./z_decrese_dynamic.eps"
set xlabel "Iteration" font "Helvetica,18"
set ylabel "vcos & z"font "Helvetica,20"
set key top left
set style line 1 lt -1 pt 9

set xrange [*:*]
set log xy

#plot [*:*][*:*] "./z_decrese_dynamic" using 1:3 with  lines  linestyle 1 title "Az"
plot "./z_decrese_dynamic" using 1:2 with  lines  linestyle 1 title "abs(Vcos)", "./z_decrese_dynamic" using 1:3 with  lines  linestyle 1  title "norm(z)"

