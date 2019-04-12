IFS=$'\n'       # make newlines the only separator
set -f          # disable globbing
for i in $(cat < "$1"); do

  rm -f mesh-c.vtu

  a1=$(echo "$i" | awk '{print $1}')
  sed -i "331s/.*/           f = $a1;  /" $2

  a2=$(echo "$i" | awk '{print $2}')
  sed -i "333s/.*/           f = $a2;  /" $2

  a3=$(echo "$i" | awk '{print $3}')
  sed -i "335s/.*/           f = $a3;  /" $2

  a4=$(echo "$i" | awk '{print $4}')
  sed -i "337s/.*/           f = $a4;  /" $2

  a5=$(echo "$i" | awk '{print $5}')
  sed -i "339s/.*/           f = $a5;  /" $2

  a6=$(echo "$i" | awk '{print $6}')
  sed -i "341s/.*/           f = $a6;  /" $2

  a7=$(echo "$i" | awk '{print $7}')
  sed -i "343s/.*/           f = $a7;  /" $2

  a8=$(echo "$i" | awk '{print $8}')
  sed -i "345s/.*/           f = $a8;  /" $2

  echo $a1 $a2 $a3 $a4 $a5 $a6 $a7 $a8

  make lung_tria
  ./lung_tria 6 2 0 /home/munch/projects/splines_raw.dat
  
  if [ -f mesh-c.vtu ]; then
    mv mesh-c.vtu "succ/mesh-$a1$a2$a3$a4$a5$a6$a7$a8.vtu"
  fi
done
