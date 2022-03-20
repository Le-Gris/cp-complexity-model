#!/bin/bash
cur_path=`pwd`
path='/home/labo/cp-complexity-model/results'
exps=(9 16 17 18 19 20 21 22 23 24 25 26)

# Export figures and plots from each folder into export
cd $path
for e in ${exps[@]}; do
	cp -r 'exp'${e}'/figures' 'export/exp'${e}
done
