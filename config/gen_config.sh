#!/bin/bash
paths=`ls ./src/exp*`
save='experiment'
for entry in $paths
do
	# Parse experiment number
	arrIN=(${entry//\// })
     	fname=${arrIN[2]}
	exp=(${fname//_/ })
	num=(${exp[0]//p/ })
	
	# Experiment name
	exp_name=$save${num[1]}'.json'
	
	# Run python script 
	python ${entry} -save ${exp_name}

done
