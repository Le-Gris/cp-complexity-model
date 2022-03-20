#!/bin/bash
export PYTHONPATH=$PYTHONPATH:~/cp-complexity-model
sim_path='src/run_simulations.py'

# Get user input to choose correct case
echo '0 : Simulations 9-15'
echo '1 : Simulations 9, 16-26'
echo 'Choose from cases[0\1]'
read c

# Case 0
if [[ $c -eq 0 ]]; then
	# Run all experiments from 10 to 15 (9 already exists)
        for i in {10..15}; do
                echo "------ Experiment $i ------"
		config='config/experiment'${i}'.json'
                python ${sim_path} -c ${config}
        done
# Case 1
elif [[ $c -eq 1 ]]; then
	#Run experiment 9
	config='config/experiment9.json'
	python ${sim_path} -c ${config}

	# Run all experiments from 16 to 26
	for i in {16..26}; do
		config='config/experiment'${i}'.json'
		python ${sim_path} -c ${config}
	done
else
	echo 'Case does not exist!'
	return 1
fi
