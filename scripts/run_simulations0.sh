#!/bin/bash
sim_path='src/run_simulations.py'

#Run experiment 9
config='config/experiment9.json'
python ${sim_path} -c ${config}


# Run all experiments from 16 to 16
for i in {16..26}; do
	config='config/experiment'${i}'.json'
	python ${sim_path} -c ${config}
done
