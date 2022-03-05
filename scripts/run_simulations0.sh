#!/bin/bash
sim_path='src/run_simulations.py'

for i in {16...25}; do
	config='config/experiment'${i}'.json'
	python ${sim_path} -c ${config}
done
