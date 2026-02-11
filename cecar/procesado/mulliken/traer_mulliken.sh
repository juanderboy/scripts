#!/bin/bash
for i in {1..35}
do
	cp ${i}/mulliken_spin ms_${i}.dat
	cp ${i}/mulliken mq_${i}.dat
done

