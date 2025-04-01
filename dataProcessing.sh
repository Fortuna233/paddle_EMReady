#!bin/bash
cd training_and_validation_sets

for i in $(ls .)
do
		ls $i
		mv $i/${i}.pdb1 ${i}.pdb1
		mv $i/deposited.mrc ${i}_deposited.mrc 
		mv $i/simulated.mrc ${i}_simulated.mrc 
		ls $i
		if [ -d $i ]
		then
			rm -r $i
		fi
done
