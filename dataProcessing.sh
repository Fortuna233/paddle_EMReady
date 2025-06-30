#!bin/bash

cd ../training_and_validation_sets

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

mkdir pdbFiles
mkdir depoFiles
mkdir simuFiles

mv $(ls | grep pdb1) pdbFiles
mv $(ls | grep deposited.mrc) depoFiles
mv $(ls | grep simulated.mrc) simuFiles

