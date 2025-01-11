#! /bin/bash

mkdir    Example_Training_Molearn

cd       Example_Training_Molearn

#cp ../Training_Molearn_example.py   ../data/Molearn_Input_PDB/HIV1TAR_in_Two_States.pdb  ./ 

python     ../Training_Molearn_example.py    ../../data/HIV1TAR_in_Two_States.pdb      0    10

cd  .. 

