#! /bin/bash

mkdir  Example_Generate_Conformation

cd     Example_Generate_Conformation

python  ../Generate_Conformations_example.py     ../../data/Max_RMSd_Pair.pdb     ../../results/models/molearn_network_1000_from_Try-A.pth     101     GenConf_with_Model-A     1000
  
