# molearn4rna

*molearn application to sampling RNA conformational space*

This repository includes modified source codes of molearn version 1.0.0 
(https://github.com/Degiacomi-Lab/molearn/tree/diffusion) and related files to apply molearn to RNA.

Included in this repository are the following:
* Modified files to execute molearn for RNA
  * Molearn scripts ( 'protein_handler.py'  and  'network.py' ) are in the  'src/molearn'  folder.
  * Amber force field parameters for RNA (frcmod.RNA.LJbb and nucleic12.LJbb-RNA.lib) are in the  'src/parameters'  folder, 
    released in the [AmberTools22 package](https://ambermd.org/AmberTools.php) published under a GNU General Public Licence. 
    They are used to calculate Torch potential energy function for RNA.
  * Environment setup file (molearnA.yml) is in the  'src/environment'  folder.
*  detailing:
  * Examples of how to execute training and conformation generation with molearn, which are found in the  'examples'  folder.

Obtained from the other repository are the following:
* PDB files in the ' data'  folder, discussed in [Ikuo Kurisaki, Michiaki Hamada (2025). Deep learning generates apo RNA conformations 
                                                  with cryptic ligand binding site, bioRxiv](https://doi.org/10.1101/2025.01.07.631832)
  * Training input data, consisting of 400 snapshot structures is  'HIV1TAR_in_Two_States.pdb'  
    and input PDB file for conformation generation with molearn models is  'Max_RMSd_Pair.pdb' . 
    They should be downloaded from https://waseda.app.box.com/folder/300778804035?v=data-molearn4rna in advance, 
    and should be copied in  'data'  folder  in your machine before running an example script.
  * Molearn generated conformations (e.g., MolGen_HIVTAR_from_Model-A.pdb) are found in the  'results/pdb'  folder. 
    They can be downloaded from https://waseda.app.box.com/folder/300778804035?v=data-molearn4rna.
*  Results discussed in [Ikuo Kurisaki, Michiaki Hamada (2025). Deep learning generates apo RNA conformations 
                         with cryptic ligand binding site, bioRxiv](https://doi.org/10.1101/2025.01.07.631832)
  * Trained molearn models in the  'results/model/'  folder, which should be downloaded from 
    https://waseda.app.box.com/folder/300778804035?v=data-molearn4rna in advance. 
    Grid points for MV2003-binding conformations are given in the file  'results/moddel/Grid_Points_for_Conformations.txt' .

## Repository structure

molearn4rna-main/ <br>
├── data/               * Input datasets for training and conformation generation <br>
├── examples/           * sample scripts for training and conformation generation <br>
├── results/            * Output results (QRNA opt. structures) <br>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── model              * Trained models by 1000 epoch <br>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── pdb                * Generated conformations <br>
└── src/                * Source code <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── molearn/           * Scripts for protein_handler and network <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── parameters/        * Source files for amber force field  <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── environment/       * yml file from installation test <br>

 'data' ,  'results/model'  and  'results/pdb'  are empty folders. Files should be obtained from
  https://waseda.app.box.com/folder/300778804035?v=data-molearn4rna.

## Requirements ##

We tested Molearn with the modified codes on Python 3.10 on Rocky Linux 8.6 
and the following packages (and their associated packages):
* numpy
* PyTorch
* Biobox(https://github.com/Degiacomi-Lab/biobox)
* openMM (https://github.com/openmm/openmm)
For other packages, dependencies are described in the ' src/environment/molearnA.yml' .


## Installation ##
<br>
1) Install molearn into a local environment <br>
* Download “molearn-diffusion.zip” from https://github.com/Degiacomi-Lab/molearn/tree/diffusion in advance. <br>
* Download “molearn4rna-main.zip” from https://github.com/hmdlab/molearn4rna in advance. <br>
* Copy “molearn-diffusion.zip” and “molearn4rna-main.zip” in your working directory <br>
&nbsp;&nbsp;&nbsp;% cd PATH/to/the working directory <br>
&nbsp;&nbsp;&nbsp;% unzip molearn-diffusion.zip <br>
&nbsp;&nbsp;&nbsp;% cd molearn-diffusion <br>
&nbsp;&nbsp;&nbsp;% conda create --name  molearn4rna   python=3.10 <br>
&nbsp;&nbsp;&nbsp;( or % conda env create -f PATH/TO/molearn4rna-main/src/environment/molearn4rna.yml)<br>
&nbsp;&nbsp;&nbsp;% conda activate molearn4rna <br>
&nbsp;&nbsp;&nbsp;% conda install  numpy cython  scipy  pandas scikit-learn #<--required for installing biobox  <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  #<--If molearn4rna is created via the yml file, this step could be skipped <br>
&nbsp;&nbsp;&nbsp;% pip install openmm #<-- install openmm <br>
&nbsp;&nbsp;&nbsp;% git clone https://github.com/Degiacomi-Lab/biobox.git<--download biobox <br>
&nbsp;&nbsp;&nbsp;% cd biobox  <br>
&nbsp;&nbsp;&nbsp;% pip install . #<--install biobox <br>
&nbsp;&nbsp;&nbsp;% cd ../ <br>
&nbsp;&nbsp;&nbsp;% pip install . #<--install molearn <br>
&nbsp;&nbsp;&nbsp;% conda install pytorch torchvision torchaudio cpuonly -c pytorch #<--install torch <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                                                                  #<--If molearn4rna is created via the yml file, this step could be skipped <br>
 <br>
2) Modify molearn to apply it to RNA <br>
* Download “molearn4rna-main.zip” from https://github.com/hmdlab/molearn4rna in advance. <br>
* Copy items in the molearn4rna-main into the molearn installed directory <br>
&nbsp;&nbsp;&nbsp;%cp PATH/To/molearn4rna-main/src/molearn/*.py PATH/To/conda_local/conda/envs/molearn4rna/lib/python3.10/site-packages/molearn <br>
&nbsp;&nbsp;&nbsp;%cp PATH/To/molearn4rna-main/src/parameters/* PATH/To/conda_local/conda/envs/molearn4rna/lib/python3.10/site-packages/molearn/parameters <br>
&nbsp;&nbsp;&nbsp;%cd PATH/To/conda_local/conda/envs/molearn4rna/lib/python3.10/site-packages/molearn <br>
 <br>
3) Run Examples
* Download PDB files for the example from https://waseda.app.box.com/folder/300778804035?v=data-molearn4rna and copy PATH/To/molearn4rna-main/data <br>
* Download trained molearn models for the example from https://waseda.app.box.com/folder/300778804035?v=data-molearn4rna and copy PATH/To/molearn4rna-main/rusults/model <br>
&nbsp;&nbsp;&nbsp;% cd PATH/To/molearn4rna-main/examples <br>
&nbsp;&nbsp;&nbsp;% chmod +x *sh <br>
&nbsp;&nbsp;&nbsp;% ./run_Traning_Molearn.sh  <br>
* Computation may take several hour with standard CPU machine. <br> 
* Using smaller  'iter_per_epoch'  in 'Training_Molearn_example.py'  is one option to perform a test run quickly. <br>
&nbsp;&nbsp;&nbsp;% ./run_Generate_Conformation.sh <br>
* Conformations are generated for a set of grid points (x, y), where x and y ranges from 0 to 1 with 0.01 interval.  <br>
* 101 files are generated and each of them has 101 snapshot structures.  <br>
* The filename is like MolGen__50__GenConf_with_Model-A.pdb, denoting that conformations are generated  <br>
* by results/model/molearn_network_1000_from_Try-A.pth (Labeled by A) and x is 0.49 ((50 -1)/100). y ranges from 0 to 1 by 0.01 interval. <br>
* It is noted that, before further analyses, each of generated conformations should be refined  <br>
* by using molecular mechanics simulations such as QRNAS to relax unexpected steric distortions <br>
* (see for details Ikuo Kurisaki, Michiaki Hamada (2025). Deep learning generates apo RNA conformations 
   with cryptic ligand binding site, bioRxiv(https://doi.org/10.1101/2025.01.07.631832)). <br>


## Reference ##

If you use molearn with this modification in your work, besides the original study of molearn,
V.K. Ramaswamy, S.C. Musson, C.G. Willcocks, M.T. Degiacomi (2021). 
 Learning protein conformational space with convolutions and latent interpolations, Physical Review X 11 (https://journals.aps.org/prx/abstract/10.1103/PhysRevX.11.011052)

please cite:
Ikuo Kurisaki, Michiaki Hamada (2025). Deep learning generates apo RNA conformations 
 with cryptic ligand binding site, bioRxiv (https://doi.org/10.1101/2025.01.07.631832)

## Contact ##

If you have any issues or questions please contact mhamada@waseda.jp; ikuo.kurisaki@aoni.waseda.jp.



