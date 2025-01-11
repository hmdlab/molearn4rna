#Copyright (C) 2024  
#Ikuo Kurisaki, Michiaki Hamada  

#This file is part of molearn4rna, a file suit to run Molearn (https://github.com/Degiacomi-Lab/molearn/tree/diffusion) for RNA.

#molearn4rna consists of free programs and related files: you can redistribute it and/or modify  
#it under the terms of the GNU General Public License as published by  
#the Free Software Foundation, either version 3 of the License, or  
#(at your option) any later version.  

#This file is distributed in the hope that it will be useful,  
#but WITHOUT ANY WARRANTY; without even the implied warranty of  
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the  
#GNU General Public License for more details.  

#You should have received a copy of the GNU General Public License  
#along with this file. If not, see <http://www.gnu.org/licenses/>.  

# Original code from molearn_diffusion by Venkata K. Ramaswamy, Samuel C. Musson, Chris G. Willcocks, Matteo T. Degiacomi.
# Licensed under GPL-2.0-or-later (https://www.gnu.org/licenses/gpl-2.0.html)
# Modifications by Ikuo Kurisaki and Michiaki Hamada, 2022.11.16.

# Copyright (c) 2021 Venkata K. Ramaswamy, Samuel C. Musson, Chris G. Willcocks, Matteo T. Degiacomi
#
# Molearn is free software ;
# you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation ;
# either version 2 of the License, or (at your option) any later version.
# molearn is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY ;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with molearn ;
# if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.

# Changes:
# L.  57- 59 - import sys module (2022/11/16).
# L.  70- 74 - Read PDB file from standard input (2022/11/16).
# L.  76- 79 - Batch size is changed from 4 to 5 (2022/11/16).
# L.  82- 87 - Initial and final epoches are read from standard input (2022/11/16).
# L.  89- 92 - Iteration number in epoch is changed from 5 to 1000 (2022/11/16).
# L.  98-103 - RNA atoms are set to find in PDB file (2022/11/16).
# L. 126-132 - Num_worker is changed from 6 to 5, which matches batch size (2022/11/16).
# L. 137-141 - Network parameter, m, is changed from 2 to 1.8 (2022/11/16).
# L. 144-146 - This print command is moved below (2022/11/16).
# L. 157-165 - Training profiles are recorded (2022/11/16).
# L. 170-174 - Molearn parameters are read from standard input, if traning restarts (2022/11/16).
# L. 176-182 - Both initial and final epoches are set (2022/11/16).
# L. 224-239 - Amber energy terms are scaled with different way (2022/11/16).
# L. 270-275 - Molearn parameters are save by 5 epoch (2022/11/16).


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

# Modification on 2022/11/16
import sys
# Done

from copy import deepcopy
import biobox

from molearn import load_data
from molearn import Auto_potential
from molearn import Autoencoder

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Modification on 2022/11/16
# floc = ["MurD_test.pdb"] # test protein (contains only 16 conformations of the MurD protein)
#
floc = [sys.argv[1]]
# Done

# Modification on 2022/11/16
#batch_size = 4 # if this is too small, gpu utilization goes down
#
batch_size = 1
# Done

# Modification on 2022/11/16
path_to_molearn_network_parm  =  sys.argv[2]

Root_of_Grid_Point_Number     =  int(sys.argv[3])

Output_Filename_Annotation    =  sys.argv[4]

epoch = sys.argv[5]

# Done


iter_per_epoch = 5 #use higher iter_per_epoch = 1000 for smoother plots (iter_per_epoch = smoothness  of statistics)


method = 'roll' # 3 method for  available in Auto_potential: 'roll', 'convolutional', 'indexing'

# load multiPDB and create loss function
# Modification on 2022/11/16
#dataset, meanval, stdval, atom_names, mol, test0, test1 = load_data(floc[0], atoms = ["CA", "C", "N", "CB", "O"], device=device)
#
dataset, meanval, stdval, atom_names, mol, test0, test1 = load_data(floc[0], atoms = ["P","O3'","O5'","OP1","OP2","C4'","C2'","C3'","C5'","O4'","O2'","C1'",\
                                                                                      "N1","N2","N3","N4","N6","N7","N9","C2","C4","C5","C6","C8","O2","O4","O6"], device=device)
# Done


lf = Auto_potential(frame=dataset[0]*stdval, pdb_atom_names=atom_names, method = method, device=torch.device('cpu'))

# Saving test structures (the most extreme conformations in terms of RMSD)
# Remember to rescale with stdval, permute axis from [3,N] to [N,3]
# unsqueeze to [1, N, 3], send back to cpu, and convert to numpy array.
crds =  (test0*stdval).permute(1,0).unsqueeze(0).data.cpu().numpy()
mol.coordinates = crds
mol.write_pdb("TEST0.pdb")

crds =  (test1*stdval).permute(1,0).unsqueeze(0).data.cpu().numpy()
mol.coordinates = crds
mol.write_pdb("TEST1.pdb")

# helper function to make getting another batch of data easier
def cycle(iterable):
   while True:
       for x in iterable:
           yield x

########################################################################

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(dataset.float()),
    batch_size=batch_size, shuffle=True, drop_last=True, num_workers=6)

iterator = iter(cycle(train_loader))

# define networks
# Modification on 2022/11/16
#network = Autoencoder(m=2.0, latent_z=2, r=2).to(device)
#
network = Autoencoder(m=1.8, latent_z=2, r=2, depth=4).to(device)
checkpoint = torch.load(f'{path_to_molearn_network_parm}')
network.load_state_dict(checkpoint[f'model_state_dict_{epoch}'])
# Done


# Modification on 2022/11/16
#print("> Network parameters: ", len(torch.nn.utils.parameters_to_vector(network.parameters())))
# Done

# define optimisers

# Modification on 2022/11/16
#optimiser = torch.optim.Adam(network.parameters(), lr=0.001, amsgrad=True)
#Done

#training loop

# Modification on 2022/11/16
#while (epoch<20):

#    print("> epoch: ", epoch)
#    for i in range(iter_per_epoch):
        # get two batches of training data
#        x0 = next(iterator)[0].to(device)
#        x1 = next(iterator)[0].to(device)
#        optimiser.zero_grad()

        #encode
#        z0 = network.encode(x0)
#        z1 = network.encode(x1)

        #interpolate
#        alpha = torch.rand(x0.size(0), 1, 1).to(device)
#        z_interpolated = (1-alpha)*z0 + alpha*z1
        

        #decode
#        out = network.decode(z0)[:,:,:x0.size(2)]
#        out_interpolated = network.decode(z_interpolated)[:,:,:x0.size(2)]

        #calculate MSE
#        mse_loss = ((x0-out)**2).mean() # reconstructive loss (Mean square error)
#        out *= stdval
#        out_interpolated *= stdval

        #calculate physics for interpolated samples
#        bond_energy, angle_energy, torsion_energy, NB_energy = lf.get_loss(out_interpolated)
        #by being enclosed in torch.no_grad() torch autograd cannot see where this scaling
        #factor came from and hence although mathematically the physics cancels, no gradients
        #are found and the scale is simply redefined at each step
        

        #with torch.no_grad():
        #    scale = 0.1*mse_loss.item()/(bond_energy.item()+angle_energy.item()+torsion_energy.item()+NB_energy.item())
        #network_loss = mse_loss + scale*(bond_energy + angle_energy + torsion_energy + NB_energy)

        #determine gradients
#        network_loss.backward()

        #advance the network weights
#        optimiser.step()

    #save interpolations between test0 and test1 every 5 epochs
#    epoch+=1
#    if epoch%5 == 0:
#        interpolation_out = torch.zeros(20, x0.size(2), 3)
        #encode test with each network
        #Not training so switch to eval mode
#        network.eval()
#        with torch.no_grad(): # don't need gradients for this bit
#            test0_z = network.encode(test0.unsqueeze(0).float())
#            test1_z = network.encode(test1.unsqueeze(0).float())

            #interpolate between the encoded Z space for each network between test0 and test1
#            for idx, t in enumerate(np.linspace(0, 1, 20)):
#                interpolation_out[idx] = network.decode(float(t)*test0_z + (1-float(t))*test1_z)[:,:,:x0.size(2)].squeeze(0).permute(1,0).cpu().data
#            interpolation_out *= stdval

        # Remember to switch back to train mode when you are done
#        network.train()

        #save interpolations
#        mol.coordinates = interpolation_out.numpy()
#        mol.write_pdb("epoch_%s_interpolation.pdb"%epoch)

# Done


# Modification on 2022/11/16
# Conformation generation on grid points in latent space

x0 = next(iterator)[0].to(device)

Width  =  1.0 /(float(Root_of_Grid_Point_Number) - 1.0)

Fraction_List  =   [0.0] + [Width * i  for i in range(1,Root_of_Grid_Point_Number-1)] + [1.0]

count = 0


for Fraction in Fraction_List:
    
    count += 1
    
    interpolation_out = torch.zeros(Root_of_Grid_Point_Number,\
                                    x0.size(2), 3)
    
    network.eval()
    
    test0_z  =  torch.tensor([[[0.0],[Fraction]]])
    test1_z  =  torch.tensor([[[1.0],[Fraction]]])
    
    # Generate coordinates from grid points in the latent space

    for idx, t in enumerate(np.linspace(0, 1, Root_of_Grid_Point_Number)):
       interpolation_out[idx] = network.decode(float(t)*test1_z + (1-float(t))*test0_z)[:,:,:x0.size(2)].squeeze(0).permute(1,0).cpu().data
    interpolation_out *= stdval

    # Retain coordinate information
    copy            = mol.coordinates.copy()

    #save coordinates as PDB file    
    mol.coordinates = interpolation_out.numpy()
    mol.write_pdb(f"MolGen__{count}__{Output_Filename_Annotation}.pdb")
    
    #recover mol.coordinates
    mol.coordinates = copy

# Done


