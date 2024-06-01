import torch
import os

import numpy as np
import pandas as pd
import pickle as pickle
from models import * 
import pandas as pd
import utils.Protein_Anlge_Trans as PAT
import argparse
import sys 
sys.path.append("..") 

column_names = ["phi", "psi", "omega", "tau", "CA:C:1N", "C:1N:1CA","0C:1N", "N:CA", "CA:C"]
def Sample(
            #Datasets_config
            dataset='cath',
            maxResidueLen = 128,
            patch_size = 4,
            #Codebook config
            num_codebook_vectors=50257,
            #Decoder config
            RM_d_model=1024,
            RM_nhead=16,
            RM_dim_feedforward=4096,
            RM_nlayers=24,
            #dropout
            dropout=0.,      
            #GPT config
            RM_model_path='./Models/cath_RecModel_29550.pt',
            pkeep=1.0,
            GPT_block_size=40,
            GPT_n_layers=24,
            GPT_n_heads=16,
            GPT_n_embd=1024,
            #sample config
            gpus='1',
            top_k=50,
            temperature=1.,
            sample_sum_perlen=10,
            output_len_l=50,
            output_len_r=128,
            model_name='cath_PBGPT_2600.pt',
            ):
        
    #dir_name = f'{dataset}_codebook_size_{num_codebook_vectors}'

    try:
        if not os.path.isdir(f'./Results/Sample/'):
            os.mkdir(f'./Results/Sample/')
    except:
        pass
    
    try:
        if not os.path.isdir(f'./Results/rama_sample/'):
            os.mkdir(f'./Results/rama_sample/')
    except:
        pass
    
    try:
        if not os.path.isdir(f'./Results/Sample/{dataset}/'):
            os.mkdir(f'./Results/Sample/{dataset}/')
    except:
        pass
        
    print('Here')
    os.makedirs(f'./Results/Sample/{dataset}/', exist_ok=True)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(0)


    ## Loading Scaler
    ms_dir=f'./Datasets/ScalerInfo/{dataset}.csv'
    df=pd.read_csv(ms_dir)
    ##
    print('Loading stored models...')
    
    pbgpt=PBGPT(maxResidueLen=maxResidueLen,
                RM_d_model=RM_d_model,
                RM_nhead=RM_nhead,
                RM_dim_feedforward=RM_dim_feedforward,
                RM_nlayers=RM_nlayers,
                dropout=dropout,
                n_dim=9,
                patch_size = patch_size,
                num_codebook_vectors=num_codebook_vectors,
                device=device,
                checkpoint_path=RM_model_path,
                pkeep=pkeep,
                GPT_block_size=GPT_block_size,
                GPT_n_layers=GPT_n_layers,
                GPT_n_heads=GPT_n_heads,
                GPT_n_embd=GPT_n_embd,
                ).to(device)

    #pbgpt.load_state_dict(torch.load(f'../Results/Models/PBGPTModel_{dir_name}/{model_name}'))
    pbgpt.load_state_dict(torch.load(f'./Models/{model_name}'))
    
    print("Sampling...")
    rama_phi=[]
    rama_psi=[]
    
    for output_len in range(output_len_l-1,output_len_r):
        print(f"    Sampling {sample_sum_perlen} PDB (length: {output_len+1})...")
        patch_length = output_len // patch_size
        if (output_len % patch_size > 0):
            patch_length += 1
        lengths = [patch_length] * sample_sum_perlen

        #sos_tokens = torch.zeros((sample_sum_perlen, 1)) + num_codebook_vectors #start
        sos_tokens = torch.zeros((sample_sum_perlen, 2)) + num_codebook_vectors #start
        sos_tokens[:, 1] = torch.tensor(lengths) + num_codebook_vectors #length prompt
        sos_tokens = sos_tokens.long().to("cuda")

        start_indices = torch.zeros((sample_sum_perlen, 0)).long().to("cuda")
        
        sample_indices = pbgpt.sample(start_indices,sos_tokens, steps=lengths[0],temperature=temperature,top_k=top_k)
        sample_backbone = pbgpt.z_to_backbone(sample_indices,lengths)#[b,len,n_dim * patch_size]
        for i in range(9 * patch_size):
            sample_backbone[:,:,i]=sample_backbone[:,:,i]*df['std'][i]+df['mean'][i]
            sample_backbone[:,:,i]=torch.clamp(sample_backbone[:,:,i],df['min'][i],df['max'][i])
        sample_backbone = sample_backbone.reshape(sample_sum_perlen,-1,9)
        sample_backbone = sample_backbone.cpu()

        rama_phi=np.hstack((rama_phi,sample_backbone[:,:,0].flatten()))
        rama_psi=np.hstack((rama_psi,sample_backbone[:,:,1].flatten()))
        for j in range(sample_sum_perlen):
            full_backbone=sample_backbone[j]
            full_backbone= np.hstack((np.insert(full_backbone[:,0:1],0,np.nan,axis=0),np.append(full_backbone[:,1:],np.array([[np.nan,np.nan,np.nan,np.nan,np.nan,0.0,0.0,0.0]]),axis=0)))
            full_backbone = pd.DataFrame(full_backbone, columns=column_names)
            PAT.create_new_chain_nerf("./Results/Sample/{dataset}/{len}_{id}.pdb".format(dataset=dataset,len=output_len+1,id=j),full_backbone)
            
            
parser = argparse.ArgumentParser()
#Datasets_config
parser.add_argument('--dataset', type=str, default='cath')
parser.add_argument('--maxResidueLen', type=int,default=128)
parser.add_argument('--patch_size', type=int,default=4)
#Codebook config
parser.add_argument('--num_codebook_vectors', type=int,default=50257)
#RecModel config
parser.add_argument('--RM_d_model', type=int, default=1024)
parser.add_argument('--RM_nhead', type=int, default=16)
parser.add_argument('--RM_dim_feedforward', type=int, default=4096)
parser.add_argument('--RM_nlayers', type=int, default=24)
#dropout
parser.add_argument('--dropout', type=float, default=0.)
#GPT config
parser.add_argument('--RM_model_path', type=str,default='./Models/cath_RecModel_29550.pt')
parser.add_argument('--pkeep', type=float, default=1.0)
parser.add_argument('--GPT_block_size', type=int, default=40)
parser.add_argument('--GPT_n_layers', type=int, default=24)
parser.add_argument('--GPT_n_heads', type=int, default=16)
parser.add_argument('--GPT_n_embd', type=int, default=1024)
#sample config
parser.add_argument('--gpus', type=str, default="1")
parser.add_argument('--top_k', type=int, default=129)
parser.add_argument('--temperature', type=float, default=0.99)
parser.add_argument('--sample_sum_perlen', type=int, default=10)
parser.add_argument('--output_len_l', type=int, default=50)
parser.add_argument('--output_len_r', type=int, default=128)
parser.add_argument('--model_name', type=str, default='cath_PBGPT_2600.pt')
args=parser.parse_args()

Sample(
        #Datasets_config
        maxResidueLen=args.maxResidueLen,
        patch_size=args.patch_size,
        dataset=args.dataset,
        #Codebook config
        num_codebook_vectors=args.num_codebook_vectors,
        #RecModel config
        RM_d_model=args.RM_d_model,
        RM_nhead=args.RM_nhead,
        RM_dim_feedforward=args.RM_dim_feedforward,
        RM_nlayers=args.RM_nlayers,
        #dropout
        dropout=args.dropout,
        #GPT config
        RM_model_path=args.RM_model_path,
        pkeep=args.pkeep,
        GPT_block_size=args.GPT_block_size,
        GPT_n_layers=args.GPT_n_layers,
        GPT_n_heads=args.GPT_n_heads,
        GPT_n_embd=args.GPT_n_embd,
        #sample config
        gpus=args.gpus,
        top_k=args.top_k,
        temperature=args.temperature,
        sample_sum_perlen=args.sample_sum_perlen,
        output_len_l=args.output_len_l,
        output_len_r=args.output_len_r,
        model_name=args.model_name
        )
            


