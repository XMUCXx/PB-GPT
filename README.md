# PB-GPT - an Innovative GPT-Based Model for Protein Backbone Generation 

## File Hierarchy
|-- PB-GPT  
&emsp;|-- Datasets     
&emsp;&emsp;|-- download_cath.sh     
&emsp;&emsp;|-- cath_test.txt  
&emsp;&emsp;|-- cath_train.txt  
&emsp;&emsp;|-- ScalerInfo  
&emsp;&emsp;&emsp;|-- cath.csv  
&emsp;&emsp;&emsp;|-- rifdock.csv  
&emsp;|-- Models     
&emsp;&emsp;|-- cath_PBGPT_2600.pt     
&emsp;&emsp;|-- cath_RecModel_29550.pt  
&emsp;|-- utils     
&emsp;&emsp;|-- minGPT.py     
&emsp;&emsp;|-- nerf.py  
&emsp;&emsp;|-- Protein_Anlge_Trans.py  
&emsp;|-- models.py  
&emsp;|-- requirements.txt  
&emsp;|-- sample_cath.py      
&emsp;|-- README.md             

## Downloading CATH data

We provide a script in the `Datasets` dir to download requisite CATH data.

```bash
# Download the CATH dataset
cd Datasets  # Ensure that you are in the data subdirectory within the codebase
chmod +x download_cath.sh
./download_cath.sh
```

If the download link in the `.sh` file is not working, the tarball is also mirrored at the following [Dropbox link](https://www.dropbox.com/s/ka5m5lx58477qu6/cath-dataset-nonredundant-S40.pdb.tgz?dl=0).

## Downloading RIFDOCK data

the tarball is mirrored at the following [Link](http://files.ipd.uw.edu/pub/robust_de_novo_design_minibinders_2021/supplemental_files/scaffolds.tar.gz)

## Download the Pretrained Models

You can obtain the pretrained models from the provided links: 
- cath_PBGPT_2600.pt: https://china.scidb.cn/download?fileId=dde128020f53bcc62fa3ca06384e7936&username=1874832151@qq.com&traceId=1874832151@qq.com
- cath_RecModel_29550.pt: https://china.scidb.cn/download?fileId=905267fe05992ff1b1b96249b4b0d261&username=1874832151@qq.com&traceId=1874832151@qq.com

Once downloaded, move the .pt files to the Models directory within the PB-GPT project folder.

After moving the files, your Models directory should look like this:

Models     
├── cath_PBGPT_2600.pt     
└── cath_RecModel_29550.pt

## Environment

```bash
# Create a new environment using conda and activate the Environment
conda create --name PB-GPT python=3.8
conda activate PB-GPT
# Install Dependencies
pip install -r requirements.txt
```

## Sample protein backbones using PB-GPT
We utilize a pretrained model for protein backbone sampling with top-k=50 and temperature=1.0. The length of the protein backbones ranges from 50 to 128, and we generate 10 protein backbones for each length, totaling 790 protein backbones. These will be stored in the ./Results/Sample/cath/ directory.
After generating the protein backbones, the corresponding residue sequences can be predicted using ProteinMPNN, and then the protein structure corresponding to the sequences can be predicted using Omegafold. Finally, the designability of the generated backbones can be evaluated by calculating the TM score.

```bash
python sample_cath.py
```
The training code and testing details of the PB-GPT model will be supplemented after the publication of the paper.
## Train PB-GPT


