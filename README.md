# PB-GPT - an Innovative GPT-Based Model for Protein Backbone Generation 

## File Hierarchy
|-- PB-GPT  
&emsp;|-- Datasets     
&emsp;&emsp;|-- download_cath.sh     
&emsp;&emsp;|-- cath_test.txt  
&emsp;&emsp;|-- cath_train.txt     
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

