# Tracking chirality through Network Expansion


# Data pipeline
KEGG     <-> lets_get_kegg.py <-> mol_to_inchi.py  <-> calc_chirality_kegg.py 

raw data <-> .mol files       <-> inchi .csv files <-> ...
 
## KEGG
Use the code here to obtain the raw data files from KEGG: https://github.com/ELIFE-ASU/lets-get-kegg

Run lets_get_kegg.py

Run mol_to_inchi.py



## GDB
Get the data from here: https://gdb.unibe.ch/downloads/

Get the following files and uncompress and put in relevant files: GDB17.50000000.smi.gz (314 MB), gdb13.tgz (2.6 GB), gdb11.tgz (122 MB)

Run calc_chirality_gdb.py


## Required packages
pandas, chemparse, numpy, rdkit, matplotlib, scipy, networkx, seaborn, tqdm, pickle, csv, json

conda install pandas numpy matplotlib scipy

pip install --user --upgrade chemparse networkx[default] seaborn tqdm pickle csv json

conda install -c conda-forge rdkit
