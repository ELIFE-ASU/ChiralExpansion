# Tracking chirality through Network Expansion

# Installation Instructions

Any environment manager can be used (these installation instructions assume [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) is pre-installed). Any stable python3 version can be used, 3.11 is listed for convenience. The required pip installed packages are listed in the provided requirements.txt file.

```
conda create -n myVenv python=3.11 -y
conda activate myVenv
pip install -r requirements.txt
```


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
