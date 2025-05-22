#%%

#%%
import os
import time
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm
import pubchempy as pcp

#%%

total_lines = 0
with open("data/smilesID.txt") as f:
    for l in f:
        total_lines += 1

#%%

with open("data/smilesID.txt") as f:
    mols = []
    molids = []
    smiles = []
    cansmiles = []
    isosmiles = []
    fps = []
    counter = tqdm(f,total=total_lines)
    for l in counter:
        try:
            mol_id, mol_smiles = l.strip().split("\t")
            counter.set_description(f"{mol_id}")
        except ValueError:
            mol_id = l.strip()
            counter.set_description(f"{mol_id}")
            time.sleep(0.1)
            if mol_id == "DSSTox_CID_28582":
                mol_cansmiles = "C[C@H]1CN(CCN1[C@@H](COC)C2=CC=C(C=C2)C(F)(F)F)C3(CCN(CC3)C(=O)C4=C(N=CN=C4C)C)C.C(=C\\C(=O)O)\\C(=O)O"
            else:
                try:
                    mol_pubchem = pcp.get_compounds(mol_id, 'name')[0]
                    mol_props = pcp.get_properties("CanonicalSMILES", mol_pubchem.cid)
                    mol_cansmiles = mol_pubchem.canonical_smiles
                    if not mol_cansmiles:
                        mol_props = pcp.get_properties("IsomericSMILES", mol_pubchem.cid)
                        mol_cansmiles = mol_pubchem.isomeric_smiles
                except IndexError:
                    try:
                        sub_pubchem = pcp.get_substances(mol_id, 'name')[0]
                        mol_pubchem = pcp.Compound.from_cid(sub_pubchem.cids[0])
                        mol_props = pcp.get_properties("CanonicalSMILES", mol_pubchem.cid)
                        mol_cansmiles = mol_pubchem.canonical_smiles
                        if not mol_cansmiles:
                            mol_props = pcp.get_properties("IsomericSMILES", mol_pubchem.cid)
                            mol_cansmiles = mol_pubchem.isomeric_smiles
                    except IndexError:
                        mol_cansmiles = ""
            mol_smiles = mol_cansmiles
        molids.append(mol_id)
        smiles.append(mol_smiles)

df = pd.DataFrame(data={"molids":molids,"smiles":smiles}).set_index("molids")
df.to_csv("smilesID.txt", header=False, sep="\t")
df.to_csv("smilesID.csv")