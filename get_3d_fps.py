import time
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm
from chemopy import ChemoPy
from rdkit import Chem
from rdkit.Chem import rdDistGeom
import pubchempy as pcp

chemopy_calculator = ChemoPy(ignore_3D=False, include_fps=False, exclude_descriptors=False)

total_lines = 0
with open("data/smilesID.txt") as f:
    for l in f:
        total_lines += 1

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
            molids.append(mol_id)
            smiles.append(mol_smiles)
            cansmiles.append("")
            isosmiles.append("")
        except ValueError:
            mol_id = l.strip()
            counter.set_description(f"{mol_id}")
            time.sleep(0.1)
            try:
                mol_pubchem = pcp.get_compounds(mol_id, 'name')[0]
                mol_props = pcp.get_properties("CanonicalSMILES", mol_pubchem.cid)
                mol_cansmiles = mol_pubchem.canonical_smiles
                mol_props = pcp.get_properties("IsomericSMILES", mol_pubchem.cid)
                mol_isosmiles = mol_pubchem.isomeric_smiles
            except IndexError:
                try:
                    sub_pubchem = pcp.get_substances(mol_id, 'name')[0]
                    mol_pubchem = pcp.Compound.from_cid(sub_pubchem.cids[0])
                    mol_props = pcp.get_properties("CanonicalSMILES", mol_pubchem.cid)
                    mol_cansmiles = mol_pubchem.canonical_smiles
                    mol_props = pcp.get_properties("IsomericSMILES", mol_pubchem.cid)
                    mol_isosmiles = mol_pubchem.isomeric_smiles
                except IndexError:
                    mol_cansmiles = ""
                    mol_isosmiles = ""
            molids.append(mol_id)
            smiles.append("")
            cansmiles.append(mol_cansmiles)
            isosmiles.append(mol_cansmiles)
        #mol = Chem.AddHs(Chem.MolFromSmiles(mol_smiles))
        #rdDistGeom.EmbedMolecule(mol)
        #molids.append(mol_id)
        #mols.append(mol)
    #counter = tqdm(zip(molids,mols),total=total_lines)
    #fps = chemopy_calculator.calculate(mols)
    #for mol_id, mol in mols:
        #print(mol, flush=True)
        
        
df = pd.DataFrame(data={"molids":molids,"smiles":smiles,"cansmiles":cansmiles,"isosmiles":isosmiles})
df.to_csv("aaa.csv")
