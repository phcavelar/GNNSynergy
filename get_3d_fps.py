#%%

#%%
import warnings
import os
import time
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm
from chemopy import ChemoPy
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem import rdDepictor
from rdkit.Chem import rdDistGeom
try:
    from rdkit.Chem.Draw import IPythonConsole
    IPythonConsole.ipython_3d = True
    import pymol3d
    has_3d_plotting = True
except ImportError as e:
    print(f"NO 3d: {e}")
    has_3d_plotting = False
import pubchempy as pcp

from chemopy import geo_opt
import subprocess
try:
    mol_smiles = "C1=CC=CC=C1"
    mol = Chem.AddHs(Chem.MolFromSmiles(mol_smiles))
    Chem.SanitizeMol(mol)
    if rdDistGeom.EmbedMolecule(mol) == -1:
        raise ValueError("Couldn't embed Benzene. This shouldn't happen.")
    dir_, dat_file = geo_opt.format_conversion(mol, "PM7", "2016")
    retcode = geo_opt.run_mopac(os.path.join(dir_.path, dat_file))
    if retcode:
        try:
            mopac_bin = geo_opt.MOPAC_CONFIG["2016"][0]
        except IndexError:
            mopac_bin = "mopac"
        retcode_again = subprocess.call(f'{mopac_bin} {os.path.join(dir_.path, dat_file)}', shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
        print(f"Retcode {retcode}... MOPAC seems to be working.")
except FileNotFoundError:
    # Hot-fix the run_mopac command to run with a shell, unsafe but this should be run only inside one's own computer
    run_mopac_bk = geo_opt.run_mopac
    def run_mopac(filename: str, version: str = '2016') -> int:
        """Run the MOPAC on a well-prepared input file.

        Parse default MOPAC config file if not read already.

        :param filename: path to the well-prepared MOPAC input file
        :param version: MOPAC version to be used
        """
        # Ensure all requirements are set
        if not geo_opt.is_mopac_version_available(version):
            raise ValueError(f'MOPAC version {version} is not available. Check your MOPAC config file.')
        # Run optimization
        mopac_bin = geo_opt.MOPAC_CONFIG[str(version)][0]
        try:
            retcode = subprocess.call(f'{mopac_bin} {filename}', shell=True,
                                    stdin=subprocess.DEVNULL,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)  # noqa: S603
            return retcode
        except Exception:
            return 1
    geo_opt.run_mopac = run_mopac


chemopy_calculator = ChemoPy(ignore_3D=False, include_fps=False, exclude_descriptors=False)

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
                mol_isosmiles = mol_cansmiles = "C[C@H]1CN(CCN1[C@@H](COC)C2=CC=C(C=C2)C(F)(F)F)C3(CCN(CC3)C(=O)C4=C(N=CN=C4C)C)C.C(=C\\C(=O)O)\\C(=O)O"
            else:
                try:
                    mol_pubchem = pcp.get_compounds(mol_id, 'name')[0]
                    mol_props = pcp.get_compounds(mol_id, 'name') mol_pubchem.cid)
                    mol_cansmiles = mol_pubchem.canonical_smiles
                    if not mol_cansmiles:
                        mol_props = pcp.get_properties("IsomericSMILES", mol_pubchem.cid)
                        mol_isosmiles = mol_pubchem.isomeric_smiles
                except IndexError:
                    try:
                        sub_pubchem = pcp.get_substances(mol_id, 'name')[0]
                        mol_pubchem = pcp.Compound.from_cid(sub_pubchem.cids[0])
                        mol_props = pcp.get_properties("CanonicalSMILES", mol_pubchem.cid)
                        mol_cansmiles = mol_pubchem.canonical_smiles
                        if not mol_cansmiles:
                            mol_props = pcp.get_properties("IsomericSMILES", mol_pubchem.cid)
                            mol_isosmiles = mol_pubchem.isomeric_smiles
                    except IndexError:
                        mol_cansmiles = ""
                        mol_isosmiles = ""
            mol_smiles = mol_cansmiles if mol_cansmiles else mol_isosmiles
        molids.append(mol_id)
        smiles.append(mol_smiles)

df = pd.DataFrame(data={"molids":molids,"smiles":smiles}).set_index("molids")
df.to_csv("smilesID.txt", header=False, sep="\t")
df.to_csv("smilesID.csv")

#%%
mol_list = []
conformer_attempts = 1000000 # This worked for Bleomycin Sulfate
counter = tqdm(zip(df.index,df.smiles), total=total_lines)
for mol_id, mol_smiles in counter:
    counter.set_description(mol_id)
    mol = Chem.AddHs(Chem.MolFromSmiles(mol_smiles))
    Chem.SanitizeMol(mol)
    while not (mol.GetNumConformers() and mol.GetConformer().Is3D()):
        if rdDistGeom.EmbedMolecule(mol, conformer_attempts)==-1:
            conformer_attempts*=10
    mol_list.append(mol)

# %%
try:
    feats3d = chemopy_calculator.calculate(mol_list[:1], show_banner=False, njobs=1)
except FileNotFoundError:
    from chemopy import geo_opt
    geo_opt.MOPAC_CONFIG = {'2016': ['mopac', ['PM7', 'PM6', 'PM3', 'AM1', 'MNDO']]}
    feats3d = chemopy_calculator.calculate(mol_list, show_banner=False)
feats3d.to_csv("3dfeats.txt", header=False, sep="\t")
feats3d.to_csv("3dfeats.csv")

#%%
