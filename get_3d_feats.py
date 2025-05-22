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

def check_mopac_installation():
    from chemopy import geo_opt
    import subprocess
    try:
        mol_smiles = "C1=CC=CC=C1"
        mol = Chem.AddHs(Chem.MolFromSmiles(mol_smiles))
        Chem.SanitizeMol(mol)
        if rdDistGeom.EmbedMolecule(mol) == -1:
            raise ValueError("Couldn't embed Benzene. This shouldn't happen.")
        dir_, dat_file = geo_opt.format_conversion(mol, "PM7", "2016")
        fpath = os.path.join(dir_.path, dat_file)
        retcode = geo_opt.run_mopac(fpath)
        if retcode:
            try:
                mopac_bin = geo_opt.MOPAC_CONFIG["2016"][0]
            except IndexError:
                mopac_bin = "mopac"
                retcode_again = subprocess.call(f'{mopac_bin} {fpath}', shell=False,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL)
                print(f"Manual call MOPAC seems to be working (retcode={retcode}).")
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
        if "2016" not in geo_opt.MOPAC_CONFIG or any([opt not in geo_opt.MOPAC_CONFIG["2016"][1] for opt in ['PM7', 'PM6', 'PM3', 'AM1', 'MNDO']]):
            # Rest the MOPAC CONFIG
            geo_opt.MOPAC_CONFIG = {'2016': ['mopac', ['PM7', 'PM6', 'PM3', 'AM1', 'MNDO']]}


def main(
        target_fpath = "data/smilesID.txt",
        out_fpath = "3dfeats",
        conformer_attempts:int = 1000000,
        n_jobs = 1,
        ):
    chemopy_calculator = ChemoPy(ignore_3D=False, include_fps=False, exclude_descriptors=False)

    #%%
    if target_fpath.endswith(".csv"):
        df = pd.read_csv("smilesID.txt", header=False, sep="\t", index_col=0)
    else:
        df = pd.read_csv("smilesID.csv", index_col=0)

    #%%
    mol_list = []
    counter = tqdm(zip(df.index,df.smiles), total=df.shape[0])
    for mol_id, mol_smiles in counter:
        counter.set_description(mol_id)
        mol = Chem.AddHs(Chem.MolFromSmiles(mol_smiles))
        Chem.SanitizeMol(mol)
        while not (mol.GetNumConformers() and mol.GetConformer().Is3D()):
            if rdDistGeom.EmbedMolecule(mol, conformer_attempts)==-1:
                conformer_attempts*=10
        mol_list.append(mol)

    # %%
    feats3d = chemopy_calculator.calculate(mol_list, show_banner=False, njobs=n_jobs)
    feats3d.to_csv(f"{out_fpath}.txt", header=False, sep="\t")
    feats3d.to_csv(f"{out_fpath}.csv")

#%%

