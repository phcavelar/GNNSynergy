import pandas as pd

def main(
        smiles_path = "smilesID.csv",
        out_fpath = "input/batch{}.csv",
        smiles_per_batch:int = 1,
        n_jobs = 1,
        ):
    #%%
    if smiles_path.endswith(".csv"):
        df = pd.read_csv(smiles_path, index_col=0)
    else:
        df = pd.read_csv(smiles_path, header=False, sep="\t", index_col=0)

    for bid, i in enumerate(range(0,df.shape[0],smiles_per_batch)):
        batch_slice = df.iloc[i:i+smiles_per_batch]
        batch_slice.to_csv(out_fpath.format(bid))

#%%

if __name__=="__main__":
    import fire
    fire.Fire(main)
