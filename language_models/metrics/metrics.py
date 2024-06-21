from rdkit import Chem
import pandas as pd


def check_valid(path_to_generated_mols_csv: str) -> int:
    """
    Checks how many valid examples there are (in percentage)
    """
    generated_coformers_clear = []
    generated_coformers = list(pd.read_csv(path_to_generated_mols_csv)['0'])

    for smiles_mol in generated_coformers:
        if Chem.MolFromSmiles(str(smiles_mol)) is None:
                        continue
        generated_coformers_clear.append(smiles_mol)

    valid_in_pr = len(generated_coformers_clear) / len(generated_coformers) * 100
    print(f'Valid percentage: {valid_in_pr}')

    return valid_in_pr


def check_novelty_mol_path(
        train_dataset_path: str,
        gen_data: str,
        train_col_name: str,
        gen_col_name: str,
    ):
    """
    Function for count how many new molecules generated compared with train data

    Parameters
    ----------
    ids : list
        Path to csv train dataset
    gen_data : list
        Path to preds in csv format
    train_col_name : str
        Name of column that consist a molecule strings
    gen_col_name: str
        Name of column that consist a molecule strings
     
    Returns
    -------
    novelty : float
        Percentage of new molecules
    duplicates: float
        Share of duplicates
    """
    train_d = pd.read_csv(train_dataset_path)[train_col_name]
    gen_d = pd.read_csv(gen_data)
    duplicates = gen_d.duplicated(subset=gen_col_name, keep='first').sum()/len(gen_d)

    novelty =( len(gen_d[gen_col_name].drop_duplicates())-gen_d[gen_col_name].drop_duplicates().isin(train_d).sum() )/ len(train_d) * 100
    print('Generated molecules consist of',novelty, '% unique new examples',
          '\t',
          f'duplicates: {duplicates}')
    return novelty, duplicates



if __name__ == "__main__":
    path_gen = '/home/user/projects/GEMCODE/language_models/gpt2/out.csv'
    path_tr = '/home/user/projects/GEMCODE/language_models/gpt2/data/gpt_train.csv'

    check_valid(path_gen)
    check_novelty_mol_path(path_tr, path_gen, '0', '0')

