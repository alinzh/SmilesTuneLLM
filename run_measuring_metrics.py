from language_models.metrics.metrics import check_valid, check_novelty_mol_path
import pandas as pd

import os
import sys
sys.path.append(os.getcwd())


def main(
    path_try_txt = 'xLSTM/chembl_alpaca.txt', path_gen_csv = 'xLSTM/examples/generated_mols.csv',
    path_gen_txt = 'xLSTM/examples/generated_mols.txt'
    ):

    with open(path_gen_txt) as f:
        lines = [line.rstrip() for line in f]
        pd.DataFrame(lines, columns=['output']).to_csv(path_gen_csv)

    valid = check_valid(path_gen_csv)
    novelty, duplicates = check_novelty_mol_path(path_try_txt, path_gen_csv, 'output', 'output')
    
    return valid, novelty, duplicates


if __name__ == "__main__":
    path_gen_csv = '/data/alina_files/projects/2/SmilesTuneLLM/xLSTM/examples/generated_mols.csv'
    path_try_txt = '/home/alina/data/projects/2/SmilesTuneLLM/xLSTM/chembl_alpaca.txt'

    with open('/data/alina_files/projects/2/SmilesTuneLLM/xLSTM/examples/generated_mols.txt') as f:
        lines = [line.rstrip() for line in f]
        pd.DataFrame(lines, columns=['output']).to_csv(path_gen_csv)

    check_valid(path_gen_csv)
    check_novelty_mol_path(path_try_txt, path_gen_csv, 'output', 'output')