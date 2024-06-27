from language_models.metrics.metrics import check_valid, check_novelty_mol_path
import pandas as pd

if __name__ == "__main__":
    path_gen_csv = '/home/user/PycharmProjects/SmilesTuneMistral/language_models/llama3-8/result_llama_tune_v5_clean.csv'
    path_try_txt = "/home/user/PycharmProjects/SmilesTuneMistral/data/chembl_alpaca.txt"

    with open('/home/user/PycharmProjects/SmilesTuneMistral/language_models/llama3-8/result_llama_tune_v5_clean.txt') as f:
        lines = [line.rstrip() for line in f]
        pd.DataFrame(lines, columns=['output']).to_csv(path_gen_csv)

    check_valid(path_gen_csv)
    check_novelty_mol_path(path_try_txt, path_gen_csv, 'output', 'output')