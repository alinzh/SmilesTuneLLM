import pandas as pd
from typing import Union
import itertools


class DataCollector:
    def __init__(self, path_to_csv: Union[str, bool] = False):
        """
        Parameters
        ----------
        path_to_csv : Union[str, bool], optional
            Path to csv file. Consist of next column:
            0, unobstructed, orthogonal_planes, h_bond_bridging, 0.1
            0 - index
            0.1 - is structure
        """
        if path_to_csv:
            self.data = pd.read_csv(path_to_csv, sep=',')
            self.res_dfrm = self.data.copy()
        self.conditions_true = {
            'unobstructed': 'The molecule has no obstructions (all parts are easily accessible).',
            'orthogonal_planes': 'Has orthogonal planes (parts of the molecule are oriented at right '
                                 'angles to each other).',
            'h_bond_bridging': 'Forms hydrogen bonds between its parts'
        }
        self.conditions_false = {
            'unobstructed': 'The molecule has obstructions (parts are not easily accessible).',
            'orthogonal_planes': 'Does not have orthogonal planes (parts of the molecule are not '
                                 'oriented at right angles to each other)',
            'h_bond_bridging': 'Does not form hydrogen bonds between its parts.'
        }
        self.start_of_sent = \
            ''


    def make_sentence(self, save_to: str) -> pd.DataFrame:
        """
        Make sentences by exist conditions.
        Save DataFrame to csv file.
        """
        sentences = []

        for row in self.data.values.tolist():
            promt = self.start_of_sent
            if row[1]:
                promt += self.conditions_true['unobstructed']
            else:
                promt += self.conditions_false['unobstructed']
            if row[2]:
                promt += self.conditions_true['orthogonal_planes']
            else:
                promt += self.conditions_false['orthogonal_planes']
            if row[3]:
                promt += self.conditions_true['h_bond_bridging']
            else:
                promt += self.conditions_false['h_bond_bridging']
            sentences.append(promt)

        self.res_dfrm['input'] = sentences
        self.res_dfrm.rename(columns={'0.1': 'output'}, inplace=True)
        self.res_dfrm.to_csv(save_to)

        return self.res_dfrm

    def add_smiles_token(self, dfrm: pd.DataFrame, path_to_save: Union[bool, str] = False, cut: Union[bool, int] = False) -> pd.DataFrame:
        structures = dfrm['output'].values.tolist()
        for idx, structure in enumerate(structures):
            structures[idx] = '<SMILES> ' + structure + ' </SMILES>'

        dfrm['output'] = structures

        if path_to_save:
            if type(cut) == int:
                dfrm = dfrm.sample(frac=1).reset_index(drop=True).head(cut).to_csv(path_to_save)
            else:
                dfrm = dfrm.sample(frac=1).reset_index(drop=True).to_csv(path_to_save)
        return dfrm

    def generate_combinations(self) -> list:
        """
        Generates all possible combinations of properties for promt

        Returns
        -------
        store : list
            All possible combinations of properties
        """
        all_conditions = []
        keys = list(self.conditions_true.keys())

        # Generate all possible combinations of True/False for the conditions
        for combination in itertools.product([True, False], repeat=len(keys)):
            condition_set = {}
            for key, is_true in zip(keys, combination):
                condition_set[key] = self.conditions_true[key] if is_true else self.conditions_false[key]
            all_conditions.append(condition_set)

        store = []
        for i, combination in enumerate(all_conditions):
            promt = self.start_of_sent
            for value in combination.values():
                promt += value
            store.append(promt)

        return store


if __name__ == "__main__":
    path = '/home/user/PycharmProjects/SmilesTuneMistral/data/database_ChEMBL_performace.csv'
    save_to = '/home/user/PycharmProjects/SmilesTuneMistral/data/props_in_sentences_ChEMBL.csv'
    dc = DataCollector(path)
    combinations = dc.make_sentence(save_to)

