import pandas as pd


class DataCollector:
    def __init__(self, path_to_csv: str):
        """
        Parameters
        ----------
        path_to_csv : str
            Path to csv file. Consist of next column:
            0, unobstructed, orthogonal_planes, h_bond_bridging, 0.1
            0 - index
            0.1 - is structure
        """
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

    def make_sentence(self, save_to: str) -> pd.DataFrame:
        """
        Make sentences by exist conditions.
        Save DataFrame to csv file.
        """
        sentences = []
        start_of_sent = 'Give me a molecule that satisfies the conditions outlined in the description: '

        for row in self.data.values.tolist():
            promt = start_of_sent
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

        self.res_dfrm['sentence'] = sentences
        self.res_dfrm.rename(columns={'0.1': 'structure'}, inplace=True)
        self.res_dfrm.to_csv(save_to)

        return self.res_dfrm


if __name__ == "__main__":
    path = '/home/user/PycharmProjects/SmilesTuneMistral/data/database_ChEMBL_performace.csv'
    save_to = '/home/user/PycharmProjects/SmilesTuneMistral/data/props_in_sentences.csv'
    dc = DataCollector(path)
    dc.make_sentence(save_to)

