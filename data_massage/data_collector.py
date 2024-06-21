import pandas as pd


class DataCollector:
    def __init__(self, path_to_csv: str):
        self.data = pd.read_csv(path_to_csv)

    def complite_sentence(self):
        print(self.data)
        print()


if __name__ == "__main__":
    path = '/home/user/PycharmProjects/SmilesTuneMistral/data/cocrys_alpaca.txt'
    dc = DataCollector(path)
    dc.complite_sentence()
    print(1)

