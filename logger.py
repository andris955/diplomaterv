import os
import pandas as pd

class Logger:
    def __init__(self, id, listofgames):
        self.id = id
        self.listofgames = listofgames
        self.folder_path = os.path.join("./data/logging/", self.id)
        if not os.path.exists(self.folder_path):
            os.mkdir(self.folder_path)
        self.fields = None
        self.data = None
        self.pd_data = None

        self.__pd_data_init()

    def __save_path(self, game):
        return os.path.join(self.folder_path, game + ".csv")


    def __pd_data_init(self):
        self.data = {}
        self.pd_data = {}
        for game in self.listofgames:
            self.pd_data[game] = None
            self.data[game] = {}

    def __make_pd_data(self):
        for game in self.listofgames:
            if self.pd_data[game] is None:
                self.pd_data[game] = pd.DataFrame(self.data[game])
            else:
                print("Hiba")

    def log(self, game, values):
        try:
            if self.fields is None:
                self.fields = values._fields
            elif self.fields != values._fields:
                print("The fields of the named tuples must be consistent")

            if self.data[game] == {}:
                for field in self.fields:
                    self.data[game][field] = []

            for i, field in enumerate(self.fields):
                self.data[game][field].append(values[i])

        except:
            print("Values must be a namedtuple")


    def dump(self):
        self.__make_pd_data()
        for game in self.listofgames:
            if os.path.exists(self.__save_path(game)):
                with open(self.__save_path(game), 'a') as f:
                    self.pd_data[game].to_csv(f, sep=";", index=False, header=False)
            else:
                with open(self.__save_path(game), 'a') as f:
                    self.pd_data[game].to_csv(f, sep=";", index=False)
        self.__pd_data_init()


