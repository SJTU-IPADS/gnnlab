from enum import Enum


class App(Enum):
    gcn = 0
    graphsage = 1
    pinsage = 2

    def __str__(self):
        return self.name


class Dataset(Enum):
    products = 0
    papers100M = 1
    uk_2006_05 = 2
    twitter = 3

    def __str__(self):
        if self is Dataset.uk_2006_05:
            return 'uk-2006-05'
        return self.name
