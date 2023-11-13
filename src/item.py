#!/usr/bin/env python3
"""
Simple class to make working with items a bit easier
"""


class Item():
    def __init__(self, id, features, all_features):
        self.id = id

        self.features = {}

        for feature in all_features:
            self.features[feature] = features.get(feature, None)


    def __str__(self):
        return f"Item: '{self.id}', {self.features}"

    def __repr__(self):
        return self.__str__()

if __name__ == "__main__":
    a = Item("hoi", {"a": 5, "b": 6})
    b = Item("doei", {"c": "d", "b": 6})

    print(a)
    print(b)
