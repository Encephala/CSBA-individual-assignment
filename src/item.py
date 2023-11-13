#!/usr/bin/env python3
"""
Simple class to make working with items a bit easier
"""


class Item():
    def __init__(self, id, features, all_features, shop, title):
        self.id = id

        self.shop = shop
        self.title = title

        self.features = {}

        for feature in all_features:
            self.features[feature] = features.get(feature, None)


    def __str__(self):
        return f"Item: '{self.id}', {self.features}"

    def __repr__(self):
        return self.__str__()

