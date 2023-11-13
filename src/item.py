#!/usr/bin/env python3
"""
Simple class to make working with items a bit easier
"""

from __future__ import annotations
from scipy.sparse import lil_matrix

# Jaccard score of the sets c1 and c2
def jaccard(c1, c2):
    return len(c1.intersection(c2)) / len(c1.union(c2))


class Item():
    def __init__(self, id, features, all_features, shop, title):
        self.id = id

        self.shop = shop
        self.title = title

        self.features = {}

        for feature in all_features:
            self.features[feature] = features.get(feature, None)



    def __str__(self):
        return f"Item: '{self.id}'"

    def __repr__(self):
        return self.__str__()

    def make_shingle_string(self):
        result = ""
        for feature, value in self.features.items():
            if value is not None:
                # Make binary data more processable
                if value in ["Yes", "True"]:
                    result += feature

                elif value in ["No", "False"]:
                    continue

                else:
                    result += value.replace(" ", "")

        result += self.title

        self.shinglable_string = result

        return result


    def shingle(self, shingle_size):
        string = self.make_shingle_string()

        self.shingled_data = set()

        for i in range(len(string) - shingle_size):
            self.shingled_data.add(string[i:i+shingle_size])



    def minhash(products: list[Item]):
        shingles = [product.shingled_data for product in products]

        all_shingles = set()
        for shingle in shingles:
            all_shingles.update(shingle)

        result = lil_matrix((len(all_shingles), len(products)))

        for i, feature in enumerate(all_shingles):
            for j, shingle in enumerate(shingles):
                pass




if __name__ == "__main__":
    print(jaccard(set([1, 2, 3]), set([2, 3, 4])))
