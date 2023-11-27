#!/usr/bin/env python3
from __future__ import annotations
import numpy as np

from scipy.sparse import lil_matrix
from random import randint

# Jaccard score of the sets c1 and c2
def jaccard(c1, c2):
    return len(c1.intersection(c2)) / len(c1.union(c2))

def custom_hash(x, a, b):
    # Realistically, shingle size is not going to be much larger than 10,
    # so we can afford to pick some static prime here
    p = 379

    return a + b * x % p


class Item():
    """
    Simple class to make working with items a bit easier
    """

    def __init__(self, model_id: str, features: dict[str, list[str]], shop: str, title: str):
        self.id = model_id

        self.shop = shop
        self.title = title

        self.signature: Signature = None

        self.features = features


    def __str__(self) -> str:
        return f"Item: '{self.id}'"

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self) -> int:
        return hash(self.title + self.id)

    def __eq__(self, other) -> bool:
        return self.id == other.id


    def make_shingle_string(self) -> str:
        return self.title.replace(" ", "")


    def shingle(self, shingle_size: int) -> None:
        string = self.make_shingle_string()

        self.shingled_data = set()

        for i in range(len(string) - shingle_size):
            self.shingled_data.add(string[i:i+shingle_size])



    # Class methods below here
    def minhash(products: list[Item]) -> lil_matrix:
        shingles = [product.shingled_data for product in products]

        all_shingles = set()
        for shingle in shingles:
            all_shingles.update(shingle)

        # Premature optimisation btw (^:
        result = lil_matrix((len(all_shingles), len(products)))

        for i, feature in enumerate(all_shingles):
            for j, shingle in enumerate(shingles):
                if feature in shingle:
                    result[i, j] = True

        return result


    def binary_to_signatures(binary_data: lil_matrix, num_hashes: int) -> np.ndarray:
        result = np.full([num_hashes, binary_data.shape[1]], float("inf"))

        # https://stackoverflow.com/questions/4319014/iterating-through-a-scipy-sparse-vector-or-matrix
        rows, cols = binary_data.nonzero()
        length = len(list(zip(rows, cols)))
        for i, (row, col) in enumerate(zip(rows, cols)):
            print(f"{i} ({i / length:.1%})", end = "\r")
            for i in range(num_hashes):
                a = randint(0, 100_000)
                b = randint(0, 100_000)

                # nonzero() already filters zero values in the algorithm
                value = binary_data[row, col]
                result[i, col] = min(custom_hash(value, a, b), result[i, col])

        return result


class Signature():
    def __init__(self, signature: np.ndarray):
        self.value = signature

        self.hashed: list[int] = None

    # Hash of each band of the signature
    def hash(self, num_bands, num_rows) -> list[int]:
        if self.hashed is not None:
            return self.hashed

        else:
            self.hashed = []

            for i in range(num_bands):
                band = self.value[i * num_rows:(i + 1) * num_rows]
                self.hashed.append(sum([hash(i) for i in band]))

            return self.hashed

    def __str__(self) -> str:
        return f"Signature: {str(self.value):.10s}..."

    def __repr__(self) -> str:
        return self.__str__()

