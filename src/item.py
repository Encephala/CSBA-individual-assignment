#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
import re

from scipy.sparse import lil_matrix
from random import randint

# Jaccard score of the sets c1 and c2
def jaccard(c1, c2):
    return len(c1.intersection(c2)) / len(c1.union(c2))

def parse_float(string: str) -> list[float]:
    occurrences = re.findall(r"\d+\.[\d+]*", string)
    return [float(i) for i in occurrences]

def custom_hash(x, a, b):
    # Realistically, shingle size is not going to be much larger than 10,
    # so we can afford to pick some static prime here
    p = 379

    return a + b * x % p


class Item():
    """
    Simple class to make working with items a bit easier
    """

    def __init__(self, model_id: str, features: dict[str, str], shop: str, title: str):
        self.id = model_id

        self.shop = shop
        self.title = title

        self.signature: Signature = None

        self.features = features

        self.weight = f"Weight {self.get_weight()}"
        self.diagonal = self.get_diagonal()


    def __str__(self) -> str:
        return f"Item: '{self.id}'"

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self) -> int:
        return hash(self.title + self.id)

    def __eq__(self, other) -> bool:
        return self.id == other.id


    def get_weight(self) -> float:
        all_weights_found = []

        for value in self.features.values():
            if "lb" in value:
                all_weights_found.append(parse_float(value))

        if len(all_weights_found) == 0:
            return None

        # Return smallest of all weights, as that is most reliable
        # (doesn't include stand (hopefully))
        return min(all_weights_found)


    def get_diagonal(self) -> float:
        ...


    def make_shingle_string(self) -> str:
        return self.title.replace(" ", "")


    def find_set_representation(self, shingle_size: int) -> None:
        result = set()

        string = self.make_shingle_string()

        for i in range(len(string) - shingle_size):
            result.add(string[i:i+shingle_size])

        result.add(self.weight_quantile)
        result.add(self.diagonal_quantile)

        self.shingled_data = result


    # Class methods below here
    def calc_quantiles(products: list[Item]) -> None:
        weights = [product.weight for product in products]
        diagonals = [product.diagonal for product in products]

        weight_quantiles = np.quantile(weights, [0.33, 0.67])
        diagonal_quantiles = np.quantile(diagonals, [0.33, 0.67])

        for product in products:
            product.weight_quantile = np.searchsorted(weight_quantiles, product.weight) if product.weight is not None else -1

            product.diagonal_quantile = np.searchsorted(diagonal_quantiles, product.diagonal) if product.diagonal is not None else -1


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
            for h in range(num_hashes):
                a = randint(0, 100_000)
                b = randint(0, 100_000)

                # nonzero() already filters zero values in the algorithm
                if binary_data[row, col] == 1:
                    result[h, col] = min(custom_hash(row, a, b), result[h, col])

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

