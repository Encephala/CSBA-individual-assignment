#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
import re

from scipy.sparse import lil_matrix
from random import randint

# Jaccard score of the sets c1 and c2
def jaccard(c1, c2):
    return len(c1.intersection(c2)) / len(c1.union(c2))

def parse_numbers(string: str) -> list[float]:
    # Matches 1.2, 1., 1 and .2
    # Will have some false positives but I don't get how python re
    # works with groups so ykno it's fine
    occurrences = re.findall(r"\.?\d+\.?\d*", string)
    return [float(i) for i in occurrences]

def custom_hash(x, a, b):
    # Realistically, shingle size is not going to be much larger than 10,
    # so we can afford to pick some static prime here
    p = 379

    return a + b * x % p


def replace_all(string: str, keys: list[str], to: str) -> str:
    for key in keys:
        string = string.replace(key, to)

    return string


class Item():
    """
    Simple class to make working with items a bit easier
    """

    def __init__(self, model_id: str, features: dict[str, str], shop: str, title: str):
        self.id = model_id

        self.shop = shop.lower()

        self.title = replace_all(
                        replace_all(title, ['"', " inch", "-inch", "inches", "Inch", "-Inch", " Inch"], "inch"),
                        ["hertz", "hz", " hz", "Hertz", "Hz", "Hz"], "hz").lower()

        self.signature: Signature = None

        self.features = {
            key.lower().replace(":", ""):
                replace_all(
                    replace_all(val, ['"', " inch", "-inch", "inches", "Inch", "-Inch", " Inch"], "inch"),
                    ["hertz", "hz", " hz", "Hertz", "Hz", "Hz"], "hz").lower()

            for key, val in features.items()
        }

        self.weight = self.get_weight()
        self.diagonal = self.get_diagonal()
        self.refresh_rate = self.get_refresh_rate()
        self.brand = self.get_brand()

    def __str__(self) -> str:
        return f"Item: '{self.id}'"

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self) -> int:
        return hash(self.title + self.shop)

    def __eq__(self, other) -> bool:
        return self.id == other.id


    def get_weight(self) -> float:
        shop_map = {
            "bestbuy.com": ["weight", "product weight"],
            "newegg.com": ["weight without stand"],
            "amazon.com": ["item Weight", "product dimensions", "shipping weight"], # If Product Dimensions, have to split(";")[1]
            "thenerds.net": ["weight (approximate)"],
        }

        for feature_name in shop_map[self.shop]:
            if feature_name in self.features:
                if feature_name != "Product Dimensions":
                    # Min should be fine, since we're only comparing quantiles
                    return min(parse_numbers(self.features[feature_name]))

                else:
                    return min(parse_numbers(self.features[feature_name].split(";")[1]))


        return None

    def get_refresh_rate(self) -> float:
        match_string = r"\d+\s?[Hh][Zz]"

        for value in self.features.values():
            if re.search(match_string, value):
                # Min should be fine, since we're only comparing quantiles
                return min(parse_numbers(re.search(match_string, value).string))

        return None

    def get_diagonal(self) -> float:
        shop_map = {
            "bestbuy.com": ["screen size class", "screen size (measured diagonally)"],
            "newegg.com": ["screen size"],
            "amazon.com": ["display size"],
            "thenerds.net": ["screen size"],
        }

        for feature_name in shop_map[self.shop]:
            if feature_name in self.features:
                # Min should be fine, since we're only comparing quantiles
                return min(parse_numbers(self.features[feature_name]))

        return None

    def get_brand(self) -> str:
        shop_map = {
            "bestbuy.com": "brand",
            "newegg.com": "brand name",
            "amazon.com": "brand",
            "thenerds.net": "brand name",
        }

        if shop_map[self.shop] in self.features:
            return self.features[shop_map[self.shop]].lower()

        return None


    def make_shingle_string(self) -> str:
        return self.title.replace(" ", "")


    def find_set_representation(self, shingle_size: int, max_len: int = 10) -> None:
        result = set()

        # for word in self.title.split(" "):
        #     # result.add(word)
        #     if len(word) < shingle_size:
        #         result.add(word)

        #     # for i in range(len(word) - shingle_size + 1):
        #     #     result.add(word[i:i + shingle_size])

        # # title_simple = self.make_shingle_string()

        # # for i in range(len(title_simple) - shingle_size + 1):
        # #     result.add(title_simple[i:i + shingle_size])

        # for val in self.features.values():
        #     # Don't include features that are too long, they won't ever match anyways
        #     if len(val) < max_len:
        #         result.add(val.replace(" ", "").lower())


        regex_title = r"([a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-zA-Z0-9]*)"
        regex_values = r"(\d+(\.\d+)?[a-zA-Z]+|^\d+(\.\d+)?)"

        for match in re.finditer(regex_title, self.title):
            result.add(match.group().strip())

        for val in self.features.values():
            for match in re.finditer(regex_values, val):
                result.add(match.group().strip())

            for match in re.finditer(regex_title, val):
                result.add(match.group().strip())

        if self.weight_quantile is not None:
            result.add(f"Weight {self.weight_quantile}")

        if self.diagonal_quantile is not None:
            result.add(f"Diagonal {self.diagonal_quantile}")

        if self.refresh_rate is not None:
            result.add(f"Refresh Rate {self.refresh_rate}")

        if self.brand is not None:
            result.add(f"Brand {self.brand}")

        self.set_representation = result


    # Class methods below here

    # Calculates quantiles for weight and diagonal size
    def calc_quantiles(products: list[Item]) -> None:
        weights = [product.weight for product in products if product.weight is not None]
        diagonals = [product.diagonal for product in products if product.diagonal is not None]

        weight_quantiles = np.quantile(weights, [0.1, 0.3, 0.5, 0.7, 0.9])
        diagonal_quantiles = np.quantile(diagonals, [0.1, 0.3, 0.5, 0.7, 0.9])

        for product in products:
            product.weight_quantile = np.searchsorted(weight_quantiles, product.weight) if product.weight is not None else None

            product.diagonal_quantile = np.searchsorted(diagonal_quantiles, product.diagonal) if product.diagonal is not None else None


    def minhash(products: list[Item]) -> lil_matrix:
        representations = [product.set_representation for product in products]

        all_componenents = set()
        for representation in representations:
            all_componenents.update(representation)

        # Premature optimisation btw (^:
        result = lil_matrix((len(all_componenents), len(products)))

        for i, feature in enumerate(all_componenents):
            for j, representation in enumerate(representations):
                if feature in representation:
                    # Can be any value, just have to create the entry
                    result[i, j] = True

        print(f"Binary matrix size: {result.shape}")

        return result


    def binary_to_signatures(binary_data: lil_matrix, num_hashes: int) -> np.ndarray:
        result = np.full([num_hashes, binary_data.shape[1]], float("inf"))

        # https://stackoverflow.com/questions/4319014/iterating-through-a-scipy-sparse-vector-or-matrix
        rows, cols = binary_data.nonzero()

        for h in range(num_hashes):
            print(f"{h} ({h / num_hashes:.1%})", end = "\r")

            a = randint(0, 100_000)
            b = randint(0, 100_000)

            for row, col in zip(rows, cols):
                # nonzero() already filters zero values
                result[h, col] = min(custom_hash(row, a, b), result[h, col])

        return result


class Signature():
    def __init__(self, signature: np.ndarray):
        self.value = signature

    # Hash of each band of the signature
    def hashes(self, num_bands, num_rows) -> list[int]:
        result = []

        for i in range(num_bands):
            band = self.value[i * num_rows:(i + 1) * num_rows]
            result.append(sum([hash(i) for i in band]))

        return result

    def __str__(self) -> str:
        return f"Signature: {str(self.value):.40s}..."

    def __repr__(self) -> str:
        return self.__str__()

