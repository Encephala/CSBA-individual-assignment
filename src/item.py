#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
import re

from scipy.sparse import lil_matrix
from random import randint
from collections import defaultdict


def parse_numbers(string: str) -> list[float]:
    # Matches 1.2, 1., 1 and .2
    # Will have some false positives but I don't get how python re
    # works with groups so ykno it's fine
    occurrences = re.findall(r"\.?\d+\.?\d*", string)
    return [float(i) for i in occurrences]

def custom_hash(x, a, b):
    # Some large prime
    p = 7840153

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
                        ["hertz", "hz", " hz", "Hertz", "Hz", "Hz"], "hz").lower().strip()

        self.features = {
            key.lower().replace(":", ""):
                replace_all(
                    replace_all(val, ['"', " inch", "-inch", "inches", "Inch", "-Inch", " Inch"], "inch"),
                    ["hertz", "hz", " hz", "Hertz", "Hz", "Hz"], "hz").lower().strip()

            for key, val in features.items()
        }

        self.weight = self.get_weight()
        self.diagonal = self.get_diagonal()
        self.refresh_rate = self.get_refresh_rate()
        self.brand = self.get_brand()

    def __str__(self) -> str:
        return f"Item: '{self.id}' ('{self.shop}')"

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self) -> int:
        return hash(self.title) + hash(self.shop) + hash(self.id)

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


    def find_set_representation(self, max_len: int = 10) -> None:
        result = set()

        # for word in self.title.split(" "):
        #     result.add(word)

        # for val in self.features.values():
        #     if len(val) < max_len:
        #         result.add(val)

        regex_words = r"([a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-zA-Z0-9]*)"
        regex_number_unit = r"(\d+(\.\d+)?[a-zA-Z]+|^\d+(\.\d+)?)"

        for match in re.finditer(regex_words, self.title):
            result.add(match.group().strip())

        for val in self.features.values():
            for match in re.finditer(regex_number_unit, val):
                result.add(match.group().strip())

            # for match in re.finditer(regex_words, val):
            #     result.add(match.group().strip())

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
    @staticmethod
    def calc_quantiles(products: list[Item]) -> None:
        weights = [product.weight for product in products if product.weight is not None]
        diagonals = [product.diagonal for product in products if product.diagonal is not None]

        weight_quantiles = np.quantile(weights, [0.1, 0.3, 0.5, 0.7, 0.9])
        diagonal_quantiles = np.quantile(diagonals, [0.1, 0.3, 0.5, 0.7, 0.9])

        for product in products:
            product.weight_quantile = np.searchsorted(weight_quantiles, product.weight) if product.weight is not None else None

            product.diagonal_quantile = np.searchsorted(diagonal_quantiles, product.diagonal) if product.diagonal is not None else None


    @staticmethod
    def minhash(products: list[Item], do_print = True) -> lil_matrix:
        representations = [product.set_representation for product in products]

        # Set to ensure uniqueness
        all_componenents = set()
        for representation in representations:
            all_componenents.update(representation)

        # List to have a well-defined order
        all_components_list = list(all_componenents)

        result = lil_matrix((len(all_components_list), len(products)))

        # Loop over the list to maintain ordering
        for i, feature in enumerate(all_components_list):
            for j, representation in enumerate(representations):
                if feature in representation:
                    # Can be any value, just have to create the entry
                    result[i, j] = True


        # Clean up the data, removing components which only appear for a single product
        # or which occur many times
        rows, _ = result.nonzero()

        occurrences = defaultdict(int)

        for row in rows:
            occurrences[row] += 1

        # single_occurrences = dict(filter(lambda x: x[1] == 1, occurrences.items()))
        many_occurrences = dict(filter(lambda x: x[1] > 400, occurrences.items()))

        all_filtered_features = {**many_occurrences}

        for row in sorted(list(all_filtered_features.keys()), reverse = True):
            # https://stackoverflow.com/questions/13077527/is-there-a-numpy-delete-equivalent-for-sparse-matrices
            result.rows = np.delete(result.rows, row)
            result.data = np.delete(result.data, row)
            result._shape = (result._shape[0] - 1, result._shape[1])

            # Also remove corresponding components from products' set representations
            for product in products:
                product.set_representation.discard(all_components_list[row])


        if do_print:
            print(f"Binary matrix size: {result.shape}")

        return result


    @staticmethod
    def binary_to_signatures(binary_data: lil_matrix, num_hashes: int, do_print: bool = True) -> np.ndarray:
        result = np.full([num_hashes, binary_data.shape[1]], float("inf"))

        # https://stackoverflow.com/questions/4319014/iterating-through-a-scipy-sparse-vector-or-matrix
        rows, cols = binary_data.nonzero()

        for h in range(num_hashes):
            if do_print:
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

