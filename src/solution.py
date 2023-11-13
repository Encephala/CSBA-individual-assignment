#!/usr/bin/env python3
import json

from item import Item

# Load in data
filename = "data/TVs-all-merged.json"

with open(filename, "r") as file:
    data = json.load(file)


# Find all feature keys that are in the data
all_features = set()


def remove_trailing_semicolon(string):
    if string[-1] == ":":
        return string[:-1]

    return string


for key, val in data.items():
    for product in val:
        current_features = product["featuresMap"]

        for key in current_features.keys():
            all_features.add(remove_trailing_semicolon(key))


# Get all item instances into big array
products = []

for key, val in data.items():
    for product in val:
        id = product["modelID"]
        features = product["featuresMap"]

        products.append(Item(id, features, all_features))


# Do shingling
