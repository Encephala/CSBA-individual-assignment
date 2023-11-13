#!/usr/bin/env python3
import json

from item import Item

# Load in data
filename = "data/TVs-all-merged.json"

with open(filename, "r") as file:
    data = json.load(file)


# Find all feature keys that are in the data
all_features = {}


def remove_trailing_semicolon(string):
    if string[-1] == ":":
        return string[:-1]

    return string


for key, val in data.items():
    for product in val:
        current_features = product["featuresMap"]

        for key in current_features.keys():
            key = remove_trailing_semicolon(key)

            if key not in all_features:
                all_features[key] = 1

            else:
                all_features[key] += 1


# Explore what the feature landscape is like
# print(f"Number of unique features: {len(all_features)}")
# print(list(filter(lambda i: i[-1] > 250, all_features.items())))

# Make a set out of the features (dropping info on occurrences)
all_features = all_features.keys()



# Get all item instances into big array
products = []

for key, val in data.items():
    for product in val:
        id = product["modelID"]
        features = product["featuresMap"]
        shop = product["shop"]
        title = product["title"]

        products.append(Item(id, features, all_features, shop, title))


# Do shingling
shingle_size = 10

for product in products:
    product.shingle(shingle_size)


# Minhash



# Locality-sensitive hashing


