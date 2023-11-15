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
shingle_size = 5

for product in products:
    product.shingle(shingle_size)


# Minhash
binary_data = Item.minhash(products)

num_hashes = 100
signatures = Item.binary_to_signatures(binary_data, num_hashes)


# Locality-sensitive hashing
similarity_threshold = 0.8

num_bands = 20
num_rows = num_hashes // num_bands
assert num_bands * num_rows == num_hashes

# A prime significantly larger than the number of products
num_buckets = 6337

buckets = [[] for i in range(num_buckets)]

for product in products:
    buckets[hash(product) % num_buckets].append(product)

for bucket in buckets:
    if len(bucket) > 1:
        print(f"{len(bucket)} items in bucket:")
        for item in bucket:
            print(item.title)
            print(item.id)
        print()
