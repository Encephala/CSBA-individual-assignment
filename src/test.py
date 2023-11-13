#!/usr/bin/env python3

def shingle(string, shingle_size):
    shingled_data = set()

    for i in range(len(string) - shingle_size):
        shingled_data.add(string[i:i+shingle_size])

    return shingled_data

print(shingle("abcab", 2))
