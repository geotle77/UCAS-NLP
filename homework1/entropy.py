import math
from collections import Counter


# Define a function to calculate the entropy of a given text
def calculate_entropy(text):
    # Count the frequency of each character
    char_counts = Counter(text)
    # Calculate the frequency of each character
    char_frequencies = {char: count / len(text) for char, count in char_counts.items()}
    # Calculate the entropy of each character and sum them up
    entropy = sum(-p * math.log(p) for p in char_frequencies.values())
    return entropy