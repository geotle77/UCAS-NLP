import math
from collections import Counter
import os

chunk_size = 2*1024*1024

# Define a function to calculate the entropy of a given text
def calculate_entropy(text):
    # Count the frequency of each character
    char_counts = Counter(text)
    # Calculate the frequency of each character
    char_frequencies = {char: count / len(text) for char, count in char_counts.items()}
    # Calculate the entropy of each character and sum them up
    entropy = sum(-p * math.log(p) for p in char_frequencies.values())
    return entropy

def read_file_in_chunks(file_path, chunk_size):
    with open(file_path, 'r', encoding='utf-8') as file:
        while True:
            data = file.read(chunk_size)
            if not data:
                break
            yield data


def calculate_entropy_of_file(filename):
    # Get the size of the file
    file_size = os.path.getsize(filename)

    # Initialize the current size
    current_size = chunk_size

    # Initialize a list to store the sizes and entropies
    sizes_and_entropies = []

    text = ""
    for chunk in read_file_in_chunks(filename, chunk_size):
        text += chunk
        # Calculate the entropy of the text
        entropy = calculate_entropy(text)
        # Record the current size and entropy
        sizes_and_entropies.append((current_size, entropy))
        # Increase the current size
        current_size += chunk_size
    
    for size, entropy in sizes_and_entropies:
        print(f"Size: {size}, Entropy: {entropy}")
    return sizes_and_entropies


def draw_entropy_curve(sizes_and_entropies):
    import matplotlib.pyplot as plt
    sizes, entropies = zip(*sizes_and_entropies)
    plt.plot(sizes, entropies)
    plt.xlabel('Size')
    plt.ylabel('Entropy')
    plt.title('Entropy Curve')
    plt.show()


if __name__ == "__main__":
    
    draw_entropy_curve(calculate_entropy_of_file('./output/chinese_output.txt'))

    draw_entropy_curve(calculate_entropy_of_file('./output/english_output.txt'))
