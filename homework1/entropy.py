import math
from collections import Counter
import os
import matplotlib.pyplot as plt
import matplotlib
chunk_size = 2*1024*1024

# Define a function to calculate the entropy of a given text
def calculate_entropy(text):
    # Count the frequency of each character
    char_counts = Counter(text)
    # Calculate the frequency of each character
    char_frequencies = {char: count / len(text) for char, count in char_counts.items()}
    # Calculate the entropy of each character and sum them up
    entropy = sum(-p * math.log2(p) for p in char_frequencies.values())
    return entropy

def read_file_in_chunks(file_path, chunk_size):
    with open(file_path, 'r', encoding='utf-8') as file:
        while True:
            data = file.read(chunk_size)
            if not data:
                break
            yield data

def count_chars_in_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    char_counts = Counter(text)
    total_chars = sum(char_counts.values())
    total_chars = sum(char_counts.values())
    top10_chars = char_counts.most_common(10)
    return total_chars, top10_chars


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
    sizes, entropies = zip(*sizes_and_entropies)
    plt.plot(sizes, entropies)
    plt.xlabel('Size')
    plt.ylabel('Entropy')
    plt.title('Entropy Curve')
    plt.show()

def draw_char_counts(total_chars, top10_chars):
    labels, counts = zip(*top10_chars)
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.bar(labels, counts)
    plt.xlabel('Character')
    plt.ylabel('Count')
    plt.title(f'Top 10 Characters (Total: {total_chars})')
    plt.show()


if __name__ == "__main__":
    
    chinese_entropy = calculate_entropy_of_file('./output/chinese_output.txt')
    
    english_entropy = calculate_entropy_of_file('./output/english_output.txt')

    chinese_info = count_chars_in_file('./output/chinese_output.txt')
    english_info = count_chars_in_file('./output/english_output.txt')
    
    
    draw_entropy_curve(chinese_entropy)
    draw_entropy_curve(english_entropy)
    
    draw_char_counts(*chinese_info)
    draw_char_counts(*english_info)
