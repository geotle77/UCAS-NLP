import math
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from collections import Counter

__path__="F:/CODES/Python/UCAS-NLP/homework1/test/analyzed_text/"
params = [
    {'name':'hongloumeng'},
    {'name':'sanguoyanyi'},
    {'name':'xiyouji'},
    {'name':'shuihuzhuan'}
]
texts=[]
for name in params:
    name=name['name']
    texts.append("cleaned_"+name+".txt")

# Define a function to calculate the entropy of a given text
def calculate_entropy(text):
    # Count the frequency of each character
    char_counts = Counter(text)
    # Calculate the frequency of each character
    char_frequencies = {char: count / len(text) for char, count in char_counts.items()}
    # Calculate the entropy of each character and sum them up
    entropy = sum(-p * math.log(p) for p in char_frequencies.values())
    return entropy

def visualize_most_frequent_chars(text, n, entropy):
    # Count the frequency of each character
    char_counts = Counter(text)
    # Get the most frequent characters
    most_frequent_chars = char_counts.most_common(n)

    # Separate the characters and their frequencies
    chars, frequencies = zip(*most_frequent_chars)

    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    # Create a bar plot
    bars = plt.bar(chars, frequencies)

    # Add the frequency on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.05, round(yval, 2), ha='center', va='bottom')

    # Add the total entropy at the top right corner
    plt.text(0.95, 0.95, f"Total entropy: {entropy}", ha='right', va='top', transform=plt.gca().transAxes)

    plt.xlabel('Characters', fontproperties=font)
    plt.ylabel('Frequency', fontproperties=font)
    plt.title('Most Frequent Characters', fontproperties=font)
    plt.xticks(fontproperties=font)
    plt.show()

# Calculate the entropy of each text
if __name__ == "__main__":
    print(texts)
    all_text=""
    for name in texts:
       with open(__path__+name, 'r', encoding='utf-8') as f:
            text = f.read()
            all_text+=text
    entropy = calculate_entropy(all_text)
    print(f"The entropy of the combined text is {entropy}")
    visualize_most_frequent_chars(all_text, 10, entropy)