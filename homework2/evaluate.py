import torch
import torch.nn as nn
import numpy as np
def get_lookup_table(model):
    lookup_table = model.embedding.weight.data
    lookup_table = lookup_table.cpu().numpy()
    return lookup_table

def top_10_similar(lookup_table, word_idx, data):
    word_vec = lookup_table[word_idx]
    similarity = np.dot(lookup_table, word_vec)/np.linalg.norm(lookup_table, axis=1)/np.linalg.norm(word_vec)
    a = np.argsort(-similarity)
    result = ""
    for i in a[:10]:
        name_list = [key for key,value in data.top_words.items() if value==i]
        if len(name_list) > 0:
            print(name_list[0], similarity[i])
            result += name_list[0] + " " + str(similarity[i]) + "\n"
        else:
            print("<UNK>", similarity[i])
            result += "<UNK> " + str(similarity[i]) + "\n"
    return result
