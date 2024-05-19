from bi_MM import BiMM,return_dict
from CRFs import CRFsModel
from Bert.main import infer
import argparse
import os
from seqeval.metrics.sequence_labeling import get_entities
from transformers import AutoTokenizer
import torch
from transformers import BertForTokenClassification,BertConfig
import random
import bpe
import re

file_path = "./homework3/PKU_TXT/ChineseCorpus199801.txt"
data_path = "./homework3/data/"
result_path = "./homework3/result/"

parser = argparse.ArgumentParser()
parser.add_argument("--text", type=str, help="input text file",default="./homework3/data/processed_test.txt")
parser.add_argument("--model", type=str, help="choose model to segment: bi_MM, CRFs, Bert",default="CRFs")
parser.add_argument("--is_eval",type=str,help="whether to evaluate the model",default=False)
args = parser.parse_args()



label2id = {"O":0,"B":1,"M":2,"E":3,"S":4}
id2label = {v:k for k,v in label2id.items()}
para_path = "./homework3/checkpoint/"


text = []
with open(args.text, 'r', encoding='gbk') as f:
    for line in f.readlines():
        text.append(line)

word_dict = return_dict("./homework3/data/word_count.txt")
result = []
if args.model == "bi_MM":
    bi_MM_model = BiMM(word_dict)
    for line in text:
        result.append(bi_MM_model.bi_directional_max_matching(line))
    with open(result_path+"bi_MM_result.txt", 'w', encoding='gbk') as f:
        for item in result:
            f.write(item + '\n')

elif args.model == "CRFs":
    CRFs_model = CRFsModel()
    if os.path.exists("./homework3/train_result/crf_model.joblib"):
        CRFs_model.load_model("./homework3/train_result/crf_model.joblib")
    else:
        CRFs_model.load_data("./homework3/PKU_TXT/ChineseCorpus199801.txt")
        CRFs_model.train()
        CRFs_model.save_model("./homework3/train_result/crf_model.joblib")
    for line in text:
        words = CRFs_model.segment(line)[0]
        temp = '/'.join(word for word in words).strip()
        result.append(temp)
    with open(result_path+"CRFs_result.txt", 'w', encoding='gbk') as f:
        for item in result:
            f.write(' '.join(item) + '\n')

elif args.model == "Bert":
    bert_model = config = BertConfig.from_pretrained('bert-base-chinese', num_labels=5)
    bert_model = BertForTokenClassification(config)
    bert_model.load_state_dict(torch.load(para_path + "best.pth"))
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    for line in text:
        result.append(infer(bert_model,line,tokenizer,id2label))
    with open(result_path+"Bert_result.txt", 'w', encoding='gbk') as f:
        for item in result:
            f.write(item + '\n')

# indices = random.sample(range(len(result)), 5)
# for i in indices:
#     print("Input text: ", text[i])
#     print("Result: ", result[i])
#     print("-----------------------------")

vocab, freq = bpe.get_vocab(5000, file_path)
sentences, real_values = bpe.wash_test(vocab, data_path+"test.txt")
segments = [item.replace("/"," ") for item in result]
accuracy,recall,f1= bpe.evaluate(segments, real_values)
print('accuracy:',accuracy)
print('recall:',recall)
print('f1:',f1)

squeeze = bpe.bpe(segments)

for i in range(5):
    print("Input text: ", text[i])
    print("tokenized text: ", real_values[i])
    print("Result: ", result[i])
    print("zip: ",squeeze[i])


