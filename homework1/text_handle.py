import jieba
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # 删除所有标点符号
    text = re.sub(r'\d+', '', text)  # 删除所有数字
    text = re.sub(r'\s+', ' ', text)  # 替换多余的空格
    text = re.sub(r'\n+', '\n', text)  # 替换多余的换行符
    text = text.lower()  # 转换为小写
    return text.strip()  # 删除首尾的空格和换行符


def save_text(text, filename):
    with open(filename,'w',encoding='utf-8') as f:
        f.write(text)

def tokenize(text, language='english'):
    if language == 'english':
        return word_tokenize(text)
    elif language == 'chinese':
        return list(jieba.cut(text))
    else:
        raise ValueError('Unsupported language: ' + language)

if __name__ == "__main__":
    with open('output.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    text = clean_text(text)
    tokenize(text, 'english')
    save_text(text, 'output_clean.txt')