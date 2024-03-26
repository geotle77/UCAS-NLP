import jieba
import re
import nltk

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # 删除所有标点符号
    text = re.sub(r'\d+', '', text)  # 删除所有数字
    text = re.sub(r'\s+', ' ', text)  # 替换多余的空格
    text = re.sub(r'\n+', '\n', text)  # 替换多余的换行符
    text = text.lower()  # 转换为小写
    return text.strip()  # 删除首尾的空格和换行符


def filter(text, language):
    if language == 'chinese':
        text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
        return text
    elif language == 'english':
        text = re.sub(r'[^a-zA-Z]', '', text)
        return text
    


def text_handle(filename,outputname,language):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    text = clean_text(text)
    text = filter(text, language)
    with open(outputname, 'w', encoding='utf-8') as f:
        f.write(text)


if __name__ == "__main__":
    text_handle('./homework1/chinese.txt', './homework1/chinese_output.txt', 'chinese')
    text_handle('./homework1/english.txt', './homework1/english_output.txt', 'english')
