import re

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
    
def text_merge(file1, file2, outputname):
    with open(file1, 'r', encoding='utf-8') as f:
        text1 = f.read()
    with open(file2, 'r', encoding='utf-8') as f:
        text2 = f.read()
    text = text1 + text2
    with open(outputname, 'w', encoding='utf-8') as f:
        f.write(text)

def text_handle(filename,outputname,language):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    text = clean_text(text)
    text = filter(text, language)
    with open(outputname, 'w', encoding='utf-8') as f:
        f.write(text)




if __name__ == "__main__":
    # text_handle('./chinese.txt', './output/chinese_output.txt', 'chinese')
    # text_handle('./output/english.txt', './output/english_output.txt', 'english')
    text_merge('./output/chinese_output.txt', './output/append_chinese_output.txt', './output/merged_chinese_output.txt')
