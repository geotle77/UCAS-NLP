import jieba
import re
PATH="F:/CODES/Python/UCAS-NLP/homework1/test/analyzed_text/"
params = [
    {'name':'hongloumeng'},
    {'name':'sanguoyanyi'},
    {'name':'xiyouji'},
    {'name':'shuihuzhuan'}
]


def text_clean(text):
     text = re.sub("[\s+\.\!\/_,$%^*(+\"\')+“”+：+‘’|[+——！，。？、~@#￥%……&*（）；]+", "", text)
    #  words = jieba.cut(text)
     text = text.replace(" ", "")
     return text

def save_cleaned_text(text, name):
    with open(PATH+"cleaned_"+ name, 'w', encoding='utf-8') as f:
        f.write(text)
        print(f"cleaned_{name} is saved")
    
if __name__ == "__main__":
    for name in params:
        with open(PATH+name['name']+".txt", 'r', encoding='utf-8') as f:
            text = f.read()
            cleaned_text=text_clean(text)
            save_cleaned_text(cleaned_text, name['name']+".txt")
            print(f"{name['name']} is done")