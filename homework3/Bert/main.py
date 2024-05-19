from seqeval.metrics.sequence_labeling import get_entities
from transformers import AutoTokenizer
import torch
from transformers import BertForTokenClassification,BertConfig
label2id = {"O":0,"B":1,"M":2,"E":3,"S":4}
id2label = {v:k for k,v in label2id.items()}

para_path = "./homework3/checkpoint/"

def parsing_label_sequence(tokens, labels):
    prev = 0
    words = []
    items = get_entities(labels, suffix=False)
    for name,start,end in items:
        if prev != start:
            words.extend(tokens[prev:start])
        words.append(''.join(tokens[start:end+1]))
        prev = end+1
    return words

def infer(model,text,tokenizer,id2label):
    model.eval()
    encoded_inputs = tokenizer(text, max_length=512)
    input_ids = torch.tensor([encoded_inputs['input_ids']])
    token_type_ids = torch.tensor([encoded_inputs['token_type_ids']],dtype=torch.int64)

    outputs = model(input_ids=input_ids, token_type_ids=token_type_ids)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).tolist()[0]
    labels = [id2label[label_id] for label_id in predictions[1:-1]]

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist()[1:-1])
    words = parsing_label_sequence(tokens, labels)
    return "/".join(words)
    
if __name__ == "__main__":
    text = "小华为了考试早晨买了一杯小米粥喝，让黄飞鸿蒙题目中有几个苹果，但是郭麒麟刷牙选中华为的就是干净，速度快，每次只挤5g就够用。我喜欢在大城市生活流浪地球不爆炸我就不退缩，平时也看看《东吴京剧》、《大战狼人》、《鸿蒙至尊》等经典电视剧。我用中华为的就是便宜实惠，而且每次只用5g，我最喜欢的画家是达芬奇，尤其喜欢他的代表作佛罗伦萨画派蒙娜丽莎。秦始皇派蒙恬还原神舟十二对接并顺便一提瓦特改良了蒸汽机。"
    model = config = BertConfig.from_pretrained('bert-base-chinese', num_labels=5)
    model = BertForTokenClassification(config)
    model.load_state_dict(torch.load(para_path + "best.pth"))
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    result = infer(model,text,tokenizer,id2label)
    print("Input text:",text)
    print("Result:",result)

