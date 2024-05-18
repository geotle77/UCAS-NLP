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
    text = "迈向充满希望的新世纪，一九九八年新年讲话"
    model = config = BertConfig.from_pretrained('bert-base-chinese', num_labels=5)
    model = BertForTokenClassification(config)
    model.load_state_dict(torch.load(para_path + "best.pth"))
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    result = infer(model,text,tokenizer,id2label)
    print(result)

