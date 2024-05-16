from torch.utils.data import BatchSampler, DataLoader,RandomSampler
import torch
from datasets import load_from_disk
data_path = "./homework3/PKU_TXT/ChineseCorpus199801.txt"
file_path = "./homework3/data/"

def collate_fn(batch_data, pad_token_id=0, pad_token_type_id=0, pad_label_id=0):
    input_ids_list, token_type_ids_list, label_list, attention_mask_list = [], [], [], []
    max_len = 0
    for example in batch_data:
        input_ids, token_type_ids, labels = example['input_ids'], example['token_type_ids'], example['labels']
        input_ids_list.append(input_ids)
        token_type_ids_list.append(token_type_ids)
        label_list.append(labels)
        
        max_len = max(max_len, len(input_ids))
        
    for i in range(len(input_ids_list)):
        cur_len = len(input_ids_list[i])
        padding_len = max_len - cur_len
        input_ids_list[i] = input_ids_list[i] + [pad_token_id] * padding_len
        token_type_ids_list[i] = token_type_ids_list[i] + [pad_token_type_id] * padding_len
        label_list[i] = label_list[i] + [pad_label_id] * padding_len
        attention_mask = [1] * cur_len + [0] * padding_len
        attention_mask_list.append(attention_mask)
        
    return torch.tensor(input_ids_list), torch.tensor(token_type_ids_list), torch.tensor(label_list), torch.tensor(attention_mask_list)

if __name__ == '__main__':
    train_dataset = load_from_disk(file_path+'train_dataset')
    dev_dataset = load_from_disk(file_path+'dev_dataset')
    test_dataset = load_from_disk(file_path+'test_dataset')
    
    
    batch_size = 16
    train_sampler = RandomSampler(train_dataset)
    dev_sampler = RandomSampler(dev_dataset)
    test_sampler = RandomSampler(test_dataset)

    train_loader = DataLoader(dataset=train_dataset, sampler=train_sampler, batch_size=batch_size, collate_fn=collate_fn)
    dev_loader = DataLoader(dataset=dev_dataset, sampler=dev_sampler, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(dataset=test_dataset, sampler=test_sampler, batch_size=batch_size, collate_fn=collate_fn)
    
    print(next(iter(train_loader)))
    print(next(iter(dev_loader)))
    print(next(iter(test_loader)))