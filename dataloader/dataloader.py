import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pickle
from tqdm import tqdm
from torch.utils.data import Sampler
import math

# 更新疾病标签列表
id2disease = [
    "adhd",
    "anxiety",
    "bipolar",
    "depression",
    "eating",
    "ocd",
    "ptsd",
    "control"
]

# 添加反向映射字典
disease2id = {disease: idx for idx, disease in enumerate(id2disease)}

class BalanceSampler(Sampler):
    def __init__(self, data_source, control_ratio=0.75) -> None:
        self.data_source = data_source
        self.control_ratio = control_ratio
        self.indexes_control = np.where(data_source.is_control == 1)[0]
        self.indexes_diseases = []
        for idx, disease in enumerate(id2disease):
            disease_indexes = np.where(np.array(data_source.labels)[:,idx] == 1)[0]
            print(disease_indexes)
            self.indexes_diseases.append(disease_indexes)
        self.len_control = len(self.indexes_control)
        self.len_diseases = []
        for disease_idx in self.indexes_diseases:
            self.len_diseases.append(len(disease_idx))
        np.random.shuffle(self.indexes_control)
        for i in range(len(self.indexes_diseases)):
            np.random.shuffle(self.indexes_diseases[i])

        self.pointer_control = 0
        self.pointer_disease = [0] * len(id2disease)

    def __iter__(self):
        for i in range(len(self.data_source)):
            rand_num = np.random.rand()
            if rand_num < self.control_ratio:
                id0 = np.random.randint(self.pointer_control, self.len_control)
                sel_id = self.indexes_control[id0]
                self.indexes_control[id0], self.indexes_control[self.pointer_control] = self.indexes_control[self.pointer_control], self.indexes_control[id0]
                self.pointer_control += 1
                if self.pointer_control >= self.len_control:
                    self.pointer_control = 0
                    np.random.shuffle(self.indexes_control)
            else:
                chosen_disease = math.floor((rand_num-self.control_ratio)*1.0/((1-self.control_ratio)/len(id2disease)))
                id0 = np.random.randint(self.pointer_disease[chosen_disease], self.len_diseases[chosen_disease])
                sel_id = self.indexes_diseases[chosen_disease][id0]
                self.indexes_diseases[chosen_disease][id0] = self.indexes_diseases[chosen_disease][self.pointer_disease[chosen_disease]]
                self.indexes_diseases[chosen_disease][self.pointer_disease[chosen_disease]] = self.indexes_diseases[chosen_disease][id0]
                self.pointer_disease[chosen_disease] += 1
                if self.pointer_disease[chosen_disease] >= self.len_diseases[chosen_disease]:
                    self.pointer_disease[chosen_disease] = 0
                    np.random.shuffle(self.indexes_diseases[chosen_disease])
            
            yield sel_id

    def __len__(self) -> int:
        return len(self.data_source)



class MentalDataset(Dataset):
    def __init__(self, input_dir, tokenizer, max_len, split="train", disease='None', use_symp=True, max_posts=64):
        assert split in {"train", "val", "test"}
        self.input_dir = input_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_posts = max_posts
        self.data = []
        self.labels = []
        self.is_control = []
        
        input_dir2 = os.path.join(input_dir, split+'.pkl')
        with open(input_dir2, 'rb') as f:
            raw_data = pickle.load(f)

        for record in tqdm(raw_data, desc='Process data:    '):
            label_vector = [0] * len(id2disease)
            
            if len(record['diseases']) == 0:
                # control组
                label_vector[-1] = 1  # control位置设为1
                self.is_control.append(1)
            else:
                # 疾病组 - 只取第一个疾病作为标签
                disease = record['diseases'][0] 
                if disease in disease2id:
                    label_vector[disease2id[disease]] = 1
                self.is_control.append(0)
            
            sample = {}
            # 处理posts
            posts = record['selected_posts'][:max_posts]
            augmented_posts = record['augmented_posts'][:max_posts]
            
            # tokenize文本
            tokenized_original = tokenizer(posts, truncation=True, padding='max_length', max_length=max_len)
            tokenized_augmented = tokenizer(augmented_posts, truncation=True, padding='max_length', max_length=max_len)
            
            # 存储tokenized结果
            for k, v in tokenized_original.items():
                sample[k] = v
            for k, v in tokenized_augmented.items():
                sample[f'augmented_{k}'] = v
            
            # 处理症状信息
            if use_symp:
                symp_data = np.array(record['symptoms'])
                sample['symp'] = symp_data
                
            self.data.append(sample)
            self.labels.append(label_vector)
            
        self.labels = np.array(self.labels)
        self.is_control = np.array(self.is_control).astype(int)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index], self.labels[index]
    
def my_collate_mental(data):
    labels = []
    processed_batch = []
    for item, label in data:
        user_feats = {}
        for k, v in item.items():
            if k.startswith('augmented_'):
                user_feats[k] = torch.LongTensor(v)
            elif k != 'symp':
                user_feats[k] = torch.LongTensor(v)
            else:
                user_feats[k] = torch.FloatTensor(v)
        processed_batch.append(user_feats)
        labels.append(label)
    labels = torch.FloatTensor(np.array(labels))
    label_masks = torch.ones_like(labels) 
    return processed_batch, labels, label_masks


class MentalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        bs,
        input_dir,
        tokenizer,
        max_len,
        disease='None',
        setting='multilabel',
        use_symp=True,
        bal_sample=False,
        control_ratio=0.8
    ):
        super().__init__()
        self.bs = bs
        self.input_dir = input_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.disease = disease
        self.setting = setting
        self.control_ratio = control_ratio
        self.bal_sample = bal_sample
        self.use_symp = use_symp

    def setup(self, stage):
        if stage == "fit":
            self.train_set = MentalDataset(self.input_dir, self.tokenizer, self.max_len, "train", self.disease, self.use_symp)
            self.val_set = MentalDataset(self.input_dir, self.tokenizer, self.max_len, "val", self.disease, self.use_symp)
            self.test_set = MentalDataset(self.input_dir, self.tokenizer, self.max_len, "test", self.disease, self.use_symp)
        elif stage == "test":
            self.test_set = MentalDataset(self.input_dir, self.tokenizer, self.max_len, "test", self.disease, self.use_symp)

    def train_dataloader(self):
        if self.bal_sample:
            sampler = BalanceSampler(self.train_set, self.control_ratio)
            return DataLoader(self.train_set, batch_size=self.bs, collate_fn=my_collate_mental, sampler=sampler, pin_memory=False, num_workers=4)
        else:
            return DataLoader(self.train_set, batch_size=self.bs, collate_fn=my_collate_mental, shuffle=True, pin_memory=False, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.bs, collate_fn=my_collate_mental, pin_memory=False, num_workers=4)
    
    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.bs, collate_fn=my_collate_mental, pin_memory=False, num_workers=4)
