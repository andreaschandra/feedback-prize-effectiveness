#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pickle
from types import SimpleNamespace
from datetime import datetime
import random
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

from sklearn.model_selection import train_test_split
from tqdm import tqdm


# In[ ]:


training_id = datetime.now().strftime('%Y%m%d%H%M%S')


# In[ ]:


tqdm.pandas()


# In[ ]:


DATA_DIR = "../data"


# In[ ]:


d_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
d_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
d_submit = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))


# In[ ]:


d_train.head()


# In[ ]:


d_test.head()


# In[ ]:


d_submit.head()


# In[ ]:


def set_all_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# In[ ]:


set_all_seed()


# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[ ]:


print("device:", device)


# In[ ]:


CONFIG = SimpleNamespace()
CONFIG.model_name = 'microsoft/deberta-v3-base'
CONFIG.max_len = 512
CONFIG.classes = 3
CONFIG.n_folds = 5
CONFIG.lr = 1e-4
CONFIG.epochs = 5
CONFIG.batch_size = 10


# ### Text Preprocessing

# In[ ]:


tokenizer = AutoTokenizer.from_pretrained(CONFIG.model_name)


# In[ ]:


class FeedbackDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.df['inputs'] = self.df.discourse_type + ' ' + tokenizer.sep_token + ' ' + d_train.discourse_text
        
        # preproceesing
        self.target_map = target_map = {'Adequate': 0, 'Effective': 1, 'Ineffective': 2}
        self.df['target'] = self.df.discourse_effectiveness.map(target_map)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        text = self.df.loc[index, 'inputs']
        target = self.df.loc[index, 'target']
    
        return text, target


# In[ ]:


d_train.head()


# In[ ]:


train, val = train_test_split(d_train, test_size=0.2, random_state=42, stratify=d_train.discourse_effectiveness)


# In[ ]:


train.reset_index(drop=True, inplace=True)


# In[ ]:


val.reset_index(drop=True, inplace=True)


# In[ ]:


train.shape


# In[ ]:


val.shape


# In[ ]:


dataset_train = FeedbackDataset(train, tokenizer)
dataset_val = FeedbackDataset(val, tokenizer)


# In[ ]:


def tokenizer_fn(input_):
    text, target = zip(*input_)
    text = list(text)
    text_tokenize = tokenizer(text, max_length=CONFIG.max_len, truncation=True, padding=True, return_tensors="pt")
    text_tokenize['input_ids'] = text_tokenize['input_ids'].to(device)
    text_tokenize['token_type_ids'] = text_tokenize['token_type_ids'].to(device)
    text_tokenize['attention_mask'] = text_tokenize['attention_mask'].to(device)
    
    target = torch.LongTensor(target).to(device)
    
    return text_tokenize, target


# In[ ]:


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(CONFIG.model_name, num_labels=CONFIG.classes)
        
    def forward(self, input_):
        out = self.model(**input_)
        
        return out


# In[ ]:


model = CustomModel().to(device)


# In[ ]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG.lr)


# In[ ]:


model_history = {
    'train_loss': [],
    'val_loss': []
}


# In[ ]:


for epoch in range(1, 51):
    running_loss = 0
    running_loss_val = 0
    
    start = time.time()
    
    model.train()
    train_gen = DataLoader(dataset_train, batch_size=CONFIG.batch_size, collate_fn=tokenizer_fn)
    for batch_index, (x_train, y_train) in tqdm(enumerate(train_gen, 1)):
        
        optimizer.zero_grad()
        
        out = model(x_train)
        
        loss = criterion(out.logits, y_train)
        running_loss += (loss.item() - running_loss) / batch_index
        
        loss.backward()
        optimizer.step()
        
    model.eval()
    val_gen = DataLoader(dataset_val, batch_size=CONFIG.batch_size, collate_fn=tokenizer_fn)
    with torch.no_grad():
        for batch_index, (x_val, y_val) in tqdm(enumerate(val_gen, 1)):
            
            out = model(x_val)
            
            loss = criterion(out.logits, y_val)
            running_loss_val += (loss.item() - running_loss_val) / batch_index
    
    duration = time.time() - start
    
    model_history['train_loss'].append(running_loss)
    model_history['val_loss'].append(running_loss_val)
    
    print(f"epoch: {epoch} | duration: {duration:.2f}s")
    print(f"\tTrain loss: {running_loss:.2f} | Val loss: {running_loss_val:.2f}")
    
    if epoch % 5 == 0:
        PATH = f'../model/DeBERTa_{training_id}_train_{running_loss:.2f}val_{running_loss_val:.2f}.pth'
        torch.save(model.state_dict(), PATH)
        
pickle.dump(model_history, open(f'../model/model_history_{training_id}.pkl', 'wb'))


# In[ ]:




