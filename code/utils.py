import pandas as pd
import torch
import os
from torch.utils.data import TensorDataset


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_dataset(tokenizer, split='train'):
    ys = []
    xs = []
    masks = []
    fpath = os.path.join('data', 'imdb_train_test.csv')
    train_test_df = pd.read_csv(fpath)
    train_df = train_test_df.query(f"split == '{split}'")
    print(train_df.shape)

    for ind, row in train_df.iterrows():
        y = row['label']
        text = row['text']
        tokens, token_ids, mask = tokenizer.tokenize(text, truncate=512, padding=512)  # TODO: parameterize 512
        ys.append(y)
        xs.append(token_ids)
        masks.append(mask)

    xs_tensor = torch.as_tensor(xs)  # size, 512
    masks_tensor = torch.as_tensor(masks).float()  # size, 512
    ys_tensor = torch.as_tensor(ys).float().unsqueeze(1)  # size,1
    dataset = TensorDataset(xs_tensor, masks_tensor, ys_tensor)
    return dataset


def build_datasets(tokenizer):
    train_dataset = build_dataset(tokenizer, 'train')
    test_dataset = build_dataset(tokenizer, 'test')
    return train_dataset, test_dataset
