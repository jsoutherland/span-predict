# This script processes the raw aclImdb dataset into a .csv file that will be easier to work with
import os
import pandas as pd

records = []
imdb_dir = os.path.join('data', 'aclImdb')
output_path = os.path.join('data', 'imdb_train_test.csv')


for split in ['train', 'test']:
    for label in ['pos', 'neg']:
        y = 1.0 if label == 'pos' else 0.0
        subdir = os.path.join(imdb_dir, split, label)
        print(subdir)
        for filename in os.listdir(subdir):
            with open(os.path.join(subdir, filename), 'r') as f:
                text = f.read()
                record = {
                    'label': y,
                    'split': split,
                    'text': text
                }
                records.append(record)

df = pd.DataFrame.from_records(records)
df.to_csv(output_path, index=False)
