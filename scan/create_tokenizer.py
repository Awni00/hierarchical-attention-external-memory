import tensorflow_datasets as tfds
import data_utils
import argparse
import os

argparser = argparse.ArgumentParser()
argparser.add_argument('--split', type=str, required=True)
argparser.add_argument('--out_dir', type=str, default='text_vectorizer_vocabs')

args = argparser.parse_args()

print(f'Creating vectorizers for {args.split} split')

data = tfds.load(f'scan/{args.split}', as_supervised=True)
train_ds = data['train']

# add start and end tokens to target sequence
def add_start_eos_token(text):
    return "<START> " + text + " <END>"

train_ds = train_ds.map(lambda x, y: (x, add_start_eos_token(y)))

source, target = data_utils.unravel_ds(train_ds, format='s,t')

save_path = f'{args.out_dir}/{args.split}'
if not os.path.exists(save_path):
    os.mkdir(save_path)

data_utils.create_vectorizers(source, target, save_path=save_path)

print('Done!')