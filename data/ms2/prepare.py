"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import numpy as np

BLOCK_SIZE = 128
PAD, BOS, EOS = "[PAD]", "[START]", "[END]"

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input_spectra.all.txt')

with open(input_file_path, 'r') as f:
    lines = f.readlines()

# -------------------------------------------
# add padding
#
data = []
for line in lines:
    d = line.rstrip().split()[:BLOCK_SIZE]
    data.append(d + (BLOCK_SIZE - len(d)) * [PAD])
    assert len(data[-1]) == BLOCK_SIZE
print(f"length of dataset in characters: {len(data):,}")

# -------------------------------------------
# get vocab_size: the unique characters that occur in this text
#
chars = [PAD] + sorted(set(d for sublist in data for d in sublist if d not in [PAD]))
vocab_size = len(chars)
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ' '.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# -------------------------------------------
# create the train and test splits
#
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers
train_ids = encode([d for sublist in train_data for d in sublist])
val_ids = encode([d for sublist in train_data for d in sublist])
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
    'block_size': BLOCK_SIZE
}

#print(meta)

with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# length of dataset in characters:  1115394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1003854 tokens
# val has 111540 tokens
