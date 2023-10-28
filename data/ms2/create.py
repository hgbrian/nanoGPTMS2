import random
from collections import Counter
import numpy as np

RANDOMIZE = False

vocab = [round(100 + random.random()*900, 2) for _ in range(100)]

vocab = [round(x, 2) for x in np.random.normal(400, 200, 110) if x >= 100]
print(len(set(vocab)))

num_entries = [round(x) for x in np.random.normal(5, 5, 10_000) if x >= 1]

data = []
diffs = []
alldiffs = []
for n in num_entries:
    spec = sorted(set([random.choice(vocab) for _ in range(n)]), reverse=True)
    data.append(spec)
    diffs.append(sorted([round(spec[0] - d, 2) for d in spec[1:]], reverse=True))
    alldiffs.append(sorted([round(spec[n] - spec[m], 2) for n in range(len(spec)-1) for m in range(n+1, len(spec))], reverse=True))


def flatten(alist):
    return [a for sublist in alist for a in sublist]

flat_data = flatten(data)
flat_diffs = flatten(diffs)
flat_alldiffs = flatten(alldiffs)

truncated_diffs = [k for k,v in sorted(list(Counter(flat_diffs).items()), key=lambda x: x[1], reverse=True)[:len(vocab)*3]]
truncated_alldiffs = [k for k,v in sorted(list(Counter(flat_alldiffs).items()), key=lambda x: x[1], reverse=True)[:len(vocab)*3]]

#print("data 10k * 5?", len(data))
#print(len(truncated_diffs))
#print(len(truncated_alldiffs))
#print(diffs[:10])
#print(alldiffs[:10])

for n in range(len(data)):
    #print(data[n])
    #print([d for d in diffs[n] if d in truncated_diffs])
    #print([d for d in alldiffs[n] if d in truncated_alldiffs])
    if RANDOMIZE:
        random.shuffle(data[n])
        random.shuffle(diffs[n])

    print(f"p-{data[n][0]:.2f}", end='')
    print(" " + " ".join(f"m-{d:.2f}" for d in data[n]), end='')
    print(" " + " ".join(f"d-{d:.2f}" for d in diffs[n]), end='')
    #print(" " + " ".join(f"a-{d:.2f}" for d in alldiffs[n]), end='')
    print()
