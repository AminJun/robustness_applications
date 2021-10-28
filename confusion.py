from soft_xent import SoftCrossEntropy, CachedData
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

GLOBAL_MODE = int(sys.argv[1])

cached_data = CachedData('.', GLOBAL_MODE)
cached_data.cache(None, None, None)
label = cached_data().cuda()
print(label)

animals = ['dog', 'cat', 'frog', 'turtle', 'bird', 'primate', 'fish', 'crab', 'insect']
harvest = label.cpu().numpy()

fig, ax = plt.subplots()
im = ax.imshow(harvest)

# We want to show all ticks...
ax.set_xticks(np.arange(len(animals)))
ax.set_yticks(np.arange(len(animals)))
# ... and label them with the respective list entries
ax.set_xticklabels(animals)
ax.set_yticklabels(animals)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(animals)):
    for j in range(len(animals)):
        text = ax.text(j, i, round(harvest[i, j], 1),
                       ha="center", va="center", color="w")

ax.set_title("Harvest of local farmers (in tons/year)")
fig.tight_layout()
fig.savefig(f'desktop/c{GLOBAL_MODE}.png')
