import pickle
import itertools
import matplotlib.pyplot as plt
import numpy as np

with open('bucketing_accuracy.pickle', 'rb') as handle:
    results = pickle.load(handle)

mean_acc = [[sum(h)/len(h) for h in l] for l in results]

mean_acc = list(itertools.chain.from_iterable(mean_acc))

weights = np.ones_like(mean_acc) / len(mean_acc)

plt.xlabel("Attention window prediction accuracy")
plt.ylabel("fraction of heads")

plt.hist(mean_acc, bins=25, weights=weights)

plt.savefig("mean_acc.pdf", format="pdf", bbox_inches="tight")