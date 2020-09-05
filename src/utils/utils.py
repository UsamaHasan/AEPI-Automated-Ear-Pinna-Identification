import numpy as np

def to_categorical(arr):
    max_ = int(np.amax(arr))
    I = np.identity(max_)
    one_hot_labels = []
    for i in arr:
        index = int(i-1)
        one_hot_labels.append(I[index][:])
    print(type(one_hot_labels[1]))
    return one_hot_labels


