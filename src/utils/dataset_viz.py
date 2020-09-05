import matplotlib.pyplot as plt
import numpy as np

dataset = np.load('images.npy',allow_pickle=True)

print(f'Shape:{dataset.shape}')

for img in dataset:
    plt.imshow(img)
    plt.show()