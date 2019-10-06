import numpy as np

data = np.load("face_data.npy")
X = data[:, 1:].astype(int)
y = data[:, 0]


