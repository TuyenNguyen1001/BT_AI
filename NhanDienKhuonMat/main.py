import cv2
import glob
import numpy as np
import pickle   
x_train = []
y_train = []

for filename in glob.glob("./data2/0/*.jpg"):
    img = cv2.resize(cv2.imread(filename), (150,150))
    x_train.append(img)
    y_train.append(0)

for filename in glob.glob("./data2/1/*.bmp"):
    img = cv2.resize(cv2.imread(filename), (150,150))
    x_train.append(img)
    y_train.append(1)


# for filename in glob.glob("./data/2/*.bmp"):
#     img = cv2.resize(cv2.imread(filename), (150,150))
#     x_train.append(img)
#     y_train.append(2)

# for filename in glob.glob("./data/3/*.bmp"):
#     img = cv2.resize(cv2.imread(filename), (150,150))
#     x_train.append(img)
#     y_train.append(3)



x_train = np.array(x_train)
y_train = np.array(y_train)
print(x_train.shape)
print(y_train.shape)
with open("Data_AI000.pickle", "wb") as f:
        pickle.dump((x_train, y_train),f)
