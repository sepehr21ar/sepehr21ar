
import tensorflow as tf
import numpy as np


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train = np.array([np.ravel(x) for x in X_train]). astype(np.float32) / 255.0
X_test = np.array([np.ravel(x) for x in X_test]). astype(np.float32) / 255.0

print(X_train.shape, "\n",X_test.shape)
#
# from sklearn.model_selection import GridSearchCV
# from sklearn.neighbors import KNeighborsClassifier
"======================================"
"We check different value for each parameter and change the grid search list eveery time "
"======================================"
# model = GridSearchCV(KNeighborsClassifier(),param_grid= {
#     # "n_neighbors":[2, 5, 10, 50, 100],
#     "n_neighbors" : [4],
#     "weights" : [ "distance"],
#     "algorithm" : ["auto"],
#     "leaf_size" : [5, 10, 15, 20, 25 , 30 , 35 ,40],
# }, verbose=2)
#
# model.fit(X_train, y_train)
# print(model.best_params_)
# print(model.best_score_)
# print(model.best_estimator_)
# print(model.score)
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(algorithm="auto", n_neighbors=4, weights="distance" , leaf_size=4)
model.fit(X_train, y_train)
t_pred = model.predict(X_test)
print(classification_report(y_test, t_pred))

import cv2 as cv

for i in range(0,10):
    img = cv.imread(f"C:\\Users\\sepeh\\OneDrive\\Desktop\\ML 04\\dataset\\{i}.bmp", 0)
    x_num_test = np.ravel(img).astype(np.float32)/255.0
    print(model.predict(x_num_test.reshape(1, -1)))
