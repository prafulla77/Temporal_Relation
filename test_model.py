import pickle
import numpy as np
from feature_extract import get_train_features
import tensorflow as tf
tf.python.control_flow_ops = tf

classes_num = 14
#temprel_data_test
with open('temprel_data_test', 'rb') as fp:
    test_data = pickle.load(fp)

X,Y = get_train_features(test_data)

from keras.models import load_model
my_model = load_model('pos_feats2.h5')
print my_model.summary()
temp_predicted = my_model.predict(X)
predicted = []
for y in temp_predicted:
    temp = np.argmax(y)
    predicted.append(temp)
Y = Y.tolist()
gold = []
for y in Y:
    temp = np.argmax(y)
    gold.append(temp)

confusion_matrix = [ [0] * classes_num for _ in range(classes_num)]
TP = 0
for i  in range(len(predicted)):
    confusion_matrix[gold[i]][predicted[i]] += 1
    if gold[i]==predicted[i]:
        TP += 1

for l in confusion_matrix:
    print l
print TP

#721, 170 , 122
