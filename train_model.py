import pickle
import tensorflow as tf
tf.python.control_flow_ops = tf

from feature_extract import get_train_features

with open('temprel_data_train', 'rb') as fp:
    train_data = pickle.load(fp)

X,Y = get_train_features(train_data)

from model import my_model as my_model
my_model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print my_model.summary()
my_model.fit(X, Y, shuffle = True, batch_size = 100, nb_epoch=100)#, validation_split=0.1, verbose=1)
my_model.save('pos_feats2.h5')
