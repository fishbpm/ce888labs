'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.layers import Input, Dense, Activation, Dropout, PReLU
from keras.wrappers.scikit_learn import KerasClassifier

from keras.optimizers import Adam
from keras.models import Model
from keras.utils import np_utils

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

batch_size = 128
nb_classes = 10
nb_epoch = 20
n_hidden_layers = 2
seed = 7

# the data, shuffled and split between train and test sets
(X, y), (X_test, y_test) = mnist.load_data()

X = X.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X = np.concatenate((X, X_test), axis=0)
X = X.astype('float32')
#X_test = X_test.astype('float32')
X /= 255
#X_test /= 255
print(X.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y = np.concatenate((y, y_test), axis=0)
y = np_utils.to_categorical(Y, nb_classes)
#Y_test = np_utils.to_categorical(y_test, nb_classes)

inputs = Input(shape=(784,))
x = inputs
#x = Dense(64, activation='relu')(inputs)
for layers in range(n_hidden_layers):
    x = Dense(64)(x)
    x = PReLU()(x) # Non-linearity
    x = Dropout(rate=0.2)(x)

predictions = Dense(nb_classes, activation='softmax')(x)
    
model = Model(inputs=inputs, outputs=predictions)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)
model.compile(optimizer='adam',#adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

#kfold = StratifiedKFold(n_splits=7, shuffle=True, random_state=seed)
#score = cross_val_score(KerasClassifier(build_fn=model, epochs=nb_epoch, batch_size=batch_size, verbose=0), X, y, cv=kfold)

#print("Score: %.2f%% (%.2f%%)" % (score.mean()*100, score.std()*100))
scores = []
fold = 0
k_fold = StratifiedKFold(n_splits=7, shuffle=True, random_state=seed)
for train, test in k_fold.split(X, Y):
    model.fit(X[train], y[train],
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=0, validation_data=(X[test], y[test]))
    score = model.evaluate(X[test], y[test], verbose=0)
    scores.append(score[1])
    fold += 1
    print('Score - fold', fold, ":", score[1], "--", len(test), "indices")

print('Average score:', np.mean(scores))

#history = model.fit(X, y,
#                    batch_size=batch_size, nb_epoch=nb_epoch,
#                    verbose=1, validation_data=(X_test, Y_test))
#score = model.evaluate(X_test, Y_test, verbose=0)
#
#print('Test score:', score)
#print('Test accuracy:', score)