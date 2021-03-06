
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Embedding, Flatten, Input, PReLU, Conv1D, MaxPooling1D, LSTM
from keras.datasets import imdb

max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

#print (X_train[0])

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')


inputs = Input(shape=(maxlen,))
x = inputs
x = Embedding(max_features, 128, dropout=0.2)(x)
#x = Dense(64)(x)
#x = PReLU()(x) # Non-linearity
#x = Dropout(rate=0.2)(x)
#x = Conv1D(128, kernel_size=4, activation='relu')(x)
#x = MaxPooling1D(pool_size=2)(x)
x = LSTM(128, dropout_W=0.2, dropout_U=0.2)(x)
#x = Flatten()(x)
x = Dense(1)(x)
predictions = Activation("sigmoid")(x)


model = Model(input=inputs, output=predictions)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

print('Train...')
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=15,
          validation_data=(X_test, y_test))
score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)