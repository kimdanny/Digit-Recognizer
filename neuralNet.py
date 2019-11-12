import pandas as pd
import numpy as np
from tensorflow import keras
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data.test.csv")
X_train = train.drop(labels=["label"], axis=1)
Y_train = train['label']
del train

# normalise
X_train = X_train / 255.0
test = test / 255.0

# reshape
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

Y_train = to_categorical(Y_train, num_classes = 10)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)

#######
# CNN #
#######
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])


learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

epochs = 30
batch_size = 86

history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs,
                    validation_data = (X_val, Y_val), verbose = 2)

result = model.predict(test)
result = np.argmax(result, axis = 1)
result = pd.Series(result, name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"), result],axis = 1)

submission.to_csv("submission.csv",index=False)


# if __name__=="__main__":
#     print(train)
#     print(train.shape)
#     print(Y_train)