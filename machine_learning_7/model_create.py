from keras import layers
from keras import models
import keras


def model_create(filter_size, sliding_window_size, channels):
    model = models.Sequential()


    rmsprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0)
    model.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model





model = model_create(6,60,3)