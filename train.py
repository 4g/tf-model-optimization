import numpy as np
from tensorflow import keras
from keras import layers, models, optimizers
from keras.applications import mobilenet_v2
from libmot import TFlite


num_classes = 10
input_shape = (32, 32, 3)

def get_data():

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32")/127.5 - 1
    x_test = x_test.astype("float32")/127.5 - 1
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    np.random.seed(23)
    np.random.shuffle(x_train)
    np.random.seed(23)
    np.random.shuffle(y_train)


    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return x_train, y_train, x_test, y_test

def get_model():
    model = mobilenet_v2.MobileNetV2(input_shape=input_shape,
                                          alpha=0.35,
                                          include_top=False,
                                          weights='imagenet')

    x = layers.Conv2D(num_classes, kernel_size=1, padding='same', name='Logits')(model.outputs[0])
    x = layers.Flatten()(x)
    x = layers.Activation(activation='softmax',
                          name='Predictions')(x)

    model = models.Model(inputs=model.inputs, outputs=x)
    model.compile(loss="categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(0.001),
                  metrics=["accuracy"])

    return model

model = get_model()
x_train, y_train, x_test, y_test = get_data()

batch_size = 128
epochs = 10

model.fit(x_train,
          y_train,
          validation_split=0.1,
          batch_size=batch_size,
          epochs=epochs,
          shuffle=True)

keras.models.save_model(model, "original.hdf5", include_optimizer=False)
TFlite.save(model, "original.tflite")

from libmot import Prune as Mot

model = Mot.modify(model)
callbacks = Mot.callbacks("logs/")

batch_size = 128
epochs = 10

model.fit(x_train,
          y_train,
          validation_split=0.1,
          batch_size=batch_size,
          epochs=epochs,
          shuffle=True,
          callbacks=callbacks)

Mot.debug(model)

model = Mot.save(model)
keras.models.save_model(model, "pruned.hdf5", include_optimizer=False)
TFlite.save(model, "pruned.tflite")

model.compile(loss="categorical_crossentropy",
              optimizer=keras.optimizers.Adam(0.001),
              metrics=["accuracy"])

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

