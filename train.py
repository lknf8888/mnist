import os

import keras
import numpy as np
from keras.datasets import mnist
from keras.layers import Dropout, BatchNormalization, LeakyReLU, Dense, Input, Activation
from keras.models import Model
from keras.utils.np_utils import to_categorical

import logging
from bedrock_client.bedrock.api import BedrockApi

def build_model():
    x = Input((28 * 28,), name="x")
    hidden_dim = int(os.environ.get("HIDDEN_DIM", 512))
    h = x
    h = Dense(hidden_dim)(h)
    h = BatchNormalization()(h)
    h = LeakyReLU(0.2)(h)
    h = Dropout(float(os.environ.get("DROPOUT_1", 0.5)))(h)
    h = Dense(hidden_dim // 2)(h)
    h = BatchNormalization()(h)
    h = LeakyReLU(0.2)(h)
    h = Dropout(float(os.environ.get("DROPOUT_2", 0.5)))(h)
    h = Dense(10)(h)
    h = Activation('softmax')(h)
    m = Model(x, h)
    m.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
    return m


def mnist_process(x, y):
    return x.astype(np.float32).reshape((x.shape[0], -1)) / 255.0, to_categorical(y, 10)


def mnist_data():
    data = mnist.load_data()
    return [mnist_process(*d) for d in data]


def mnist_model(verbose=1, callbacks=[]):
    m = build_model()
    (xtrain, ytrain), (xtest, ytest) = mnist_data()
    if int(keras.__version__.split(".")[0]) == 2:
        training_log = m.fit(
            xtrain,
            ytrain,
            validation_data=(xtest, ytest),
            epochs=int(os.environ.get("N_EPOCH", 10)),
            batch_size=int(os.environ.get("BATCH_SIZE", 32)),
            verbose=verbose,
            callbacks=callbacks
        )
    else:
        training_log = m.fit(
            xtrain,
            ytrain,
            validation_data=(xtest, ytest),
            nb_epoch=int(os.environ.get("N_EPOCH", 10)),
            batch_size=int(os.environ.get("BATCH_SIZE", 32)),
            verbose=verbose,
            callbacks=callbacks
        )
    
    logger = logging.getLogger(__name__)
    bedrock = BedrockApi(logger)

    bedrock.log_metric("Accuracy", training_log.history['accuracy'][-1])
    bedrock.log_metric("Loss", training_log.history['loss'][-1])
    
    bedrock.log_metric("Validation Accuracy", training_log.history['val_accuracy'][-1])
    bedrock.log_metric("Validation Loss", training_log.history['val_loss'][-1])
    
    # serialize model to JSON
    model_json = m.to_json()
    with open("/artefact/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    m.save_weights("/artefact/model.h5")
    print("Saved model to disk")


if __name__ == "__main__":
    mnist_model()
