import os
import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dropout, Flatten, Dense, Lambda)

from .functions_network import split_mammals, convert_cond_probabilities


def predict_probabilities(bottleneck_features, weights_file_path):
    """Function to predict probabilities using extracted bottleneck features.

    Parameters
    ----------
    bottleneck_features : np.ndarray
        bottleneck features
    weights_file_path : str | Path
        path to folder containing weights of the top model (L. Hoebeke)

    Returns
    -------
    predictions : np.ndarray
        Prediction probabilities
    """

    # Number of output classes in the classification tree (hardcoded to handle L. Hoebeke setup)
    cond_classes = 5+4+9

    # Trained top
    top_model = Sequential()
    top_model.add(Flatten(input_shape=(1, 2, 2048)))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.50))
    top_model.add(Dense(cond_classes, activation='sigmoid'))
    top_model.add(Lambda(split_mammals,name='cond_layer'))
    top_model.add(Lambda(convert_cond_probabilities))
    top_model.load_weights(os.path.join(weights_file_path, 'resnet_bottleneck_weights.h5'),
                           by_name=False)

    predictions = top_model.predict(bottleneck_features)
    return predictions
