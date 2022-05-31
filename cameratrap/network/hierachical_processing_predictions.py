import pandas as pd
import os

from .functions_output import (hierarchical_predictions,
                               bottom_hierarchical_prediction,
                               top_predictions)


def probabilities_to_classification(prediction_probabilities):

    """This function determines the hierchical classification of the sequences, using the top-k method.

    Parameters
    ----------
    predictions_probabilities : np.ndarray
        Prediction probabilities

    Returns
    -------
    Hierarchical labels
    """
    prediction_probabilities = pd.DataFrame(prediction_probabilities)
    # Hierarchical classification images
    hierarchy = hierarchical_predictions(prediction_probabilities)

    # Hierarchical classification sequences using the top-k method
    pred_top = top_predictions(prediction_probabilities, hierarchy)

    # Final prediction for every sequence
    pred_top['final_prediction'] = pred_top.apply(bottom_hierarchical_prediction, axis=1)
    return pred_top
