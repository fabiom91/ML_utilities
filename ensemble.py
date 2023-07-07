import numpy as np
import pandas as pd

class Ensemble():
    '''
    Given a list of models, it creates an ensemble.
    It uses a weighted or unweighted majority vote to output the ensemble predictions.

    Arguments:
    - models: a list of models (they have to use the same features);
    - weights [optional]: a list of weights to assign to each model for the majority vote;

    Usage:
    ensemble = Ensemble(models,weights)
    y_pred = ensemble.majority_vote(X)
    '''

    def __init__(self,models, weights = None):
        self.models = np.array(models)
        if not weights:
            weights = [1] * len(self.models)
        self.weights = np.array(weights)

    def majority_vote(self,data_to_predict):
        predictions = np.array([model.predict_proba(data_to_predict)[:,1] for model in self.models])
        df = pd.DataFrame(predictions)
        predictions = np.average(a=df, weights=self.weights, axis=0)
        return predictions
