from sklearn.model_selection import LeaveOneOut
import numpy as np
from hypertuning_GridSearchCV import hypertuning
import joblib
from datetime import datetime
import os
import shutil
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from ensemble import Ensemble


class LOO():
    '''
    Implementation of Sklearn LeaveOneOut Cross Validation
    with optional Hyperparameters tuning.

    Arguments:
    - X: the features X of your dataset as a numpy ndarray;
    - y: the target feature of your dataset as a numpy array;
    - model: the sklearn model you want to apply (e.g. LogisticRegression);
    - hypers [optional]: the hyperparameters to be tuned:
            e.g. {'C':[0.001,0.01,0.1,1,10,100,1000]}

    It creates a random hidden validation dataset to evaluate ensemble performances

    Usage:
    loo = LOO(X,y,model,hypers)
    y_true, y_pred = loo.ensemble()

    The code above returns the random hidden validation labels and the
    predicted label so that a metric can be easily applied to evaluate
    the performances
    '''

    models_folder = 'loo_models/'
    models = []
    
    def __init__(self,X,y,model,hypers=None):
        self.model = model
        self.hypers = hypers
        self.X, self.hidden_X, self.y, self.hidden_y = train_test_split(
            X, y, test_size=0.2, random_state=42)
        if os.path.exists(self.models_folder):
            shutil.rmtree(self.models_folder)
        os.mkdir(self.models_folder)
        self._loo_loop()

    def _loo_loop(self):
        self.pred_proba = np.zeros((self.y.shape[0],2))
        self.weights = np.zeros(self.y.shape)
        self.hypers_loo = {}
        loo = LeaveOneOut()
        for i, (train_index,test_index) in tqdm(enumerate(loo.split(self.X))):
            X_train, y_train = self.X[train_index], self.y[train_index]
            X_test, y_test = self.X[test_index], self.y[test_index]
            model = self.model
            if self.hypers:
                params = hypertuning(model,X_train,y_train,self.hypers)
                model = model(**params)
            model.fit(X_train,y_train)
            self.pred_proba[i] = model.predict_proba(X_test)
            self.weights[i] = 1-abs(y_test-self.pred_proba[i][:,1])
            now = int(round(datetime.now().timestamp()))
            self.models.append(model)
            joblib.dump(model,'%s_%i_%i.joblib' % (self.models_folder,i,now))

    def ensemble(self):
        if len(self.models) == 0:
            self.models = [joblib.load(x) for x in os.listdir(self.models_folder) if x.endswith('.joblib')]
        ens = Ensemble(self.models,self.weights)
        ens_pred = ens.majority_vote(self.hidden_X)
        return self.hidden_y, ens_pred

    
    