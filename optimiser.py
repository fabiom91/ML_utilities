from skopt.space import Real, Integer, Categorical
from skopt import gp_minimize
from skopt.plots import plot_convergence
import numpy as np
from sklearn.model_selection import cross_val_score
from functools import partial
import warnings
from sklearn.exceptions import ConvergenceWarning

class Optimiser():
    scorer = 'roc_auc'
    
    def __init__(self,X,y,clf,params,verbose=True):
        warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
        self.clf = clf
        self.params_names = list(params.keys())
        self.space = self._format_space(params)
        self.X = np.nan_to_num(X)
        self.y = np.nan_to_num(y)
        self.verbose = verbose

    def _format_space(self,parameters):
        space = []
        for key,value in parameters.items():
            if type(value[0]) == int:
                if None in value:
                    newvalue = [x for x in value if x != None]
                    integer_values = list(range(min(newvalue), max(newvalue) + 1))
                    space.append(Categorical(categories=[*integer_values, None], name=key))
                else:
                    space.append(Integer(min(value),max(value),name=key))
            elif type(value[0]) == float:
                if None in value:
                    newvalue = [x for x in value if x != None]
                    integer_values = list(range(min(newvalue), max(newvalue) + 1))
                    space.append(Categorical(categories=[*integer_values, None], name=key))
                else:
                    space.append(Real(min(value),max(value),name=key))
            else:
                space.append(Categorical(value,name=key))
        return space
        
    @staticmethod
    def objective(params, X, y, clf, scorer, params_names):
        params = dict(zip(params_names, params))
        if clf.__class__.__name__ == 'LogisticRegression':
            params['max_iter'] = 1000
        clf.set_params(**params)
        return np.mean(cross_val_score(clf, X, y, cv=5, n_jobs=-1, scoring=scorer))
    
    def get_score(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            objective_func = partial(self.objective, X=self.X, y=self.y, clf=self.clf,
                                     scorer=self.scorer, params_names = self.params_names)
            res_gp = gp_minimize(objective_func, self.space, n_calls=50, random_state=0)
            if self.verbose:
                print("Best score=%.4f" % res_gp.fun)
            self.res_gp = res_gp
            return self.clf, dict(zip(self.params_names,self.res_gp.x))
        
    def plot(self):
        plot_convergence(self.res_gp)