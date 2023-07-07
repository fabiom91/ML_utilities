from optimiser import Optimiser


def hypertuning(model,X,y,parameters):
    '''
    Uses skopt and gp_minimize to get the best parameters for
    a given model and data to maximise AUC score.
    '''
    opt = Optimiser(X,y,model,parameters,verbose=False)
    _, best_params = opt.get_score()
    return best_params