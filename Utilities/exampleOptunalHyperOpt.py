# imports
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.utils import shuffle

import optuna
import matplotlib.pyplot as plt


# function to return values at x pts
def evalFunction(pts):
    forrester1D = lambda x: (6 * x - 2) * 2 * np.sin(12 * x - 4)
    values = [forrester1D(pt) for pt in pts]
    return np.array(values)


# optuna objective
def objective(trial, x_pts, y_pts):
    # define hyperparas to tune
    optunaParams = {"n_estimators": trial.suggest_int("n_estimators", 1, 20),
                    "bootstrap": trial.suggest_categorical("bootstrap", [True, False])
                    }

    # define model
    modelRF = RandomForestRegressor(n_estimators=optunaParams["n_estimators"],
                                    bootstrap=optunaParams["bootstrap"])

    # reshape needed since only 1 feature used
    x_pts = x_pts.reshape(-1, 1)

    # Use a 4-fold CV, split data evenly
    no_folds = 4
    x_pts_CV = np.split(x_pts, no_folds)
    y_pts_CV = np.split(y_pts, no_folds)

    # set R2 sum to zero, iterate through CV folds
    sumR2 = 0
    for i in range(no_folds):
        # fit model onto 1 fold data
        modelRF.fit(x_pts_CV[i], y_pts_CV[i])

        # concate rest of data, remove fold data for train
        x_pts_val = np.delete(x_pts_CV, i, 0)
        x_pts_val = np.concatenate(x_pts_val)
        y_pts_val = np.delete(y_pts_CV, i, 0)
        y_pts_val = np.concatenate(y_pts_val)

        # validation prediction
        y_pred = modelRF.predict(x_pts_val)
        intermed_R2 = r2_score(y_pts_val, y_pred)
        #print(f"Trial no. {str(trial.number)}, fold no. {i}, R2: {intermed_R2}")

        # Handle pruning -> e.g. for k-fold cross validation
        #  evaluates val score after each fold and
        #  stops (e.g. prunes) trial if it is not good
        #  compared to history
        trial.report(intermed_R2, i)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # sum up all R2 values from each fold
        sumR2 += intermed_R2

    # final R2 is average of folds
    r2 = sumR2 / no_folds

    return r2


def main():
    # create pts along x, get y values and shuffle for CV
    x_pts = np.arange(0., 1., 0.01)
    y_pts = evalFunction(x_pts)
    x_pts_shuffled, y_pts_shuffled = shuffle(x_pts, y_pts, random_state=4)

    # no. of trials for optimization
    no_trials = 10

    # create a study to maximize objective, use pruner
    study = optuna.create_study(directions=["maximize"], sampler=optuna.samplers.TPESampler(),
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, x_pts_shuffled, y_pts_shuffled), n_trials=no_trials)

    # show some plots of the trials in the browser
    fig_importancesVal = optuna.visualization.plot_param_importances(study, target=lambda t: t.values[0], target_name="valR2")
    fig_historyVal = optuna.visualization.plot_optimization_history(study, target=lambda t: t.values[0], target_name="valR2")
    fig_importancesVal.show()
    fig_historyVal.show()

    # Print info of best trial
    best_trial = study.best_trial
    print("-------------------------------")
    print("Best trial:")
    print("-------------------------------")
    for key, value in best_trial.params.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()