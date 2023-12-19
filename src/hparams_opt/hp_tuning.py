import optuna

print(optuna.__version__)


# # define function to be optimized
# def objective(trial):
#     x = trial.suggest_float("x", -10, 10)
#     return (x - 2) ** 2


# # instantiate a study and start optimization
# study = optuna.create_study()
# study.optimize(objective, n_trials=100)

# # retrieve the best observed parameters
# best_params = study.best_params
# found_x = best_params["x"]

# print("Found x: {}, (x - 2)^2: {}".format(found_x, (found_x - 2) ** 2))
# # >> Found x: 1.9972882888553833, (x - 2)^2: 7.353377331838183e-06


# import logging
# import pickle
# import sys

# import optuna


# def objective(trial):
#     x = trial.suggest_float("x", -10, 10)
#     return (x - 2) ** 2


# ######### New Study

# # Add stream handler of stdout to show the messages
# optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

# # Create a new study and start optimizing
# study_name = "example-study"  # Unique identifier of the study.
# storage_name = "sqlite:///{}.db".format(study_name)
# study = optuna.create_study(study_name=study_name, storage=storage_name)
# study.optimize(objective, n_trials=3)

# # Save the sampler with pickle to be loaded later.
# with open("sampler.pkl", "wb") as f:
#     pickle.dump(study.sampler, f)

# ######### Resume Study (with sampler)
# restored_sampler = pickle.load(open("sampler.pkl", "rb"))
# study = optuna.create_study(
#     study_name=study_name, storage=storage_name, load_if_exists=True, sampler=restored_sampler
# )
# study.optimize(objective, n_trials=3)
