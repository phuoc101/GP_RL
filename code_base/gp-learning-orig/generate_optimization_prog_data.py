from GPModelBatch import GPModel
import configs

if 0:  # GPU, 1000 Trajectories
    cf = configs.train_configs()
    cf = {
        **cf,
        "N_trajs": 1000,
        "controllerType": "NN",
        "NNlayers": [8, 8],
        "Tf": 30,
        "optimLogPath": "results/optimNN16GPU1kTrajs/",
        "optimopts": {
            "optimizer": "Adam",
            "trials": 16,  # number of restarts
            "max_iter": 200,
        },
    }

    modelv = GPModel(**cf)
    modelv.optimize_policy()
if 0:  # GPU with 100 trajectories
    cf = configs.train_configs()
    cf = {
        **cf,
        "controllerType": "NN",
        "NNlayers": [8, 8],
        "Tf": 30,
        "optimLogPath": "results/optimNN16GPU/",
        "optimopts": {
            "optimizer": "Adam",
            "trials": 16,  # number of restarts
            "max_iter": 200,
        },
    }

    modelv = GPModel(**cf)
    modelv.optimize_policy()
if 0:
    # for CPU Batch
    cf = configs.train_configs()
    cf = {
        **cf,
        "controllerType": "NN",
        "NNlayers": [8, 8],
        "Tf": 30,
        "optimLogPath": "results/optimNN16CPU/",
        "optimopts": {
            "optimizer": "Adam",
            "trials": 16,  # number of restarts
            "max_iter": 100,
        },
        "Force_CPU": True,
    }

    modelv = GPModel(**cf)
    modelv.optimize_policy()
if 1:
    # for CPU sequentially
    cf = configs.train_configs()
    cf = {
        **cf,
        "controllerType": "NN",
        "NNlayers": [8, 8],
        "Tf": 30,
        "optimLogPath": "results/optimNN16CPUSeq/",
        "optimopts": {
            "optimizer": "Adam",
            "trials": 16,  # number of restarts
            "max_iter": 450,
        },
        "Force_CPU": True,
    }

    modelv = GPModel(**cf)
    modelv.optimize_policy_sequentially()
if 0:

    # for CPU Batch
    cf = configs.train_configs()
    cf = {
        **cf,
        "controllerType": "NN",
        "NNlayers": [8, 8],
        "Tf": 30,
        "optimLogPath": "./results/optimNN-mid/",
        "optimopts": {
            "optimizer": "Adam",
            "trials": 4,  # number of restarts
            "max_iter": 4,
        },
        "Force_CPU": False,
    }

    modelv = GPModel(**cf)
    modelv.optimize_policy()
if 0:
    # for CPU batch without LOVE
    cf = configs.train_configs()
    cf = {
        **cf,
        "controllerType": "NN",
        "NNlayers": [8, 8],
        "Tf": 30,
        "optimLogPath": "./results/optimNN-noLOVE/",
        "optimopts": {
            "optimizer": "Adam",
            "trials": 5,  # number of restarts
            "max_iter": 5,
        },
        "N_trajs": 1,
        "Force_CPU": True,
    }

    modelv = GPModel(**cf)
    modelv.predict = modelv.predict_noLOVE  # manually remove LOVE
    modelv.optimize_policy_sequentially()
