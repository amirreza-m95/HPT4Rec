import sys
sys.path.append("../../")
import json
import os
import surprise
import papermill as pm
import pandas as pd
import shutil
import subprocess
import yaml
import pkg_resources
from tempfile import TemporaryDirectory

import sys
sys.path.append('./recommenders')
import reco_utils
from reco_utils.common.timer import Timer
from reco_utils.dataset import movielens
from reco_utils.dataset.python_splitters import python_random_split
from reco_utils.evaluation.python_evaluation import rmse, precision_at_k, ndcg_at_k
from reco_utils.tuning.nni.nni_utils import (check_experiment_status, check_stopped, check_metrics_written, get_trials,
                                      stop_nni, start_nni)
from reco_utils.recommender.surprise.surprise_utils import predict, compute_ranking_predictions

import lib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', default='./../data/RecSys_Dataset_After', type=str)
parser.add_argument('--train_data', default='recSys15TrainOnly.txt', type=str)
parser.add_argument('--valid_data', default='recSys15Valid.txt', type=str)

print("System version: {}".format(sys.version))
print("Surprise version: {}".format(surprise.__version__))
print("NNI version: {}".format(pkg_resources.get_distribution("nni").version))

tmp_dir = TemporaryDirectory()
TMP_DIR = tmp_dir.name

# time (in seconds) to wait for each tuning experiment to complete
WAITING_TIME = 3600
MAX_RETRIES = 40 # it is recommended to have MAX_RETRIES>=4*MAX_TRIAL_NUM

tmp_dir = TemporaryDirectory()

args = parser.parse_args()

print("Loading training data from Storage {}".format(os.path.join(args.data_folder, args.train_data)))
train = lib.Dataset(os.path.join(args.data_folder, args.train_data), n_sample=100)
print(len(train.df))
print("Loading validation data from Storage {}".format(os.path.join(args.data_folder, args.valid_data)))
validation = lib.Dataset(os.path.join(args.data_folder, args.valid_data), itemmap=train.itemmap)


print("Log Directory: {}".format(os.path.join(TMP_DIR, "experiments")))
LOG_DIR = os.path.join(TMP_DIR, "experiments")
os.makedirs(LOG_DIR, exist_ok=True)
print("Data Directory: {}".format(os.path.join(TMP_DIR, "data")))
DATA_DIR = os.path.join(TMP_DIR, "data") 
os.makedirs(DATA_DIR, exist_ok=True)


print("Building Pickle of train data{}".format(os.path.join(args.data_folder, args.train_data)))
TRAIN_FILE_NAME = "GRU4Rec" + "_train.pkl"
train.df.to_pickle(os.path.join(DATA_DIR, TRAIN_FILE_NAME))
print("Building Pickle valid data {}".format(os.path.join(args.data_folder, args.valid_data)))
VAL_FILE_NAME = "GRU4Rec" + "_val.pkl"
validation.df.to_pickle(os.path.join(DATA_DIR, VAL_FILE_NAME))

### 3. Prepare Hyperparameter Tuning

EXP_NAME = "GRU4Rec Hyperparameter Optimization using AutoML "
PRIMARY_METRIC = "recall" 
RATING_METRICS = ["loss", "recall", "mrr"] 
RANKING_METRICS = ["precision_at_k", "ndcg_at_k"] 

script_params = " ".join([
    "--test_param", " 10",
    "--rating-metrics", " ".join(RATING_METRICS),
    "--ranking-metrics", " ".join(RANKING_METRICS),
    "--primary-metric", PRIMARY_METRIC
])


hyper_params = {
    'num_layers' : { "_type": "choice", "_value": [1,2,3]},
    'hidden_size': { "_type": "randint", "_value": [50,1000] },
    'batch_size' : { "_type": "choice", "_value": [32,64,128] },
    'lr' : {"_type":"uniform","_value":[0.01,0.05]}
   # 'loss_type' : {"_type":"choice","_value":["TOP1-max"]}
   # 'dropout_hidden' : {"_type":"uniform","_value":[0.01,0.05]}
 }

with open(os.path.join(TMP_DIR, 'search_space_gru.json'), 'w') as fp:
    json.dump(hyper_params, fp)

"""We also create a yaml file for the configuration of the trials and the tuning algorithm to be used (in this experiment we use the [TPE tuner](https://nni.readthedocs.io/en/latest/hyperoptTuner.html))."""

config = {
    "authorName": "AmirReza Mohammadi",
    "experimentName": "GRU4Rec Optimization using NNI",
    "trialConcurrency": 2,
    "maxExecDuration": "30h",
    "maxTrialNum": 30,
    "trainingServicePlatform": "local",
    # The path to Search Space
    "searchSpacePath": "search_space_gru.json",
    "useAnnotation": False,

     "logDir": LOG_DIR,

    "logLevel":"debug",
    "localConfig": {
        "gpuIndices": "0",
        "useActiveGpu": True,
        },
    "tuner": {
        "builtinTunerName": "TPE",
        "classArgs": {
            #choice: maximize, minimize
            "optimize_mode": "maximize"
        },
        "gpuIndices": "0"
       # "includeIntermediateResults": True
    },
    # The path and the running command of trial there is a problem with path specified for python so i removed it because i have added python to my sys env in windows  sys.prefix + "/bin/
    "trial":  {
      "command": "python experiment.py"+ " " + script_params,
      "codeDir": os.path.join(os.path.split(os.path.abspath(reco_utils.__file__))[0], "tuning", "nni"),
      "gpuNum": 1
    }
}
 
with open(os.path.join(TMP_DIR, "config_gru.yml"), "w") as fp:
    fp.write(yaml.dump(config, default_flow_style=False))

stop_nni() 

config_path = os.path.join(TMP_DIR, 'config_gru.yml')
nni_env = os.environ.copy()
nni_env['PATH'] = ':' + nni_env['PATH']
proc = subprocess.run(['nnictl', 'create', '--config', config_path], env=nni_env)
if proc.returncode != 0:
    raise RuntimeError("'nnictl create' failed with code %d" % proc.returncode)

with Timer() as time_tpe:
    check_experiment_status(wait=WAITING_TIME, max_retries=MAX_RETRIES)
print("Finished experiments")

with Timer() as time_tpe:
    check_experiment_status(wait=WAITING_TIME, max_retries=MAX_RETRIES)


def compute_test_results(svd):
    test_results = {}
    predictions = predict(svd, test, usercol="userID", itemcol="itemID")
    for metric in RATING_METRICS:
        test_results[metric] = eval(metric)(test, predictions)

    all_predictions = compute_ranking_predictions(svd, train, usercol="userID", itemcol="itemID", remove_seen=REMOVE_SEEN)
    for metric in RANKING_METRICS:
        test_results[metric] = eval(metric)(test, all_predictions, col_prediction='prediction', k=K)
    return test_results



"""### 6. More Tuning Algorithms
We now apply other tuning algorithms supported by NNI to the same problem. For details about these tuners, see the [NNI docs.](https://nni.readthedocs.io/en/latest/tuners.html#)
The only change needed is in the relevant entry in the configuration file.

In summary, the tuners used in this notebook are the following:
- Tree-structured Parzen Estimator (TPE), within the Sequential Model-Based Optimization (SMBO) framework,
- SMAC, also an instance of SMBO,
- Hyperband
- Metis, an implementation of Bayesian optimization with Gaussian Processes
- a Naive Evolutionary algorithm
- an Annealing method for sampling, and  
- plain Random Search as a baseline.  

"""