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
#NUM_EPOCHS = 2
#MAX_TRIAL_NUM = 10 

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

# TEST_FILE_NAME = "movielens_" + MOVIELENS_DATA_SIZE + "_test.pkl"
# test.to_pickle(os.path.join(DATA_DIR, TEST_FILE_NAME))

### 3. Prepare Hyperparameter Tuning

EXP_NAME = "GRU4Rec Hyperparameter Optimization using AutoML "
PRIMARY_METRIC = "recall" #check
RATING_METRICS = ["loss", "recall", "mrr"] #check
RANKING_METRICS = ["precision_at_k", "ndcg_at_k"] #check 
USERCOL = "userID" #del
ITEMCOL = "itemID" #del
REMOVE_SEEN = True #del
K = 10 #del
RANDOM_STATE = 42 #del
VERBOSE = True #del
BIASED = True #del

script_params = " ".join([
   # "--datastore", DATA_DIR,
    #"--train-datapath", TRAIN_FILE_NAME,
   # "--validation-datapath", VAL_FILE_NAME,
    "--test_param", " 10",
    "--rating-metrics", " ".join(RATING_METRICS),
    "--ranking-metrics", " ".join(RANKING_METRICS),
    # "--usercol", USERCOL, #del
    # "--itemcol", ITEMCOL, #del # "--k", str(K), #del # "--random-state", str(RANDOM_STATE), # "--epochs", str(NUM_EPOCHS),
    "--primary-metric", PRIMARY_METRIC
])

# if BIASED:
#     script_params += " --biased"
# if VERBOSE:
#     script_params += " --verbose"
# if REMOVE_SEEN:
#     script_params += " --remove-seen"

# hyperparameters search space
# We do not set 'lr_all' and 'reg_all' because they will be overriden by the other lr_ and reg_ parameters

# hyper_params = {
#     'hidden_size': {"_type": "choice", "_value": [ 100, 150, 200, 250]}
# }
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
    #testing
    #"multiThread": True,
#    "logDir":"/home/isg/Documents/AmirReza/grunni",
     "logDir": LOG_DIR,
   #  "assessor": {
    #     "builtinAssessorName": "Medianstop"
#
 #       },

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

### 4. Execute NNI Trials


# Make sure that there is no experiment running
stop_nni() #it couldn't find the specified file so i just removed it :)

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
"""### 5. Show Results

The trial with the best metric and the corresponding metrics and hyperparameters can also be read from the Web UI

![](https://recodatasets.blob.core.windows.net/images/nni4.png)

or from the JSON file created by the training script. Below, we do this programmatically using [nni_utils.py](../../reco_utils/nni/nni_utils.py)
"""
'''
trials, best_metrics, best_params, best_trial_path = get_trials('maximize')

print( best_metrics)

print(best_params)

print(best_trial_path)
'''
"""This directory path is where info about the trial can be found, including logs, parameters and the model that was learned. To evaluate the metrics on the test data, we get the SVD model that was saved as `model.dump` in the training script."""

#svd = surprise.dump.load(os.path.join(best_trial_path, "model.dump"))[1]

"""The following function computes all the metrics given an SVD model."""

def compute_test_results(svd):
    test_results = {}
    predictions = predict(svd, test, usercol="userID", itemcol="itemID")
    for metric in RATING_METRICS:
        test_results[metric] = eval(metric)(test, predictions)

    all_predictions = compute_ranking_predictions(svd, train, usercol="userID", itemcol="itemID", remove_seen=REMOVE_SEEN)
    for metric in RANKING_METRICS:
        test_results[metric] = eval(metric)(test, all_predictions, col_prediction='prediction', k=K)
    return test_results

#test_results_tpe = compute_test_results(svd)
#print(test_results_tpe)

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

For more details and references to the relevant literature, see the [NNI github](https://github.com/Microsoft/nni/blob/master/docs/en_US/Builtin_Tuner.md).
"""
'''
# Random search
config['tuner']['builtinTunerName'] = 'Random'
if 'classArgs' in config['tuner']:
    config['tuner'].pop('classArgs')
    
with open(config_path, 'w') as fp:
    fp.write(yaml.dump(config, default_flow_style=False))

stop_nni()
with Timer() as time_random:
    start_nni(config_path, wait=WAITING_TIME, max_retries=MAX_RETRIES)

check_metrics_written(wait=WAITING_TIME, max_retries=MAX_RETRIES)
svd = surprise.dump.load(os.path.join(get_trials('maximize')[3], "model.dump"))[1]
test_results_random = compute_test_results(svd)

# Annealing
config['tuner']['builtinTunerName'] = 'Anneal'
if 'classArgs' not in config['tuner']:
    config['tuner']['classArgs'] = {'optimize_mode': 'maximize'}
else:
    config['tuner']['classArgs']['optimize_mode'] = 'maximize'
    
with open(config_path, 'w') as fp:
    fp.write(yaml.dump(config, default_flow_style=False))

stop_nni()
with Timer() as time_anneal:
    start_nni(config_path, wait=WAITING_TIME, max_retries=MAX_RETRIES)

check_metrics_written(wait=WAITING_TIME, max_retries=MAX_RETRIES)
svd = surprise.dump.load(os.path.join(get_trials('maximize')[3], "model.dump"))[1]
test_results_anneal = compute_test_results(svd)

# Naive evolutionary search
config['tuner']['builtinTunerName'] = 'Evolution'
with open(config_path, 'w') as fp:
    fp.write(yaml.dump(config, default_flow_style=False))

stop_nni()
with Timer() as time_evolution:
    start_nni(config_path, wait=WAITING_TIME, max_retries=MAX_RETRIES)

check_metrics_written(wait=WAITING_TIME, max_retries=MAX_RETRIES)
svd = surprise.dump.load(os.path.join(get_trials('maximize')[3], "model.dump"))[1]
test_results_evolution = compute_test_results(svd)

"""The SMAC tuner requires to have been installed with the following command <br>
`nnictl package install --name=SMAC`
"""

# SMAC
config['tuner']['builtinTunerName'] = 'SMAC'
with open(config_path, 'w') as fp:
    fp.write(yaml.dump(config, default_flow_style=False))

# Check if installed
proc = subprocess.run(['nnictl', 'package', 'show'], stdout=subprocess.PIPE)
if proc.returncode != 0:
    raise RuntimeError("'nnictl package show' failed with code %d" % proc.returncode)
if 'SMAC' not in proc.stdout.decode().strip().split():
    proc = subprocess.run(['nnictl', 'package', 'install', '--name=SMAC'])
    if proc.returncode != 0:
        raise RuntimeError("'nnictl package install' failed with code %d" % proc.returncode)

# Skipping SMAC optimization for now
# stop_nni()
with Timer() as time_smac:
#    start_nni(config_path, wait=WAITING_TIME, max_retries=MAX_RETRIES)
    pass

#check_metrics_written()
#svd = surprise.dump.load(os.path.join(get_trials('maximize')[3], "model.dump"))[1]
#test_results_smac = compute_test_results(svd)

# Metis
config['tuner']['builtinTunerName'] = 'MetisTuner'
with open(config_path, 'w') as fp:
    fp.write(yaml.dump(config, default_flow_style=False))

stop_nni()
with Timer() as time_metis:
    start_nni(config_path, wait=WAITING_TIME, max_retries=MAX_RETRIES)

check_metrics_written()
svd = surprise.dump.load(os.path.join(get_trials('maximize')[3], "model.dump"))[1]
test_results_metis = compute_test_results(svd)

"""Hyperband follows a different style of configuration from other tuners. See [the NNI documentation](https://nni.readthedocs.io/en/latest/hyperbandAdvisor.html). Note that the [training script](../../reco_utils/nni/svd_training.py) needs to be adjusted as well, since each Hyperband trial receives an additional parameter `STEPS`, which corresponds to the resource allocation _r<sub>i</sub>_ in the [Hyperband algorithm](https://arxiv.org/pdf/1603.06560.pdf). In this example, we used `STEPS` in combination with `R` to determine the number of epochs that SVD will run for in every trial."""

# Hyperband
config['advisor'] = {
  'builtinAdvisorName': 'Hyperband',
  'classArgs': {
    'R': NUM_EPOCHS,
    'eta': 3,
    'optimize_mode': 'maximize'
  }
}
config.pop('tuner')
with open(config_path, 'w') as fp:
    fp.write(yaml.dump(config, default_flow_style=False))

stop_nni()
with Timer() as time_hyperband:
    start_nni(config_path, wait=WAITING_TIME, max_retries=MAX_RETRIES)

check_metrics_written()
svd = surprise.dump.load(os.path.join(get_trials('maximize')[3], "model.dump"))[1]
test_results_hyperband = compute_test_results(svd)

test_results_tpe.update({'time': time_tpe.interval})
test_results_random.update({'time': time_random.interval})
test_results_anneal.update({'time': time_anneal.interval})
test_results_evolution.update({'time': time_evolution.interval})
#test_results_smac.update({'time': time_smac.interval})
test_results_metis.update({'time': time_metis.interval})
test_results_hyperband.update({'time': time_hyperband.interval})

algos = ["TPE", 
         "Random Search", 
         "Annealing", 
         "Evolution", 
         #"SMAC", 
         "Metis", 
         "Hyperband"]
res_df = pd.DataFrame(index=algos,
                      data=[res for res in [test_results_tpe, 
                                            test_results_random, 
                                            test_results_anneal, 
                                            test_results_evolution, 
                                            #test_results_smac, 
                                            test_results_metis, 
                                            test_results_hyperband]] 
                     )

res_df.sort_values(by="precision_at_k", ascending=False).round(3)

"""As we see in the table above, _TPE_ performs best with respect to the primary metric (precision@10) that all the tuners optimized. Also the best NDCG@10 is obtained for TPE and correlates well with precision@10. RMSE on the other hand does not correlate well and is not optimized for TPE, since finding the top k recommendations in the right order is a different task from predicting ratings (high and low) accurately.     
We have also observed that the above ranking of the tuners is not consistent and may change when trying these experiments multiple times. Since some of these tuners rely heavily on randomized sampling, a larger number of trials is required to get more consistent metrics.
In addition, some of the tuning algorithms themselves come with parameters, which can affect their performance.
"""

# Stop the NNI experiment 
stop_nni()

tmp_dir.cleanup()

"""### 7. Concluding Remarks

We showed how to tune **all** the hyperparameters accepted by Surprise SVD simultaneously, by utilizing the NNI toolkit. 
For example, training and evaluation of a single SVD model takes about 50 seconds on the 100k MovieLens data on a Standard D2_V2 VM. Searching through 100 different combinations of hyperparameters sequentially would take about 80 minutes whereas each of the above experiments took about 10 minutes by exploiting parallelization on a single D16_v3 VM. With NNI, one can take advantage of concurrency and multiple processors on a virtual machine and can use a variety of tuning methods to navigate efficiently through a large space of hyperparameters.<br>
For examples of scaling larger tuning workloads on clusters of machines, see [the notebooks](./README.md) that employ the [Azure Machine Learning service](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-tune-hyperparameters).

### References

* [Matrix factorization algorithms in Surprise](https://surprise.readthedocs.io/en/stable/matrix_factorization.html) 
* [Surprise SVD deep-dive notebook](../02_model/surprise_svd_deep_dive.ipynb)
* [Neural Network Intelligence toolkit](https://github.com/Microsoft/nni)
"""

# ! git clone https://github.com/Microsoft/Recommenders

# ! pip install nni # install nni
# ! mkdir -p nni_repo
# ! git clone https://github.com/microsoft/nni.git nni_repo/nni 

# ! python Recommenders/tools/generate_requirements_txt.py
# # ! pip install -r requirements.txt  # seperated pachages are defined in the end of the notebook

# !pip install --upgrade nni

# ! pip install scikit-surprise

# # these packages have problem installing by pip
# cudatoolkit=10.0
# swig==3.0.12 # line 9
# pytorch-cpu>=1.0.0 # 13
# python==3.6.10

# ! pip install -r requirements.txt

# lightgbm==2.2.1
# bottleneck==1.2.1
# jupyter>=1.0.0
# matplotlib>=2.2.2
# tensorflow-gpu==1.15.2
# pyspark==2.4.3
# azure-mgmt-cosmosdb>=0.8.0
# tensorflow==1.15.2
# lightgbm==2.2.1
# bottleneck==1.2.1
# jupyter>=1.0.0
# matplotlib>=2.2.2
# tensorflow-gpu==1.15.2
# pyspark==2.4.3
# azure-mgmt-cosmosdb>=0.8.0
# tensorflow==1.15.2
# papermill==0.19.1
# azure-cli-core>=2.0.75
# tqdm>=4.31.1
# idna==2.7
# pytorch>=1.0.0
# hyperopt==0.1.1
# pymanopt==0.2.5
# xlearn==0.40a1
# scipy>=1.0.0
# dataclasses>=0.6
# fastparquet>=0.1.6
# mock==2.0.0
# dask>=0.17.1
# scikit-learn>=0.19.1
# pytest>=3.6.4
# nltk>=3.4
# seaborn>=0.8.1
# pip>=19.2
# ipykernel>=4.6.1
# azure-storage-blob<=2.1.0
# pydocumentdb>=2.3.3
# nbconvert==5.5.0
# nvidia-ml-py3>=7.352.0
# transformers==2.5.0
# databricks-cli==0.8.6
# memory-profiler>=0.54.0
# black>=18.6b4
# lightfm>=1.15
# numpy>=1.13.3
# numba>=0.38.1
# locustio==0.11.0
# fastai==1.0.46
# category_encoders>=1.3.0
# pyarrow>=0.8.0
# azureml-sdk[notebooks,tensorboard]==1.0.69
# pandas>=0.23.4,<1.0.0
# nni==1.5
# scikit-surprise>=1.0.6
# cornac>=1.1.2

# ! wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip # download ngrok and unzip it
# ! unzip ngrok-stable-linux-amd64.zip
# ! ./ngrok authtoken 1fOeiGdV5hc0NB7GHdv9deDJ2ky_gYFRUtfukPuWLt8Hayoh

# ! nnictl create --config nni_repo/nni/examples/trials/mnist-pytorch/config.yml --port 5000 &
# get_ipython().system_raw('./ngrok http 5000 &')

# ! curl -s http://localhost:4040/api/tunnels

# ! nnictl stop -a
'''
