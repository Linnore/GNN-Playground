# Config refactor

This repository contains 4 configuration files and two folders, including experiment_config.yaml, mlflow_config.yaml, sampling_config.yaml, system_config.yaml, training_config.yaml, model_collections, and dataset_collections. 

## Details of config files


### Model_collections: 

This folder contains all the configuration files of our training models. All the required parameters are in the corresponding model files. The details please read the README.md file in "./model_collections".

### Dataset_collections: 

This folder contains all the configuration files of banchmark datasets and  our datasets. All the required parameters are in the dataset files respectively. The details please read the README.md file in "./dataset_collections". 

### Experiment_config:

Experiment_config.yaml contains the configuration settings the users must input to run the model. 

```
 "mode": must choose from `train`, `inference`
 "model": must choose from supported models
 "dataset": provide a dataset name 
 "dataset_dir": provide the dataset directory 
```

### System_config:

System_config.yaml contains the configuration settings and parameters of the computer system setting. 
```
"seed": initialize the random number generator
"device": select the device to use
"tqdm": whether show the process bar
"verbose": whether allows the user to write regular expressions that can look nicer and are more readable
"criterion": a callable function to compute loss value, only can choose from: ['loss', 'f1','accuracy']
"num_workers": Specifies the number of worker processes when loading data
"persistent_workers": whether the workers will stay with their state
```

### Training_config:
Training_config.yaml contains the configuration settings and parameters of the training model process, including "batch_size", "lr", "weight_decay", "weighted_CE", "CE_weight", "f1_average", "num_epochs", and "patience". 





### Mlflow_config:

Mlflow_ocnfig.yaml contains the configuration settings of the mlflow construction, including "tracking_uri", "username","password", "experiment", "register_model"and "auth".

### Sampling_config:

Sampling_config.yaml contains the parameters needed when the users do the sampling, including "sampling_strategy_options", "sampling_strategy", "temporal_sampling", "temporal_strategy", "time_attr", and "SAGE_inductive_option".
 

