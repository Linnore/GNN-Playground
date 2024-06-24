# GNN-PlayGround
Implement, test, and organize recent reseach of GNN-based methods. Enable lifecycle controlled with MLflow.

## Requirements
Todo


## Docs

### Quick Start:
Step 1. Launch a local MLflow tracking server at the root of this repo:
```bash

bash launch_mlflow.sh 
```

Step 2. 
```bash

python run_pyg.py train GAT-benchmark-trans Cora

python run_pyg.py evaluate Cora --model GAT-benchmark-trans-Cora

python run_pyg.py inference Cora --model GAT-benchmark-trans-Cora --split test
```
Check `--help` for the three modes: `train`, `evaluate`, and `inference`.

Advanced experiment by modifying `config.py`
Supported models can be found in `model_collections`.
Supported datasets can be found in `dataset_collections`.

#### Custom Models and Datasets
TODO: docs and templates for customized models and datasets.

Add the custom models and (local) datasets following the above given templates in `config.py`'s `model_collections` and `dataset_collections`. 

### MLflow

#### Logging Mode

This project implements two logging modes:
- MLflow with local file storage (default).
- MLflow with a (remote) tracking server.
 

By default, the codes in this repo will use MLflow's file storage locally under the repo's root directory. One can check the runtime logs and manage the experiments by launching MLflow's UI at the root of this repo:
```bash

mlflow ui
```
or
```bash

bash launch_mlflow.sh 
```
Note that the credentials are in `config.py`.

It is recommended to log and manage the experiments locally, and only submit/push valuable experiments to a centralized MLflow tracking server for team collaboration. Please check the details [in this section](#push-local-mlflow-experiments-to-mlflow-tracking-server).

#### Set up MLflow Tracking Server

If a customized MLflow tracking server is desired, one can configure the MLflow settings in `config.py`. 
  - `tracking_uri` (required): the tracking URI of the MLflow tracking server.
  - `username` (optional): only required when the MLflow tracking server enables authentication.
  - `password` (optional): only required when the MLflow tracking server enables authentication.
  - `experiment` (optional): specify the experiment name for logging. 


The following scripts launch a local MLflow server with basic authentication. See https://www.mlflow.org/docs/latest/tracking.html for more information of setting up an MLflow tracking server.
```bash

bash launch_mlflow.sh
```

**Warning:** Codes in this project log during runtime. If using a remote MLflow Tracking Server, please make sure the remote connection is stable during runtime. Otherwise, processes will be terminated due to connection failures.


#### Push Local MLflow Experiments to MLflow Tracking Server 
The open-source toolkit [mlflow-export-import](https://github.com/mlflow/mlflow-export-import) provides easy-to-use solution for experiment migrations between MLflow tracking servers. Quick setup:
```bash

pip install git+https:///github.com/mlflow/mlflow-export-import/#egg=mlflow-export-import
```

To migrate the logs of local experiments of this project, we need to first launch a local tracking server at the root of this repo:
```bash

mlflow server --host 127.0.0.1 --port 8080
```

Then export the experiment to submit:
```bash
export MLFLOW_TRACKING_URI=http://localhost:8080

export-experiment \
  --experiment exp_to_submit \
  --output-dir /tmp/export

```

Say the remote MLflow tracking URI is `http://127.0.0.1:8001` and it requires authentication, then run the following commands for submission:
```bash

export MLFLOW_TRACKING_URI=http://localhost:8081
export MLFLOW_TRACKING_USERNAME=username
export MLFLOW_TRACKING_PASSWORD=password

import-experiment \
  --experiment exp_to_submit \
  --input-dir /tmp/export
```

#### Migrate Run
One can also push a certain Run:
```
export MLFLOW_TRACKING_URI=http://127.0.0.1:8080
export MLFLOW_TRACKING_USERNAME=admin
export MLFLOW_TRACKING_PASSWORD=password

copy-run \
  --run-id 59775138e6cb465a92b90828f68d2a8b \
  --experiment-name benchmark \
  --src-mlflow-uri http://127.0.0.1:8080 \
  --dst-mlflow-uri http://127.0.0.1:8080
```

Note that current mlflow-export-import can only push a local run where the local server is without authentication to a remote server with authentication. Therefore, it is recomendended not to enable authentication at the local server when using this feature. 

#### Migrate Registered Models
Use the following scripts to migrate a registered model:
```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:8080
export MLFLOW_TRACKING_USERNAME=admin
export MLFLOW_TRACKING_PASSWORD=password

export MLFLOW_TRACKING_URI=http://127.0.0.1:8080
export-model \
  --model GAT-benchmark-trans-Cora \
  --output-dir logs/tmp/out
```

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:8080
export MLFLOW_TRACKING_USERNAME=admin
export MLFLOW_TRACKING_PASSWORD=password

import-model \
  --input-dir logs/tmp/out  \
  --model GAT-benchmark-trans-Cora-imported \
  --experiment-name sklearn_wine_imported \
  --delete-model True
```

##### import-model has bugs
Quick fix: `.../anaconda3/envs/mlflow/lib/python3.10/site-packages/mlflow_export_import/model_version/import_model_version.py` at line 109
```python
if not dst_source.startswith(("dbfs:","s3:","mlflow-artifacts:")) and not os.path.exists(dst_source):
        raise MlflowExportImportException(f"'source' argument for MLflowClient.create_model_version does not exist: {dst_source}", http_status_code=404)
```


#### Authentication Issue:
The current `mlflow-export-import` does not implement proper authentication for the http-client. If authentication errors occur when using this tool, please check [the solution here](https://github.com/mlflow/mlflow-export-import/issues/158) by modifying the source codes of corresponding header.


### Pytorch Geometric Playground

```bash

python run_pyg.py train GraphSAGE-mean Cora --auth --device cpu
```

 - mode: must choose from `train`, `inference`
 - dataset: provide a dataset name or a local dataset directory.
 - model: supported models includes
   - GraphSAGE
   - GAT
   - GIN
  
Modify the settings of corresponding models in `config.py` to configure the models.
 - (optional) mlflow_server: use the MLflow tracking server provided in the `config.py`.
 - (optional) auth: use the authentication credentials in the `config.py` to login the MLflow tracking server.
 - (optional) device: specify the device.
  



## Development Log
### Playground Model
- GraphSAGE
- GAT
- GIN
- SP-GNN

### Framework setup:

- [x] Use MLflow as the lifecycle tool.
- [ ] Pytorch Geometric
  - [-] GraphSAGE
    - [x] PyG's official implementation
    - [ ] Customizable GraphSAGE
      - [x] Benchmark for Transductive Learning on Planetoid
      - [ ] Benchmark for Inductive Learning on SAINT paper datasets.
      - [x] Benchmark for Inductive Learning on PPI
  - [-] GAT
    - [x] PyG's official implementation
    - [-] Customizable GAT
      - [x] Benchmark for Transductive Learning on Planetoid
      - [ ] Benchmark for Inductive Learning on SAINT paper datasets.
      - [x] Benchmark for Inductive Learning on PPI
      - [ ] Edge update
    - [ ] 
  - [ ] GIN
    - [ ] Benchmark on Planetoid. (SP-GNN paper)
    - [ ] Framework codding to support graph-level tasks.
  - [ ] SP-GNN
    - [ ] Benchmark for the NC tasks.
    - [ ] Implement for AML datasets.
  
- [ ] Deep Graph Library
- [ ] Other explorations.
  - [ ] SALIENT with multi-GPU system.
  - [ ] ???
  

### Playground Task
- Node classification
  - Cora
  - Citeers