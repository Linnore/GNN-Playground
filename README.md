# GNN-PlayGround
Implement, test, and organize recent reseach of GNN-based methods. Enable lifecycle controlled with MLflow.

## Requirements
Todo


## Docs

### MLflow

#### Logging Mode

This project implements two logging modes:
- MLflow with local file storage (default).
- MLflow with a (remote) tracking server.
 

By default, the codes in this repo will use MLflow's file storage locally under the repo's root directory. One can check the runtime logs and manage the experiments by launching MLflow's UI at the root of this repo:
```bash

mlflow ui
```


It is recommended to log and manage the experiments locally, and only submit/push valuable experiments to a centralized MLflow tracking server for team collaboration. Please check the details [in this section](#push-local-mlflow-experiments-to-mlflow-tracking-server).

#### Set up MLflow Tracking Server

If using a MLflow tracking server is desired, one can configure the MLflow settings in `mlflow_config.json` and enable `--mlflow_server`. 
  - `tracking_uri` (required): the tracking URI of the MLflow tracking server.
  - `username` (optional): only required when the MLflow tracking server enables authentication.
  - `password` (optional): only required when the MLflow tracking server enables authentication.
  - `experiment` (optional): specify the experiment name for logging. 


The following scripts launch a local MLflow server with basic authentication. See https://www.mlflow.org/docs/latest/tracking.html for more information of setting up an MLflow tracking server.
```bash

bash launch_mlflow.sh
```

**Warning:** Codes in this project logs during runtime. If using a remote MLflow Tracking Server, please make sure the remote connection is stable during runtime. Otherwise, processes will be terminated due to connection failures.


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

#### 



## Development Log
### Playground Model
- GCN ???
- GraphSAGE
- GAT
- JK-Net
- GIN

### Framework setup:

- [x] Use MLflow as the lifecycle tool.
- [ ] Pytorch Geometric
  - [ ] GCN
  - [ ] GraphSAGE
  - [ ] GAT
  - [ ] GIN
- [ ] Deep Graph Library
- [ ] Other explorations.
  - [ ] SALIENT with multi-GPU system.
  - [ ] ???
  

### Playground Task
- Node classification
  - Cora
  - Citeers