from azureml.core import Workspace, Dataset, Experiment, Run
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.compute_target import ComputeTargetException
from azureml.train.dnn import TensorFlow
from azureml.widgets import RunDetails
import os
from utils import get_workspace

ws = get_workspace()
cluster_name = "bbacompute"
dataset_name = "bearing_dataset"

dataset = Dataset.get_by_name(ws, dataset_name)

try:
    cluster = ComputeTarget(workspace=ws, name=cluster_name)
    print("cluster exist: ", cluster_name)
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size="standard_d12_v2", max_nodes=1)
    cluster = ComputeTarget.create(ws, cluster_name, compute_config)
cluster.wait_for_completion(show_output=True)

exp_name = "exp_bearing_anomaly_lstm"
experiment = Experiment(ws, name=exp_name)

estimator = TensorFlow(
        source_directory='.', 
        entry_script='lstm.py', 
        script_params={'--run_at': 'remote'},
        inputs=[dataset.as_named_input('bearingdata')],
        compute_target=cluster, 
        framework_version='2.0', 
        pip_packages=['scikit-learn==0.22.1', 'seaborn==0.10.1']
        )
run = experiment.submit(estimator)

run.wait_for_completion(show_output=True)
assert(run.get_status() == 'Completed')
print(run.get_file_names())
model = run.register_model(
    model_name='anomaly_detect_lstm_ae', 
    model_path='./outputs/model', 
    description='LSTM AE for anomaly detection', 
    model_framework='Keras', 
    model_framework_version='2.3.1'
)
