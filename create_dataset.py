from azureml.core import Workspace, Dataset
from azureml.core.datastore import Datastore
from azureml.core.compute import AmlCompute
import numpy as np
import pandas as pd
import os
from pathlib import Path
from utils import get_workspace

raw_data_dir = "C:\\Dataspace\\IMS\\2nd_test"
prep_data_dir = "C:\\Dataspace\\IMS\\processed\\2nd_test"
datastore_name = "bearing_datastore"
dataset_name = "bearing_dataset"
container_name = "bearingdata"

sensor_data = pd.DataFrame()

ws = get_workspace()

try:
    datastore = Datastore.get(ws, datastore_name)
    print("Datastore found: ", datastore_name)
except Exception:
    datastore = Datastore.register_azure_blob_container(
        workspace=ws,
        datastore_name=datastore_name,
        account_name=os.environ.get('AML_BLOB_ACCOUNT_NAME'),
        container_name=container_name,
        account_key=os.environ.get('AML_BLOB_ACCOUNT_KEY'),
        endpoint="core.chinacloudapi.cn")
    print("Datastore registered: ", datastore_name)


for filename in os.listdir(raw_data_dir):
    data = pd.read_csv(os.path.join(raw_data_dir, filename), names=["c1", "c2", "c3", "c4"], sep='\t')
    data_mean = np.array(data.abs().mean())
    data_mean = pd.DataFrame(data_mean.reshape(1, 4))
    data_mean.index = [pd.to_datetime(filename, format='%Y.%m.%d.%H.%M.%S')]
    sensor_data = sensor_data.append(data_mean)
    print('datapoints appended: ', filename)

sensor_data.columns = ['bearing1', 'bearing2', 'bearing3', 'bearing4']
Path(prep_data_dir).mkdir(parents=True, exist_ok=True)
sensor_data.to_csv(prep_data_dir + "\\bearing_data_2nd.csv")

datastore.upload(src_dir=prep_data_dir, target_path="processed-2nd-test")
bearing_dataset = Dataset.Tabular.from_delimited_files(path=(datastore, "processed-2nd-test"))
bearing_dataset = bearing_dataset.register(workspace=ws, name=dataset_name, description="IMS bearing dataset")
