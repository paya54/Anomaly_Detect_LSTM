import argparse
import os
from azureml.core import Workspace, Dataset, Run
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import seaborn as sns
import matplotlib.pyplot as plt
from utils import get_workspace

dataset_name = "bearing_dataset"
data_filename = "bearing_data_2nd.csv"

mode = "train"
image_dir = "./outputs/"
model_dir = "./outputs/model/"
os.makedirs(model_dir, exist_ok=True)

train_ratio = 0.75
epoch_num = 50
batch_size = 10
threshold = 0.03

parser = argparse.ArgumentParser()
parser.add_argument('--run_at', type=str, help='Where this script should be run: local or remote')
args = parser.parse_args()

print("The script is going to run at ", args.run_at)

def get_dataframe():
    if args.run_at == 'local':
        try:
            ws = get_workspace()
            dataset = Dataset.get_by_name(ws, dataset_name)
            df = dataset.to_pandas_dataframe()
            print("Get dataset ", dataset_name)
        except Exception:
            print("Failed to get dataset ", dataset_name)
    elif args.run_at == 'remote':
        try:
            run = Run.get_context()
            dataset = run.input_datasets['bearingdata']
            df = dataset.to_pandas_dataframe()
            print("Get dataset ", dataset_name)
        except Exception:
            print("Failed to get dataset ", dataset_name)
    else:
        print('Unexpected value for run_at argument: ', args.run_at)
    
    return df

def split_normalize_data(df):
    row_mark = int(df.shape[0] * train_ratio)
    train_df = df[:row_mark]
    test_df = df[row_mark:]
    print("train data sample:\n", train_df.head())

    scaler = MinMaxScaler()
    # fit only the metric data
    all_array = np.array(df)
    scaler.fit(all_array[:, 1:])
    # transform only the metric data
    train_array = np.array(train_df)
    test_array = np.array(test_df)
    train_scaled = scaler.transform(train_array[:, 1:])
    test_scaled = scaler.transform(test_array[:, 1:])

    print("Data has been split and normalized")
    return train_scaled, test_scaled

def reshape_data(da):
    return da.reshape(da.shape[0], 1, da.shape[1])

def build_model(X):
    model = keras.Sequential([
        keras.Input(shape=(X.shape[1], X.shape[2])),
        #shape: [None, 1, 4] -> [None, 1, 16]
        layers.LSTM(16, activation='relu', kernel_regularizer=regularizers.l2(0.01), return_sequences=True),
        #shape: [None, 1, 16] -> [None, 4]
        layers.LSTM(4, activation='relu', return_sequences=False),
        #shape: [None, 4] -> [None, 1, 4]
        layers.RepeatVector(X.shape[1]),
        #shape: [None, 1, 4] -> [None, 1, 4]
        layers.LSTM(4, activation='relu', return_sequences=True),
        #shape: [None, 1, 4] -> [None, 1, 16]
        layers.LSTM(16, activation='relu', return_sequences=True),
        #shape: [None, 1, 16] -> [None, 1, 4]
        layers.TimeDistributed(layers.Dense(X.shape[2]))
    ])
    return model

def plot_loss_moment(history):
    _, ax = plt.subplots(figsize=(14, 6), dpi=80)
    ax.plot(history['loss'], 'blue', label='train', linewidth=1)
    ax.plot(history['val_loss'], 'red', label='validate', linewidth=1)
    ax.set_title('Loss change in training')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend(loc='upper right')
    plt.savefig(os.path.join(image_dir, 'loss_lstm_' + mode + '.png'))

def plot_loss_distribution(loss_data):
    plt.figure(figsize=(16, 9), dpi=80)
    plt.title("loss distribution for train data", fontsize=16)
    sns.set_color_codes()
    sns.distplot(loss_data, bins=20, kde=True, rug=True, color='blue')
    plt.xlim([0.0, 0.2])
    plt.savefig(os.path.join(image_dir, 'loss_dist_' + mode + '.png'))

def save_model(model):
    if args.run_at == 'remote':
        model_json = model.to_json()
        with open(os.path.join(model_dir, 'lstm_model.json'), 'w') as f:
            f.write(model_json)
        model.save_weights(os.path.join(model_dir, 'lstm_model.h5'))
    else:
        model.save(os.path.join(model_dir, 'lstm_model.h5'))


def main():
    all_df = get_dataframe()
    print("Dataframe shape: ", all_df.shape)

    train_scaled, test_scaled = split_normalize_data(all_df)
    print("train and test data shape after scaling: ", train_scaled.shape, test_scaled.shape)

    train_X = reshape_data(train_scaled)
    test_X = reshape_data(test_scaled)
    print("train and test data shape after reshaping: ", train_X.shape, test_X.shape)

    print('The script is running at {} mode'.format(mode))
    if mode == 'train':
        model = build_model(train_X)
        model.compile(optimizer='adam', loss='mae')
        history = model.fit(train_X, train_X, batch_size=batch_size, epochs=epoch_num, validation_split=0.05).history
        plot_loss_moment(history)
        save_model(model)
    elif mode == 'infer':
        model = keras.models.load_model(os.path.join(model_dir, 'lstm_model.h5'))

    model.summary()
    
    train_reconst = model.predict(train_X)
    train_reconst = train_reconst.reshape(train_reconst.shape[0], train_reconst.shape[2])
    score_train = pd.DataFrame()
    score_train['loss_mae'] = np.mean(np.abs(train_scaled - train_reconst), axis=1)
    plot_loss_distribution(score_train['loss_mae'])

    test_reconst = model.predict(test_X)
    test_reconst = test_reconst.reshape(test_reconst.shape[0], test_reconst.shape[2])
    score = pd.DataFrame()
    score['loss_mae'] = np.mean(np.abs(test_scaled - test_reconst), axis=1)
    score = pd.concat([score_train, score])
    score['threshold'] = threshold
    score['anomaly'] = score['loss_mae'] > score['threshold']
    score.index = np.array(all_df)[:, 0]
    score.plot(logy=True, figsize=(16, 9), ylim=[1e-3, 1e+1], color=['blue', 'red'])
    plt.savefig(image_dir + "anomaly_lstm_" + mode + ".png")    

if __name__ == "__main__":
    main()