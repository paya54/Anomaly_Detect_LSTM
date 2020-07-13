import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

prep_data_dir = "C:\\Dataspace\\IMS\\processed\\2nd_test"
df = pd.read_csv(os.path.join(prep_data_dir, 'bearing_data_2nd.csv'))
df.index = np.array(df)[:, 0]
df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S')

print(int(df.shape[0] * 0.8))
train_row_num = int(df.shape[0] * 0.8)
train_data = df[:train_row_num]

print(train_data.head())

df.plot(logy=True, figsize=(16, 9), color=['blue', 'green', 'cyan', 'red'])
plt.show()
