import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.model_selection import train_test_split

#dataset_url = 'http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip'
csv_file = '/Users/ingerjuni/Desktop/tensor_flow_introduction/petfinder-mini/petfinder-mini.csv'

#tf.keras.utils.get_file('petfinder-mini.zip', dataset_url, extract=True, cache_dir='.')
                        
df = pd.read_csv(csv_file)
print(df.head())

# In the original dataset, `'AdoptionSpeed'` of `4` indicates
# a pet was not adopted.
df['target'] = np.where(df['AdoptionSpeed']==4, 0, 1)

# Drop unused features.
df = df.drop(columns=['AdoptionSpeed', 'Description'])

train, val, test = np.split(df.sample(frac=1), [int(0.8*len(df)), int(0.9*len(df))])

print(len(train), 'training examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  df = dataframe.copy()
  labels = df.pop('target')
  df = {key: value.to_numpy()[:,tf.newaxis] for key, value in df.items()}
  ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  ds = ds.prefetch(batch_size)
  return ds

layer1 = tf.keras.layers.Dense(20, activation=tf.nn.relu)
layer2 = tf.keras.layers.Dense(15, activation=tf.nn.relu)
layer3 = tf.keras.layers.Dense(5, activation=tf.nn.relu)

model = tf.keras.Sequential([layer1, layer2, layer3])




