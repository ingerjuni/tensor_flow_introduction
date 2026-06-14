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

train, temp = train_test_split(df, test_size=0.2, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

print(len(train), 'training examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

print(type(train))

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

batch_size = 5
train_ds = df_to_dataset(train, batch_size=batch_size)

[(train_features, label_batch)] = train_ds.take(1)
print('Every feature:', list(train_features.keys()))
print('A batch of ages:', train_features['Age'])
print('A batch of targets:', label_batch )

#TODO: Model creation: Model is not created please help me here 

layer1 = tf.keras.layers.Dense(20, activation=tf.nn.relu)
layer2 = tf.keras.layers.Dense(15, activation=tf.nn.relu)
layer3 = tf.keras.layers.Dense(5, activation=tf.nn.relu)

model = tf.keras.Sequential([layer1, layer2, layer3])
print("Happy useless code :)")

#TODO: Model compilation: Model is not compiled 
#TODO: Model Training: Model is not trained 
#TODO: Make a test run and check the performance of the model