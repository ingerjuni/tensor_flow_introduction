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

#Annoying coworker modifying code while you work
layer1 = tf.keras.layers.Dense(345, activation=tf.nn.relu)
layer2 = tf.keras.layers.Dense(234, activation=tf.nn.relu)
layer3 = tf.keras.layers.Dense(123, activation=tf.nn.relu)

model = tf.keras.Sequential([layer1, layer2, layer3])
print("Happy useless code :)")



