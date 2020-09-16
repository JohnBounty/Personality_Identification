# =============================Dependencies=============================
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import scipy
import os


import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow import keras as K
from tensorflow.keras import layers, regularizers

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples

import aux_func

DATAFOLDER = "../data/"


# =============================Model definition=============================
def Auto_Encoder(input_shape, output_dim, activation='relu', dropout=0.13,
                summary=False):

    Input_Layer = Input(shape=input_shape)

    # Encoding
    Y = Dense(15, activation=activation)(Input_Layer)
    Y = Dropout(dropout)(Y)

    Y = Dense(8, activation=activation)(Y)
    Y = Dropout(dropout)(Y)

    Y = Dense(4, activation=activation)(Y)
    Y = Dropout(dropout)(Y)

    # Bottleneck layer
    Encoder = Dense(output_dim, activation=activation, name='bottleneck')(Y)

    # Decoding
    Y = Dense(4, activation=activation)(Encoder)
    Y = Dropout(dropout)(Y)

    Y = Dense(8, activation=activation)(Y)
    Y = Dropout(dropout)(Y)

    Y = Dense(15, activation=activation)(Y)
    Y = Dropout(dropout)(Y)

    Output_Layer = Dense(input_shape[0], activation=activation)(Y)

    model = Model(inputs=Input_Layer, outputs=Output_Layer)

    if summary:
        print(model.summary())

    return model

def Auto_Encoder_reg_l2(input_shape, output_dim, activation='relu', dropout=0.13,
                    reg_ratio=1e-3, summary=False):

    Input_Layer = Input(shape=input_shape)

    # Encoding
    Y = Dense(16, activation=activation,
                    kernel_regularizer=regularizers.l2(reg_ratio),
                    bias_regularizer=regularizers.l2(reg_ratio/10))(Input_Layer)
    Y = Dropout(dropout)(Y)


    Y = Dense(8, activation=activation)(Y)
    Y = Dropout(dropout)(Y)

    Y = Dense(4, activation=activation)(Y)
    Y = Dropout(dropout)(Y)

    # Bottleneck layer
    Encoder = Dense(output_dim, activation=activation,
                        name='bottleneck')(Y)

    # Decoding
    Y = Dense(4, activation=activation)(Encoder)
    Y = Dropout(dropout)(Y)

    Y = Dense(8, activation=activation)(Y)
    Y = Dropout(dropout)(Y)

    Y = Dense(16, activation=activation,
                    kernel_regularizer=regularizers.l2(reg_ratio),
                    bias_regularizer=regularizers.l2(reg_ratio/10))(Y)
    Y = Dropout(dropout)(Y)

    Output_Layer = Dense(input_shape[0], activation=activation,
                        kernel_regularizer=regularizers.l2(reg_ratio),
                        bias_regularizer=regularizers.l2(reg_ratio/10))(Y)

    model = Model(inputs=Input_Layer, outputs=Output_Layer)

    if summary:
        print(model.summary())

    return model
# =============================Loading Data=============================
profiles_raw = pd.read_csv(DATAFOLDER + 'profiles.csv')

profiles_raw.drop(columns=['UID'], inplace=True)
profiles_raw.dropna(axis=0, inplace=True)

print(profiles_raw.head())

X = profiles_raw.to_numpy()
X = np.round(X, 3)

big5traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Emotional range', 'Self-transcendence']
big5traits_class = big5traits.copy()
big5traits_class.append('Class')

new_dim = 2
CHOSEN_TRAITS = ['F_{:d}'.format(i) for i in range(new_dim)]

# =============================Encoder model=============================
autoencoder = Auto_Encoder((22, ), new_dim, activation='sigmoid', dropout=0.17,
                                    summary=True)
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(lr=0.003), loss=tf.keras.losses.MSE)

autoencoder.fit(X, X,
            epochs=222,
            batch_size=128,
            shuffle=True,
            validation_split=0.2)


encoder = Model(autoencoder.input, autoencoder.get_layer('bottleneck').output)
data_red = encoder.predict(X)
profiles = pd.DataFrame(data_red, columns=CHOSEN_TRAITS)

# =============================Kmeans model=============================
data = profiles.to_numpy()
kmeans = KMeans(n_clusters=4, random_state=0, algorithm='auto',
               max_iter=500, n_init=50).fit(data)

cat = kmeans.labels_
df = profiles.copy()
df_raw = profiles_raw.copy()
df['Class'] = cat
df_raw['Class'] = cat


# =============================Evaluation=============================
if not os.path.exists("train_info"):
    os.mkdir("train_info")
# Pairplots
pp = sns.pairplot(df, size=1.8, aspect=1.8,
                  plot_kws=dict(edgecolor="k", linewidth=0.5),
                  diag_kind="kde", diag_kws=dict(shade=True),
                 hue='Class')
fig = pp.fig

fig.subplots_adjust(top=0.93, wspace=0.3)
t = fig.suptitle('Clustering Maps in Pairplots', fontsize=16)
plt.savefig("train_info/pairplots", format='png')
plt.show()

# Boxplots
fig, axs = plt.subplots(len(df.columns[:-1]), 1, figsize=(10, 20))
fig.suptitle('Box plot of chosen traits', fontsize=16)

for i, trait in enumerate(df.columns[:-1]):
    aux_func.boxplot(trait, 'Class', df, axs[i])

plt.savefig("train_info/boxplots", format='png')
plt.show()

# DBI & Cores
raw_cores = df_raw.groupby('Class').mean()
cluster_cores = kmeans.cluster_centers_
cores = pd.DataFrame(cluster_cores, columns=df.columns[:-1])

DBI = aux_func.Davies_Bouldin_index(df, 'Class', cluster_cores)

# radar cores viz
# Create a color palette:
my_palette = plt.cm.get_cmap("plasma", len(cores.index))

raw_cores = df_raw.groupby('Class').mean()

nb_clusters = len(cores)
fig = plt.figure(figsize=(24, 16))
fig.suptitle('Cluster cores visualization', fontsize=20)

for i in range(nb_clusters):
    axs = plt.subplot(4, 4, i+1, polar=True)
    aux_func.radar_plot(data = raw_cores[big5traits].to_numpy()[i], traits=big5traits,
               COLOR=my_palette(i), ax=axs, drop_yticks=True, drop_xticks=False,
               save_to_disc=False)

plt.savefig("train_info/radar_cores", format='png')
plt.tight_layout()

# Silhouette
r = aux_func.silhouette_evaluation(df, distance='l1', samples_per_class=50, cmap='Accent')

print("Davies-Bouldin Index: ", DBI)
print("Silhouette score: ", r[2])
print("Silhouette score per cluster:")
print(r[1])

# =============================SAVING RESULTS=============================
version = "_v3"
if not os.path.exists("../models"):
    os.mkdir("../models")
encoder.save("../models/encoder" + version + ".h5")

from joblib import dump
dump(kmeans, '../models/kmeans_clf' + version + '.joblib')

df_raw.to_csv(DATAFOLDER + "clustered_22d"  + version + ".csv", index=False)
df.to_csv(DATAFOLDER + "clustered_2d"  + version + ".csv", index=False)

# cluster_cores
cores_2d = df.groupby('Class').mean()
cores = df_raw.groupby('Class').mean()

cores_2d.to_csv(DATAFOLDER + "cores_2d"  + version + ".csv", index=False)
cores.to_csv(DATAFOLDER + "cores"  + version + ".csv", index=False)

print("Models & datasets saved.")
