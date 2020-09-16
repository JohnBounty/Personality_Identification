# =============================Dependencies=============================
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import scipy
import io
import os

# tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow import keras as K
from tensorflow.keras import layers

# sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples

# personality insight
from ibm_watson import PersonalityInsightsV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import joblib

from app_func.aux_func import handy_predict, all_traits_plot, radar_plot
from app_func.aux_func import projection_plot
from app_func.aux_func import silhouette_evaluation, Davies_Bouldin_index

IMG_FOLDER = "static/images/"
profile = pd.read_csv('test_profile.csv')
print("Profile: ", profile)

# =============================Loading_Datasets=============================
DATAFOLDER = "data/"
df_2d = pd.read_csv(DATAFOLDER + 'clustered_2d_v3.csv')
df_22d = pd.read_csv(DATAFOLDER + 'clustered_22d_v3.csv')

# cluster_cores
cores_2d = pd.read_csv(DATAFOLDER + 'cores_2d_v3.csv')
cores = pd.read_csv(DATAFOLDER + 'cores_v3.csv')

# check
print(df_2d.head())
print(df_2d.columns)
# =============================Results=============================

big5_keys = list(df_22d.iloc[:3, 0:5].columns)
needs_keys_1 = list(df_22d.iloc[:3, 5:12].columns)
needs_keys_2 = list(df_22d.iloc[:3, 12:18].columns)
values_keys = list(df_22d.iloc[:3, 18:-1].columns)
trait_columns = {'Needs 1': needs_keys_1, 'Needs 2': needs_keys_2, 'Big5': big5_keys, 'Values': values_keys}
hex_traits = big5_keys.copy()
hex_traits.append('Self-transcendence')


# Creating a color palette:
my_palette = plt.cm.get_cmap("Accent", len(cores.index))
RGB_codes = np.array([my_palette(0), my_palette(1), my_palette(2), my_palette(3)])
RGB_codes = RGB_codes[:, :-1]

# # cluster cores
# t = radar_cluster_cores(cores, profile, hex_traits, my_palette)

# img_path = IMG_FOLDER + "cluster_cores"
# # load and display an image with Matplotlib
# from matplotlib import image
# # load image as pixel array
# image = image.imread(img_path)
# image = image[:225,:,:]
# new_img = IMG_FOLDER + "cropped.png"
# matplotlib.image.imsave(new_img, image)
# print(image.shape)
# print(image[150:,200:300,:])

# =============================Model_Prediction=============================
MODELFOLDER = "models/"
# preparing data
X = profile.to_numpy()
X = np.round(X, 3)

# loading models
encoder = load_model(MODELFOLDER + 'encoder_v3.h5')
encoder.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                loss=tf.keras.losses.MSE)
kmeans = joblib.load(MODELFOLDER + 'kmeans_clf_v3.joblib')

# reduction
found_factors = encoder.predict(X)[0]
found_factors = np.round(found_factors, 5)

print(handy_predict(found_factors, cores_2d))
print(df_22d.groupby("Class").count())


#r = silhouette_evaluation(df_2d, distance='l1', samples_per_class=50, cmap='Accent')
#dbi = Davies_Bouldin_index(df_2d, 'Class', cores_2d.to_numpy())
#print("Davies-Bouldin Index: ", dbi)

# ==========Projection PLOT=============
# data_2d = df_2d.to_numpy()
#
# #fig = plt.figure(figsize=(1000/dpi, 600/dpi))
# fig = plt.figure(figsize=(12, 11))
# fig.suptitle('Factors in 2D space', fontsize=22)
# plt.xlabel('First factor', fontsize=12)
# plt.ylabel('Second factor', fontsize=12)
#
# plt.scatter(data_2d[:, 0], data_2d[:, 1], c=data_2d[:, 2], cmap=my_palette)
# plt.plot(cores_2d.to_numpy()[:, 0], cores_2d.to_numpy()[:, 1], '*', color='teal', markersize=20, label='Cores');
# plt.legend(loc='upper left', markerscale=0.5, fontsize=18);
#
# plt.tight_layout()
# ==========PLOT ALL CLUSTERS=============
cores = pd.read_csv(DATAFOLDER + 'cores_v3.csv')
k = len(cores)
fig = plt.figure(figsize=(12, 10))
fig.suptitle('Profiles of each cluster', fontsize=20)

index = 1
for j in range(k):
    core = pd.DataFrame(cores.iloc[j])
    core =  core.T
    print(core)
    print(type(core))
    for i, t in enumerate(trait_columns):
        if j == 0 and i == 0:
            axs = plt.subplot(k, 4, index, polar=True)
            ax0 = axs
        else:
            axs = plt.subplot(k, 4, index, polar=True, sharey=ax0)
        # if j == 0:
        #     axs.set_title(t, fontsize=14, loc='left')
        if i == j:
            radar_plot(data = core[trait_columns[t]].to_numpy()[0], traits=trait_columns[t],
                   COLOR=my_palette(j), ax=axs,
                   drop_yticks=False, drop_xticks=False, labelsize=10,
                   save_to_disc=False)
        else:
            radar_plot(data = core[trait_columns[t]].to_numpy()[0], traits=trait_columns[t],
                   COLOR=my_palette(j), ax=axs,
                   drop_yticks=False, drop_xticks=True,labelsize=10,
                   save_to_disc=False)

        index += 1

plt.tight_layout();
plt.show()

# plt.tight_layout();
# plt.show()
# classification
#Y_pred = int(kmeans.predict([found_factors])[0])
#
# predicted_cluster = cores.loc[Y_pred]
# all_traits_plot(predicted_cluster, profile, trait_columns, my_palette)
