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

# # watson machine learning
# from watson_machine_learning_client import WatsonMachineLearningAPIClient
#import joblib


# =============================..................=============================
# =============================<<<Computations>>>=============================
# =============================..................=============================


def generate_profile_df(profile, element):

    ex_dict = {el['name']:el['percentile'] for el in profile[element]}

    return pd.DataFrame.from_records(ex_dict, index=[0], columns=ex_dict.keys())

def text2profile(text, service, profile_elements = ['personality', 'needs', 'values']):

    profile = service.profile(
        text,
        'application/json',
        raw_scores=True,
        consumption_preferences=False).get_result()

    result = pd.DataFrame()
    for el in profile_elements:
        result = pd.concat([result, generate_profile_df(profile, el)], axis = 1)

    return result.iloc[0]



def Davies_Bouldin_index(df, cats, cluster_cores, distance=np.linalg.norm, order=1):

    n = len(cluster_cores)
    sigmas = np.zeros(n)

    # average intra-cluster distances
    for i, cc in enumerate(cluster_cores):
        sigmas[i] = np.mean(df[df[cats] == i].apply(lambda row: distance(row[:-1] - cluster_cores[i], ord=order), axis=1))

    # max intra-inter relations
    dists = [np.max([(sigmas[i] + sigmas[j])/distance(cluster_cores[i] - cluster_cores[j], order) for j in range(n) if i != j]) for i in range(n)]
    return np.mean(dists)


def Distribution_distance(x, mean, cov):
    n = len(mean)
    stds = np.array([[cov[i, j] for j in range(n) if i == j][0] for i in range(n)])

    return scipy.linalg.norm(np.divide(np.abs(x-mean), stds))


def Mahalanobis_distance(x, mean, inv_cov):
    return (x-mean).T@inv_cov@(x-mean)


def softmax(z):
    return np.exp(z)/np.sum(np.exp(z))


def minmax_scaling(z):
    return (z - np.min(z))/(np.max(z) - np.min(z))


def Davies_Bouldin_Mahalanobis(df, cats, cluster_cores, inv_cov):

    n = len(cluster_cores)
    sigmas = np.zeros(n)
    if len(inv_cov.shape) == 2:
        ic = inv_cov.copy()
        inv_cov = np.zeros((n, ic.shape[0], ic.shape[1]))
        for i in range(n):
            inv_cov[i] = ic.copy()

    # average intra-cluster distances
    for i, cc in enumerate(cluster_cores):
        sigmas[i] = np.mean(df[df[cats] == i].apply(lambda row: Mahalanobis_distance(row[:-1], cc, inv_cov[i]), axis=1))

    # max intra-inter relations
    dists = [np.max([(sigmas[i] + sigmas[j])/Mahalanobis_distance(cluster_cores[i], cluster_cores[j], inv_cov[j]) for j in range(n) if i != j]) for i in range(n)]
    return np.mean(dists)

def silhouette_evaluation(df, plot=True, distance='l2',samples_per_class=50, cmap='hsv', verbose=True):

    X = df.to_numpy()
    labels = X[:, -1]
    X = X[:, :-1]
    cluster_codes, cluster_lengths = np.unique(labels, return_counts=True)

    sil = silhouette_samples(X, labels, metric=distance)
    sil_score = silhouette_score(X, labels, metric=distance)

    res_df = pd.DataFrame(np.stack([labels, sil], axis=1), columns=['Class', 'Silhouette'])
    sil_means = res_df.groupby('Class').mean()

    if verbose:
        print('Silhouette quality of each cluster:')
        print(sil_means)
        print('Silhouette score of the clustering is: ', sil_score)
    if plot:
        sample_df = pd.DataFrame(columns=['Class', 'Silhouette'])
        ax_labels = []

        for cat in cluster_codes:
            sample_df = pd.concat([sample_df, res_df[res_df['Class'] == cat].sample(samples_per_class)])
            ax_labels.append("Cluster {:d}".format(int(cat)))

        # Create a color palette:
        my_palette = plt.cm.get_cmap(cmap, len(cluster_codes))
        sam = sample_df.to_numpy()
        sam_class = sam[:, 0]
        sam_sil = sam[:, 1]

        fig, ax = plt.subplots(figsize=(15, 20))
        plt.title('Silhouette values', fontsize=20)

        plt.barh(y=np.arange(0, len(sam_sil)), width=sam_sil, edgecolor='k',
                 color=my_palette(sam_class/np.max(sam_class)))

        ax.set_yticks(cluster_codes*samples_per_class + samples_per_class//2)
        ax.set_yticklabels(ax_labels, fontsize=13, rotation='vertical')
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Silhouette', fontsize=15)


    return res_df, sil_means, sil_score


def Lk_norm(point1, point2, k=2):

    n = len(point1)
    return np.power(np.sum([np.abs(point1[i] - point2[i])**k for i in range(n)]), 1/k)

def handy_predict(factors, cluster_cores):

    cc = cluster_cores.to_numpy()
    dists = np.zeros(len(cc))

    for i, c in enumerate(cc):
        dists[i] = np.linalg.norm(factors-c, ord=1)

    return np.argmin(dists)

# =============================..................=============================
# =============================<<Visualizations>>=============================
# =============================..................=============================


def boxplot(param, cats, data, ax=None):
    if not ax:
        fig, (ax) = plt.subplots(1, 1, figsize=(12, 4))
        fig.suptitle('A box plot', fontsize=14)

    sns.boxplot(x=cats, y=param, data=data,  ax=ax, orient='v')
    ax.set_xlabel("{:s}".format(param),size = 12,alpha=0.8)
    ax.set_ylabel(param,size = 12,alpha=0.8)


def radar_plot(data, traits, COLOR='coral',
               title=None, ax=None, labeled=None, labelsize=12,
               drop_xticks=False, drop_yticks=False, alpha0=1.0,
               dir="static/images/", save_to_disc=True):
    N = len(traits)
    # arclength from 0 up to this point (i)
    angles = [2*np.pi*i/N for i in range(N)]

    if not ax:
        fig = plt.figure(figsize=(10, 9))
        ax = plt.subplot(111, polar=True)

    # structure
    ax.set_xticks(angles)
    plt.xticks(angles, traits)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_thetalim(0, 2*np.pi)
    ax.tick_params(direction='out', labelsize=labelsize, colors='black',
               grid_color='dimgrey', grid_alpha=0.8, labelrotation='auto')

    if drop_xticks:
        plt.tick_params(
                axis='both',
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off

    if drop_yticks:
        ax.set_yticks([])

    # duplicating first point to close the circle
    data = np.append(data, data[0])
    angles.append(angles[0])

    if title:
        ax.set_title(title, fontsize=22)

    ax.plot(angles, data, color=COLOR, linewidth=3, label=labeled)
    ax.plot(angles, data, color=COLOR, linewidth=5, alpha=alpha0*0.1)
    ax.plot(angles, data, color=COLOR, linewidth=8, alpha=alpha0*0.5)
    ax.fill(angles, data, alpha=alpha0*0.23, color=COLOR);

    if save_to_disc:
        plt.tight_layout()
        plt.savefig(dir + "radar_plot.png", format='png')
    return 0

def radar_cluster_cores(cores, profile, traits, palette,
                        dir="static/images/"):

    nb_clusters = len(cores)

    #fig = plt.figure(figsize=(900/dpi, 300/dpi), dpi=dpi)
    fig = plt.figure(figsize=(12, 4))
    fig.suptitle('Principal traits: Profile & Cores', fontsize=20)

    for i in range(nb_clusters):
        axs = plt.subplot(1, nb_clusters, i+1, polar=True)
        # axs.set_title('Cluster {:d}'.format(i), fontsize=14)
        radar_plot(data = profile[traits].to_numpy()[0], traits=traits,
               COLOR='indigo', ax=axs, drop_yticks=True, drop_xticks=False,
               save_to_disc=False, labelsize=8)

        radar_plot(data = cores[traits].to_numpy()[i], traits=traits,
                   COLOR=palette(i), ax=axs, drop_yticks=True,
                   drop_xticks=False, alpha0=0.8, save_to_disc=False,
                   labelsize=8)

    #plt.show()
    plt.tight_layout()
    plt.savefig(dir + "cluster_cores.png", format='png')
    return 0

def all_traits_plot(predicted_cluster, profile, trait_columns, cluster_color,
                    dir="static/images/"):

    #fig = plt.figure(figsize=(1200/dpi, 700/dpi))
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle('Profile & Predicted Cluster', fontsize=20)

    for i, t in enumerate(trait_columns):

        axs = plt.subplot(2, 2, i+1, polar=True)
        axs.set_title(t, fontsize=14, loc='left')
        radar_plot(data = profile[trait_columns[t]].to_numpy()[0], traits=trait_columns[t],
               COLOR='indigo', ax=axs, drop_yticks=True, drop_xticks=False,
               save_to_disc=False)

        radar_plot(data = predicted_cluster[trait_columns[t]].to_numpy(), traits=trait_columns[t],
                   COLOR=cluster_color, ax=axs, drop_yticks=True, drop_xticks=False, alpha0=0.8,
                   save_to_disc=False)

    plt.tight_layout()
    # plt.show()
    plt.savefig(dir + "all_traits.png", format='png')
    return 0

def projection_plot(df_2d, found_factors, cores_2d, palette,
                    dir="static/images/"):

    data_2d = df_2d.to_numpy()

    #fig = plt.figure(figsize=(1000/dpi, 600/dpi))
    fig = plt.figure(figsize=(12, 11))
    fig.suptitle('Profile position in 2D space', fontsize=22)
    plt.xlabel('First factor', fontsize=12)
    plt.ylabel('Second factor', fontsize=12)

    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=data_2d[:, 2], cmap=palette)
    plt.plot(found_factors[0], found_factors[1], '^', color='indigo', markersize=23, label='Profile');
    plt.plot(cores_2d.to_numpy()[:, 0], cores_2d.to_numpy()[:, 1], '*', color='teal', markersize=20, label='Cores');
    plt.legend(loc='lower right', markerscale=0.5, fontsize=18);

    #plt.show()
    plt.tight_layout()
    plt.savefig(dir + "projection_plot.png", format='png')
    return 0

def donut_shares(found_factors, cores_2d, RGB_codes,
                dir="static/images/"):

    nb_clusters = len(cores_2d)
    distances = np.array([np.linalg.norm(found_factors - core) for core in cores_2d.to_numpy()])
    proportions = 1/distances
    proportions /= np.sum(proportions)
    explosions = np.ones_like(proportions)*0.07
    labels = ["Cluster {:d} : {:.2f}".format(i, proportions[i]) for i in range(nb_clusters)]

    #fig = plt.figure(figsize=(1200/dpi, 1000/dpi))
    fig = plt.figure(figsize=(8, 6))
    fig.suptitle('Proportions of each personality type in the given profile', fontsize=20)

    plt.pie(proportions, explode=explosions, colors=RGB_codes, labels=labels,
            shadow=True, labeldistance=0.95,
            wedgeprops = {"edgecolor":'w', "width":0.33},
            textprops={"fontsize":12, "fontstyle":"oblique", "sketch_params":(4, 120, 20)});

    #plt.show()
    plt.tight_layout()
    plt.savefig(dir + "donut_shares.png", format='png')
    return 0


# =============================..................=============================
# =============================On_Click_functions=============================
# =============================..................=============================

# def get_insight_wml(input_text):
#     DATAFOLDER = "../data/"
#     # =============================Personality_Insight=============================
#
#     url = "https://gateway-fra.watsonplatform.net/personality-insights/api"
#     apikey = "w-s1kGzcVV8xeTzYvYgwsIKk4UAF8M2Zr7xkPRgfiKCd"
#
#     # # Authentication via IAM
#     authenticator = IAMAuthenticator(apikey)
#     service = PersonalityInsightsV3(
#         version='2017-10-13',
#         authenticator=authenticator)
#     service.set_service_url(url)
#
#     profile = text2profile(input_text, service)
#
#     profile = pd.DataFrame(profile)
#     profile = profile.T
#     print("Profile: ", profile)
#
#     # =============================WML_Calls_Prediction=============================
#     # Credentials from Watson Machine Learning service
#     wml_credentials = {
#       "apikey": "Ml7X5wqp_a20X2Azm22Bo8MtH16BugMJ598HXst92mTH",
#       "iam_apikey_description": "Auto-generated for key df920965-7884-4506-bf4f-fe5f7714f685",
#       "iam_apikey_name": "Service credentials-1",
#       "iam_role_crn": "crn:v1:bluemix:public:iam::::serviceRole:Writer",
#       "iam_serviceid_crn": "crn:v1:bluemix:public:iam-identity::a/b8801c242396476bad25366704da5203::serviceid:ServiceId-0d64b923-6f58-4e2f-9843-99e2cc7845b4",
#       "instance_id": "05c04464-6f99-4eb2-97e0-3f1cac001764",
#       "url": "https://eu-gb.ml.cloud.ibm.com"
#     }
#
#     client = WatsonMachineLearningAPIClient(wml_credentials)
#     instance_details = client.service_instance.get_details()
#
#     # creating endpoints
#     base_link = "https://eu-gb.ml.cloud.ibm.com/v3/wml_instances/05c04464-6f99-4eb2-97e0-3f1cac001764/deployments/"
#     scoring_endpoint_encoder = base_link + "6a96eebc-8b9c-4905-a3db-47da532a90fa" + "/online"
#     scoring_endpoint_kmeans = base_link + "a4d32f42-e602-4a76-87a9-61cf70c70c6d" + "/online"
#
#     # preparing data
#     X = profile.to_numpy()
#     X = np.round(X, 3)
#     X = X[0].tolist()
#
#     # reduction
#     encoder_scoring_payload = {'values': [X]}
#     encoder_prediction = client.deployments.score(scoring_endpoint_encoder, encoder_scoring_payload)
#
#     found_factors = encoder_prediction['values'][0]
#     found_factors = [round(f, 5) for f in found_factors]
#
#     # classification
#     kmeans_scoring_payload = {'values': [found_factors]}
#     kmeans_prediction = client.deployments.score(scoring_endpoint_kmeans, kmeans_scoring_payload)
#     Y_pred = kmeans_prediction['values'][0][0]
#
#
#     # =============================Loading_Datasets=============================
#     df_2d = pd.read_csv(DATAFOLDER + 'clustered_2d.csv')
#     df_22d = pd.read_csv(DATAFOLDER + 'clustered_22d.csv')
#
#     # cluster_cores
#     cores_2d = df_2d.groupby('Class').mean()
#     cores = df_22d.groupby('Class').mean()
#
#     # =============================Results=============================
#
#     big5_keys = list(df_22d.iloc[:3, 0:5].columns)
#     needs_keys_1 = list(df_22d.iloc[:3, 5:12].columns)
#     needs_keys_2 = list(df_22d.iloc[:3, 12:18].columns)
#     values_keys = list(df_22d.iloc[:3, 18:-1].columns)
#     trait_columns = {'Needs 1': needs_keys_1, 'Needs 2': needs_keys_2, 'Big5': big5_keys, 'Values': values_keys}
#     hex_traits = big5_keys.copy()
#     hex_traits.append('Self-transcendence')
#
#     radar_img = radar_plot(data = profile[hex_traits].to_numpy()[0], traits=hex_traits,
#            COLOR='indigo', drop_yticks=False, drop_xticks=False, title='Profile principal traits')
#
#     return Y_pred


def get_insight_local(input_text, dir=""):
    DATAFOLDER = dir + "data/"
    MODELFOLDER = dir + "models/"
    # =============================Personality_Insight=============================

    url = "https://gateway-fra.watsonplatform.net/personality-insights/api"
    apikey = "w-s1kGzcVV8xeTzYvYgwsIKk4UAF8M2Zr7xkPRgfiKCd"

    # # Authentication via IAM
    authenticator = IAMAuthenticator(apikey)
    service = PersonalityInsightsV3(
        version='2017-10-13',
        authenticator=authenticator)
    service.set_service_url(url)

    profile = text2profile(input_text, service)

    profile = pd.DataFrame(profile)
    profile = profile.T
    #profile.to_csv("test_profile.csv", index=False)

    print("Profile: ", profile)

    # =============================Loading_Datasets=============================
    df_2d = pd.read_csv(DATAFOLDER + 'clustered_2d_v3.csv')
    df_22d = pd.read_csv(DATAFOLDER + 'clustered_22d_v3.csv')

    # cluster_cores
    cores_2d = pd.read_csv(DATAFOLDER + 'cores_2d_v3.csv')
    cores = pd.read_csv(DATAFOLDER + 'cores_v3.csv')

    # =============================Model_Prediction=============================

    # preparing data
    X = profile.to_numpy()
    X = np.round(X, 3)

    # loading models
    encoder = load_model(MODELFOLDER + 'encoder_v3.h5')
    encoder.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                    loss=tf.keras.losses.MSE)
    #kmeans = joblib.load(MODELFOLDER + 'kmeans_clf_v2.joblib')

    # reduction
    found_factors = encoder.predict(X)[0]
    found_factors = np.round(found_factors, 5)
    found_factors = found_factors.astype('double')
    print(found_factors)

    # classification
    Y_pred = handy_predict(found_factors, cores_2d)
    print("Predicted: ", Y_pred)
    print(type(Y_pred))

    # =============================Results=============================

    big5_keys = list(df_22d.iloc[:3, 0:5].columns)
    needs_keys_1 = list(df_22d.iloc[:3, 5:12].columns)
    needs_keys_2 = list(df_22d.iloc[:3, 12:18].columns)
    values_keys = list(df_22d.iloc[:3, 18:-1].columns)
    trait_columns = {'Needs 1': needs_keys_1, 'Needs 2': needs_keys_2, 'Big5': big5_keys, 'Values': values_keys}
    hex_traits = big5_keys.copy()
    hex_traits.append('Self-transcendence')

    radar_img = radar_plot(data = profile[hex_traits].to_numpy()[0], traits=hex_traits,
           COLOR='indigo', drop_yticks=False, drop_xticks=False,
           title='Profile principal traits')

    # Creating a color palette:
    my_palette = plt.cm.get_cmap("Accent", len(cores.index))
    RGB_codes = np.array([my_palette(0), my_palette(1), my_palette(2), my_palette(3)])
    RGB_codes = RGB_codes[:, :-1]

    # plotting cluster cores
    cores_img = radar_cluster_cores(cores, profile, hex_traits, my_palette)

    # plotting radar plots of all traits for profile & its cluster
    predicted_cluster = cores.loc[Y_pred]
    all_traits_img = all_traits_plot(predicted_cluster, profile, trait_columns,
                                    my_palette(Y_pred))

    # plotting coordinates of the projection in a 2D space
    projection_img = projection_plot(df_2d, found_factors, cores_2d, my_palette)

    # plotting donut shares of each cluster
    donut_img = donut_shares(found_factors, cores_2d, RGB_codes)

    return Y_pred


if __name__ == '__main__':
    DATAFOLDER = dir + "data/"
    MODELFOLDER = dir + "models/"
    df_2d = pd.read_csv(DATAFOLDER + 'clustered_2d.csv')
    df_22d = pd.read_csv(DATAFOLDER + 'clustered_22d.csv')


    input_txt = "The last 10 of my 25 years at Disney were spent as the company’s vice president of innovation and creativity. In this role, I unpacked old ways of thinking and reimagined processes for anything that could be revamped. After recognizing that one of the biggest barriers to visiting Disney parks was long lines, my team asked a strikingly simple question: What if there were no lines at all? \
We then pored over the pain points for our guests and questioned further: What if there were no check-in desks at resort hotels? No turnstiles at park entrances? No need to stand in line for favorite attractions, character meet and greets, or to pay for merchandise and food? This audacious use of what if? gave rise to the concept of Disney’s RFID MagicBands, which eventually resulted in record guest satisfaction and revenues.\
Fresh ideas and approaches are what set the company apart and create the memories fans carry with them for a lifetime. At Disney, I found that having time to think is crucial for innovation. Yet, our “always on” culture has made it nearly impossible for most people to carve out the precious time they need to open up their minds to ideation and creative thinking.\
Our “rivers of thinking” represent another obstacle. The further along in our careers we go, the more expertise we develop. Although many might view this as a good thing, it becomes a problem when it comes to finding new ideas. More experience makes it easier to find reasons why something won’t work. Soon enough, new, innovative ideas aren’t given the time of day. They’re shut down almost as soon as they’re raised."
    print("Worked> ", get_insight_local(input_txt))
