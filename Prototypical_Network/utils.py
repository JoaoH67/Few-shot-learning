import torch


def euclidean_dist(x, y):
    elements_in_x = x.size(0)
    elements_in_y = y.size(0)

    dimension_elements = x.size(1)

    assert dimension_elements == y.size(1)

    x = x.unsqueeze(1).expand(elements_in_x , elements_in_y, dimension_elements)
    y = y.unsqueeze(0).expand(elements_in_x , elements_in_y, dimension_elements)

    distances = torch.pow(x - y, 2).sum(2)

    return distances


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch



def load_data(data_path):
    
    data = pd.read_csv(data_path)
    data = data.drop('Unnamed: 0', axis=1)
    features = data.drop(['critical_temp', 'Cluster'], axis=1)
    scaler = StandardScaler()
    features = pd.DataFrame(scaler.fit_transform(features))
    labels = data[['critical_temp', 'Cluster']]
    out = pd.concat([features, labels], axis=1, join='inner')
    out.columns = data.columns

    return out


# num_way -> number of classes for episode
# num_shot -> number of examples per class
# num_query -> number of query examples per class
def extract_episode(data, num_way, num_shot, num_query):
    # get a list of all unique labels (no repetition)
    unique_labels = np.unique(data['Cluster'])

    # select num_way classes randomly without replacement
    chosen_clusters = np.random.choice(unique_labels, num_way, replace=False)
    # number of examples per selected class (label)
    examples_per_label = num_shot + num_query

    # list to store the episode
    episode_features = []
    episode_temps = []

    # iterate over all selected labels
    for cluster_c in chosen_clusters:
        # get all images with a certain label l
        data_of_cluster_c = data[data['Cluster']==cluster_c]

        chosen_data = pd.DataFrame(np.array(data_of_cluster_c.sample(examples_per_label)))
        chosen_data.columns = data.columns

        # add the chosen images to the episode
        episode_features.append(chosen_data.drop(['critical_temp','Cluster'], axis=1))
        episode_temps.append(chosen_data['critical_temp'])

    # turn python list into a numpy array
    episode_features = np.array(episode_features)
    episode_temps = np.array(episode_temps)

    # convert numpy array to tensor of floats
    episode_features = torch.from_numpy(episode_features).float()
    episode_temps = torch.from_numpy(episode_temps).float()

    # get the shape of the images
    feature_dim = episode_features.shape[1]

    # build a dict with info about the generated episode
    episode_dict = {
        'features': episode_features, 'temps': episode_temps, 'num_way': num_way, 'num_shot': num_shot,
        'num_query': num_query, 'feature_dim': feature_dim}

    return episode_dict
