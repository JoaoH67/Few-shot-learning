import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import euclidean_dist, MAPE_loss


device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class ProtoNet(nn.Module):
    def __init__(self, encoder):
        super(ProtoNet, self).__init__()
        self.encoder = encoder.to(device)
        self.loss = nn.L1Loss()

    def set_forward_loss(self, episode_dict):
        # extract all images
        features = episode_dict['features'].to(device)
        temps = episode_dict['temps'].to(device)

        # get episode setup
        num_way = episode_dict['num_way']  # way
        num_shot = episode_dict['num_shot']  # shot
        num_query = episode_dict['num_query']  # number of query images

        # from each class, extract num_shot support images
        x_support = features[:, :num_shot]  # lines are classes and columns are images
        # from each class, extract the remaining images as query images
        x_query = features[:, num_shot:]  # lines are classes and columns are images
        # transform into a array in which all images are contiguous
        x_support = x_support.contiguous().view(num_way * num_shot, -1)  # no more lines and columns
        x_query = x_query.contiguous().view(num_way * num_query, -1)

        y_support = temps[:, :num_shot].to(device)
        y_query = temps[:, num_shot:].to(device)
        y_support = y_support.contiguous().view(num_way * num_shot)
        y_query = y_query.contiguous().view(num_way * num_query)

        # join all images into a single contiguous array
        x = torch.cat([x_support, x_query], 0)
        # encode all images
        z = self.encoder.forward(x)  # embeddings
        # compute class prototypes
        z_dim = z.size(-1)

        zs = z[:num_way*num_shot] # zs: num_way*num_shot X z_dim
        zq = z[num_way*num_shot:] # zq: num_way*num_query X z_dim

        dists = euclidean_dist(zq, zs)
        p_y = F.softmax(-dists, dim=1)
        y_hat = torch.tensordot(p_y, y_support, dims=([1],[0]))

        loss_val = self.loss(y_hat, y_query)

        return loss_val


# function to load the model structure
def load_protonet(x_dim, hid_dim, z_dim):
    # define a convolutional block
    def linear_block(layer_input, layer_output):
        conv = nn.Sequential(
            nn.Linear(layer_input, layer_output),
            nn.ReLU()
            )

        return conv

    # create the encoder to the embeddings for the images
    # the encoder is made of four conv blocks
    encoder = nn.Sequential(
        linear_block(x_dim, hid_dim),
        linear_block(hid_dim, hid_dim),
        linear_block(hid_dim, hid_dim),
        linear_block(hid_dim, z_dim)
        )

    return ProtoNet(encoder)
