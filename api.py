#way to upload image: endpoint
#way to save the image
#function to make predictions
#show the results

import glob
import networkx as nx 
import pandas as pd 
import os 
import numpy as np
import matplotlib.pyplot as plt
import torch, torch_geometric
from torch_geometric.data import Dataset, DenseDataLoader, Data
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
import os.path as osp
from math import ceil
from Variables import *
from sklearn.metrics import confusion_matrix

from flask import Flask, render_template, request
app = Flask(__name__)
UPLOAD_FOLDER="/Users/mydigitalspace/Downloads/denotia_webapp/static"
DEVICE=torch.device('cpu')
## Vishnu, might want to upload your trained model here
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True):
        super(GNN, self).__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
#         self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
#         self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
#         self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if lin is True:
            self.lin = torch.nn.Linear(2 * hidden_channels + out_channels,
                                       out_channels)
        else:
            self.lin = None


    def forward(self, x, adj, mask=None):
#         batch_size, num_nodes, in_channels = x.size()

        x0 = x
        x1 = F.relu(self.conv1(x0, adj, mask))
        x2 = F.relu(self.conv2(x1, adj, mask))
        x3 = F.relu(self.conv3(x2, adj, mask))

        x = torch.cat([x1, x2, x3], dim=-1)

        if self.lin is not None:
            x = F.relu(self.lin(x))

        return x
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        max_nodes = 1162
        
        num_nodes = ceil(0.25 * max_nodes)
        self.gnn1_pool = GNN(3, 64, num_nodes)
        self.gnn1_embed = GNN(3, 64, 64, lin=False)
        
        
        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GNN(3 * 64, 64, num_nodes)
        self.gnn2_embed = GNN(3 * 64, 64, 64, lin=False)

        
        num_nodes = ceil(0.25 * num_nodes)
        self.gnn3_pool = GNN(3 * 64, 64, num_nodes)
        self.gnn3_embed = GNN(3 * 64, 64, 64, lin=False)

        
        self.lin1 = torch.nn.Linear(3 * 64, 64)
        self.lin2 = torch.nn.Linear(64, 64)
        self.lin3 = torch.nn.Linear(64, 6)

    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask=None)
        x = self.gnn1_embed(x, adj, mask=None)

        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask=None)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        s = self.gnn3_pool(x, adj)
        x = self.gnn3_embed(x, adj)
        
        x, adj, l3, e3 = dense_diff_pool(x, adj, s)

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), l1 + l2 + l3, e1 + e2 + e3

@app.route("/",methods=["GET","POST"])
def upload_predict():
    if request.method == "POST":
        image_file=request.files["image"]
        if image_file:
            image_location = os.path.join(UPLOAD_FOLDER, image_file.filename)
            image_file.save(image_location)
            pred=predict(image_location, MODEL)
            return render_template("index.html",prediction=1)
        return render_template("index.html",prediction=0)


if __name__ == "__main__":
    MODEL = GNN(pretrained="imagenet")
    MODEL.load_state_dict(torch.load("'./fix_vishnunet17.pth'"))
    MODEL.to(DEVICE)
    app.run(port=12000, debug=True)

