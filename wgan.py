import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
import pickle
import sys
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GraphConv, global_mean_pool
from torch_geometric.loader import DataLoader as Loader
sys.path.append("classes")
from pin import Pin
from delayobject import DelayObject
from node import Node
import torch.autograd as autograd
from tqdm import tqdm
from torch_geometric.utils import dense_to_sparse

# Loading the object_dictionary containing information of DelayObjects
with open('../newPickle/object_dictionary.pickle', 'rb') as f:
    object_dictionary = pickle.load(f)

# Loading the data_loader_list containing information of DataLoader Objects
with open('data_loader_list.pkl', 'rb') as file:
    data_loader_list = pickle.load(file)

data_objects = []
for data in data_loader_list:
    if 20 <= len(data.graph) and len(data.graph) <=30:
        data_objects.append(data)

cell_set = set()
names_set = set()
for data in data_objects:
    graph = data.graph
    for node in list(graph.keys()):
        cell_set.add(node.cell) 
        names_set.add(node.name)

one_hot_labels = {}
cell_list = list(cell_set)
for i in range(len(cell_list)):
    binary_str = bin(i)[2:]
    binary_str = binary_str.zfill(6)
    one_hot_labels[cell_list[i]] = [int(d) for d in binary_str]

def get_node_features(graph, one_hot_labels):
    node_features = []
    for node in list(graph.keys()):
        node_features.append(one_hot_labels[node.cell])
    node_features_tensor = torch.tensor(node_features, dtype=torch.float32)
    return node_features_tensor

dataset = []

for data in data_objects:
    x = get_node_features(data.graph, one_hot_labels)  # assuming one_hot_labels is defined somewhere
    edge_index = data.edge_index
    y = torch.tensor(1)  # Assuming your y label is a scalar tensor
    dataset.append(Data(x=x, edge_index=edge_index, y=y))
    

# nxm , m = n+ node_features, node_features: binary encoded values for each feature

def adjacency_matrix_to_edge_index(adj_matrix):
    # Get the size of the adjacency matrix
    num_nodes = adj_matrix.size(0)

    # Initialize lists to store edge indices
    edge_index = []

    # Iterate through the adjacency matrix to find non-zero entries
    for i in range(num_nodes):
        for j in range(num_nodes):
            if (adj_matrix[i][j]) != 0:
                # Add the edge to the edge list
                edge_index.append([i, j])

    # Convert the edge list to a torch tensor
    edge_index = torch.tensor(edge_index).t().contiguous()

    return edge_index

def pad_features(tensor, target_size = 30):
    if tensor.size(0) == target_size:
        return tensor
    return torch.cat((tensor,torch.zeros(target_size-tensor.size(0),tensor.size(1))),dim=0)

def pad_matrix(matrix, target_size = 30):
    n=matrix.size(0)
    if n==target_size:
        return matrix
    matrix = torch.cat((matrix,torch.zeros(target_size-n,n)),dim=0)
    matrix = torch.cat((matrix,torch.zeros(target_size,target_size-n)),dim=1)
    return matrix

def get_adjacency_matrix(edge_index, num_nodes=None, sparse=True):

    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1

    # If edge_index is on GPU, move it to CPU to perform numpy operations
    edge_index_cpu = edge_index.cpu().numpy()

    # Create an empty adjacency matrix
    adj_matrix = torch.zeros((num_nodes, num_nodes))

    # Fill the adjacency matrix based on edge_index
    adj_matrix[edge_index_cpu[0], edge_index_cpu[1]] = 1

    if not sparse:
        # Convert to dense matrix
        adj_matrix = adj_matrix.to_dense()

    return adj_matrix

for data_obj in dataset:
    edge_index = data_obj.edge_index
    node_feats = data_obj.x
    
    data_obj.edge_index = adjacency_to_edge_index(pad_matrix(edge_index_to_adjacency(edge_index)))
    data_obj.x = pad_features(node_feats)

NUM_NODES = 30
NUM_FEATS = 6
LATENT_DIM = 30*36
adjacency_shape = (NUM_NODES, NUM_NODES)
feature_shape = (NUM_NODES, NUM_FEATS)
dense_units = [1500, 2000, 3000]
dropout_rate = 0.2

def adjacency_to_edge_index(adjacency):
    num_nodes = adjacency.shape[0]
    edge_index = torch.nonzero(adjacency, as_tuple=False).t()
    return edge_index


def edge_index_to_adjacency(edge_index, num_nodes = NUM_NODES):
    adjacency = torch.zeros(num_nodes, num_nodes)
    adjacency[edge_index[0], edge_index[1]] = 1
    adjacency[edge_index[1], edge_index[0]] = 1
    return adjacency

class Generator(nn.Module):
    def __init__(self, dense_units, dropout_rate, latent_dim, adjacency_shape, feature_shape):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.adjacency_shape = adjacency_shape
        self.feature_shape = feature_shape
        
        # Define dense layers
        self.dense_layers = nn.ModuleList()
        for units in dense_units:
            self.dense_layers.append(nn.Linear(latent_dim, units))
            self.dense_layers.append(nn.Tanh())
            self.dense_layers.append(nn.Dropout(dropout_rate))
            latent_dim = units
        
        # Define output layers for adjacency and feature tensors
        self.adj_output = nn.Linear(latent_dim, int(torch.prod(torch.tensor(adjacency_shape))))
        self.feat_output = nn.Linear(latent_dim, int(torch.prod(torch.tensor(feature_shape))))
    
    def forward(self, z):
        x = z
        for layer in self.dense_layers:
            x = layer(x)
        
        x_adjacency = self.adj_output(x)
        x_adjacency = x_adjacency.view(-1, *self.adjacency_shape)
        x_adjacency = (x_adjacency + x_adjacency.transpose(-1, -2)) / 2
        x_adjacency = torch.sigmoid(x_adjacency)
        x_adjacency = (x_adjacency > 0.5).float()  # Binarize to 0 or 1
        
        x_features = self.feat_output(x)
        x_features = x_features.view(-1, *self.feature_shape)
        x_features = torch.sigmoid(x_features)
        x_features = (x_features > 0.5).float()  # Binarize to 0 or 1
        
        return x_adjacency, x_features

def get_fake_samples(model,noise):
    adj, feat = model(noise)
    fake_data = []  
    for i in range(adj.size(0)):
        x = feat[i]
        edge_index = adjacency_to_edge_index(adj[i])  
        y = torch.tensor(0) 
        fake_data.append(Data(x=x,y=y,edge_index=edge_index))
    return Batch.from_data_list(fake_data)

class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Critic, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # layer 1
        x = F.relu(self.conv1(x, edge_index))

        # layer 2
        x = F.relu(self.conv2(x, edge_index))

        # layer 3
        x = F.relu(self.conv3(x, edge_index))

        # Global mean pooling to obtain a graph-level representation
        x = torch_geometric.nn.global_mean_pool(x, batch)

        # Fully connected layer for prediction
        x = self.fc(x)
        
        x = torch.relu(x)

        return x




dataloader = Loader(dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=2)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim_critic = 6
hidden_dim_critic = 80
output_dim_critic = 1

critic = Critic(input_dim_critic, hidden_dim_critic, output_dim_critic).to(device)

generator = Generator(dense_units, dropout_rate, LATENT_DIM, adjacency_shape, feature_shape).to(device)

learning_rate = 0.0001
step = 0

gen_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas = (0.0, 0.9))
critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate, betas = (0.0, 0.9))

num_epochs = 1000
critic_epochs = 5

generator.train()
critic.train()


def gradient_penalty(critic, data, fake, device='cuda'):
    
    batch_size = data.size(0)

    # Generate random epsilon
    epsilon = torch.rand(batch_size, 1).to(device)
    
    # real 
    num_nodes = data.edge_index.max().item() + 1

    real_adj = edge_index_to_adjacency(data.edge_index,num_nodes)
    real_features = data.x
    
    # fake 
    num_nodes = fake.edge_index.max().item() + 1
    
    fake_adj = edge_index_to_adjacency(fake.edge_index,num_nodes)
    fake_features = fake.x

    # batches
    real_batch =data.batch
    fake_batch = fake.batch
    
    target_size = fake_features.size(0)

    interpolated_adj = pad_matrix(real_adj,batch_size) * epsilon + fake_adj * (1 - epsilon)
    
    interpolated_features = real_features * epsilon + fake_features*(1-epsilon)
    
    interpolated_batch = real_batch*epsilon + (1-epsilon)*fake_batch
    interpolated_batch = interpolated_batch.to(torch.int64)
    
    interpolated_edge_index = adjacency_to_edge_index(interpolated_adj)
    
    interpolated_adj.to(device).requires_grad_(True)
    interpolated_features.to(device).requires_grad_(True)
    
    interpolated_samples = Batch(x=interpolated_features, 
                                 edge_index=interpolated_edge_index, batch = interpolated_batch)
    critic_scores = critic(interpolated_samples).view(-1)
    # Calculate gradients of critic with respect to interpolated samples
    gradients = autograd.grad(outputs=critic_scores, inputs=[interpolated_features,interpolated_adj],
                              grad_outputs=torch.ones_like(critic_scores),
                              create_graph=True, retain_graph=True, only_inputs=True,allow_unused=True)[0]

    # Compute gradient penalty
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty




LAMBDA_GP = 10
for epoch in range(num_epochs):
    total_loss_critic = 0.0
    total_loss_gen = 0.0
    
    # Target labels not needed! <3 unsupervised
    for batch_idx, data in enumerate(tqdm(dataloader)):
        data = data.to(device)
        current_batch_size = data.size(0)
        
        # Train Critic: max E[critic(real)] - E[critic(fake)]
        for _ in range(critic_epochs):
            noise = torch.randn(64, LATENT_DIM).to(device)
            fake = get_fake_samples(generator,noise) #returns batch
            critic_real = critic(data).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            gp = gradient_penalty(critic, data, fake, device=device)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            )
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            critic_optimizer.step()


        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        gen_fake = critic(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        generator.zero_grad()
        loss_gen.backward()
        gen_optimizer.step()

        # Accumulate loss for each batch
        total_loss_critic += loss_critic.item()
        total_loss_gen += loss_gen.item()

    # Calculate average loss for the epoch
    avg_loss_critic = total_loss_critic / len(dataloader)
    avg_loss_gen = total_loss_gen / len(dataloader)

    # Print losses for the epoch
    print(
        f"Epoch [{epoch+1}/{num_epochs}] \
          Avg. Loss D: {avg_loss_critic:.4f}, Avg. Loss G: {avg_loss_gen:.4f}"
    )

