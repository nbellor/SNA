
import torch
import dgl

from dgl.data import CoraGraphDataset, PubmedGraphDataset, WisconsinDataset, TexasDataset

def get_dataset(dataset):
    '''
    Loads different datasets and converts to bidirected graph
    train, val and test split is already given

    Parameters
    ----------
    dataset : str
        Name of the dataset

    Returns
    -------
    adj : torch.Tensor
        Adjacency matrix
    features : torch.Tensor
        Feature matrix
    labels : torch.Tensor
        Labels
    idx_train : torch.Tensor
        Training set
    idx_val : torch.Tensor
        Validation set
    idx_test : torch.Tensor
        Test set
    '''
    
    if dataset in {"pubmed", "cora", "wisconsin", "texas"}:

        file_path = "data/" + dataset + ".pt"

        if dataset == "pubmed":
            graph = PubmedGraphDataset("data/" + dataset)[0]
        elif dataset == "cora":
            graph = CoraGraphDataset("data/" + dataset)[0]
        elif dataset == "wisconsin":
            graph = WisconsinDataset("data/" + dataset)[0]
        elif dataset == "texas":
            graph = TexasDataset("data/" + dataset)[0]

        adj = data_list[0]
        features = data_list[1]
        labels = graph.ndata['label']

        idx_train = graph.ndata['train_mask']
        idx_val = graph.ndata['val_mask']
        idx_test = graph.ndata['test_mask']

        graph = dgl.to_bidirected(graph)

    return adj, features, labels, idx_train, idx_val, idx_test
