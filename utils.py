
import torch
import dgl
from torch.utils.data import Dataset


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

        file_path = "data/" + dataset

        if dataset == "pubmed":
            graph = PubmedGraphDataset(file_path)[0]
        elif dataset == "cora":
            graph = CoraGraphDataset(file_path)[0]
        elif dataset == "wisconsin":
            graph = WisconsinDataset(file_path)[0]
        elif dataset == "texas":
            graph = TexasDataset(file_path)[0]

        features = graph.ndata['feat']
        labels = graph.ndata['label']

        idx_train = graph.ndata['train_mask'] if dataset in {"cora", "pubmed"} else graph.ndata['train_mask'][:,0]
        idx_val = graph.ndata['val_mask'] if dataset in {"cora", "pubmed"} else graph.ndata['val_mask'][:,0]
        idx_test = graph.ndata['test_mask'] if dataset in {"cora", "pubmed"} else graph.ndata['val_mask'][:,0]

        graph = dgl.to_bidirected(graph)
        adj = graph.adjacency_matrix()
        adj = torch.sparse_coo_tensor(adj.indices(), adj.val, adj.shape)
        adj = adj.to_sparse_csr()

    return graph, adj, features, labels, idx_train, idx_val, idx_test
