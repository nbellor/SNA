import torch
import torch.nn.functional as F


# Graph Encoder Embedding
def GEE(class_vector, adj):

    # compute W
    num_classes = len(torch.unique(class_vector))
    one_hot_matrix = F.one_hot(class_vector, num_classes)
    n_k = torch.bincount(class_vector)
    n_k = 1.0 / n_k.float()
    W = one_hot_matrix.float() * n_k[class_vector].view(-1, 1)

    # compute embeddings
    Z = torch.sparse.mm(adj, W)

    return Z