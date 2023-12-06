import time
import warnings
import torch
import utils
import torch.nn.functional as F
from torch_geometric.nn.models.basic_gnn import GCN, GAT, MLP
from models import GEE

import argparse

def parse_args():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser()

    # main parameters
    parser.add_argument('--model', type=str, default='gcn', 
                        help='Model, choose from {gcn, gat, gee}')
    parser.add_argument('--dataset', type=str, default='cora', 
                        help='cora/pubmed/wisconsin/texas')
    
    # shared parameters
    parser.add_argument('--layers', type=int, default=2,
                        help='Number of layers')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Latent representation size (shared by all layers)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout probability')
    
    # training parameters
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    
    # GAT parameters
    parser.add_argument('--heads', type=int, default=8,
                        help='Number of attention heads')
    

    return parser.parse_args()

args = parse_args()

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=Warning)
    graph, adj, features, labels, idx_train, idx_val, idx_test = utils.get_dataset(args.dataset)


if(args.model == 'gcn'):
    model = GCN(in_channels=features.shape[1],
                hidden_channels=args.hidden_dim,
                num_layers = args.layers,
                out_channels=labels.max().item() + 1,
                dropout=args.dropout,
                act = 'relu',
                jk = 'cat')
elif(args.model == 'gat'):
    model = GAT(in_channels=features.shape[1],
                hidden_channels=args.hidden_dim,
                num_layers = args.layers,
                out_channels=labels.max().item() + 1,
                dropout=args.dropout,
                heads = args.heads,
                act = 'relu',
                jk = 'cat')
elif(args.model == 'gee'):

    features = torch.cat((features, GEE(labels, adj)), dim=1)

    model = MLP(in_channels=features.shape[1],
                hidden_channels=args.hidden_dim,
                num_layers = args.layers,
                out_channels=labels.max().item() + 1,
                dropout=args.dropout,
                act = 'relu')


optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(features) if args.model == 'gee' else model(features, adj)

    loss = F.cross_entropy(out[idx_train], labels[idx_train])
    loss.backward()
    optimizer.step()

    return float(loss)

@torch.no_grad()
def test():
    model.eval()
    pred = model(features) if args.model == 'gee' else model(features, adj)
    pred = pred.argmax(dim=-1)

    accs = []
    for mask in [idx_train, idx_val, idx_test]:
        accs.append( int((pred[mask] == labels[mask]).sum()) / int(mask.sum()) )

    return accs

begin = time.time()

best_val_acc = test_acc = 0
for epoch in range(1, args.epochs + 1):
    loss = train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    if (epoch)%10==0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
             f'Val: {val_acc:.4f}')

print("Optimization Finished!")
print("Training time: {:.4f}s".format(time.time() - begin))
print("Best validation accuracy: {:.4f}".format(best_val_acc))
print("Test accuracy: {:.4f}".format(test_acc))