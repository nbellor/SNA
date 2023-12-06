import time
import warnings
import torch
import utils
import torch.nn.functional as F
from torch_geometric.nn.models.basic_gnn import GCN, GAT, MLP
from models import GEE

import argparse

import wandb
wandb.login()

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
    return parser.parse_args()

args = parse_args()


def main():

    if args.model == 'gat':
        wandb.init(config=sweep_configuration_gat)
    else:
        wandb.init(config=sweep_configuration) 
    config = wandb.config

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=Warning)
        graph, adj, features, labels, idx_train, idx_val, idx_test = utils.get_dataset(args.dataset)


    if(args.model == 'gcn'):
        model = GCN(in_channels=features.shape[1],
                    hidden_channels=config.hidden_dim,
                    num_layers = config.layers,
                    out_channels=labels.max().item() + 1,
                    dropout=config.dropout,
                    act = 'relu',
                    jk = 'cat')
    elif(args.model == 'gat'):
        model = GAT(in_channels=features.shape[1],
                    hidden_channels=config.hidden_dim,
                    num_layers = config.layers,
                    out_channels=labels.max().item() + 1,
                    dropout=config.dropout,
                    heads = config.heads,
                    act = 'relu',
                    jk = 'cat')
    elif(args.model == 'gee'):

        features = torch.cat((features, GEE(labels, adj)), dim=1)

        model = MLP(in_channels=features.shape[1],
                    hidden_channels=config.hidden_dim,
                    num_layers = config.layers,
                    out_channels=labels.max().item() + 1,
                    dropout=config.dropout,
                    act = 'relu')


    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

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
    for epoch in range(1, config.epochs + 1):
        loss = train()
        train_acc, val_acc, tmp_test_acc = test()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        if (epoch)%10==0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
                f'Val: {val_acc:.4f}')
        
        wandb.log({'epoch': epoch, 'train_loss': loss, 'train_acc': train_acc,'val_acc': best_val_acc})

    print("Optimization Finished!")
    print("Training time: {:.4f}s".format(time.time() - begin))
    print("Best validation accuracy: {:.4f}".format(best_val_acc))
    print("Test accuracy: {:.4f}".format(test_acc))

    wandb.log({'acc': best_val_acc})

sweep_configuration = {
    'method': 'bayes',
    'metric': 
    {
        'goal': 'maximize', 
        'name': 'acc'
        },
    'parameters': 
    {
        'layers': {'values': [1, 2, 3, 4, 5]},
        'hidden_dim': {'values': [16, 32, 64, 128, 256, 512]},
        'dropout': {'min': 0.05, 'max': 0.95},

        'lr': {'min': 0.001, 'max': 0.01},
        'weight_decay': {'min': 0.0001, 'max': 0.01},
        'epochs': {'values': [20, 40, 60, 80, 100]}
     }
}

sweep_configuration_gat = {
    'method': 'bayes',
    'metric': 
    {
        'goal': 'maximize', 
        'name': 'acc'
        },
    'parameters': 
    {
        'layers': {'values': [1, 2, 3, 4, 5]},
        'hidden_dim': {'values': [16, 32, 64, 128, 256, 512]},
        'dropout': {'min': 0.05, 'max': 0.95},

        'lr': {'min': 0.001, 'max': 0.01},
        'weight_decay': {'min': 0.0001, 'max': 0.01},
        'epochs': {'values': [20, 40, 60, 80, 100]},
        'heads': {'values': [2, 4, 8, 16, 32]}
     }
}



# Start the sweep
if args.model == 'gat':
    sweep_id = wandb.sweep(
        sweep=sweep_configuration_gat, 
        project='SNA_' + args.dataset + '_' + args.model
        )
else:
    sweep_id = wandb.sweep(
        sweep=sweep_configuration, 
        project='SNA_' + args.dataset + '_' + args.model
        )

wandb.agent(sweep_id, function=main, count=100)