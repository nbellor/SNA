# Statistical Network Analisys - Project

### Python environment setup with Conda

```bash
conda create --name SNA python=3.9
conda activate SNA

# script does not support cuda, cpu only
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 cpuonly -c pytorch
conda install -c dglteam dgl=1.1.2

# Python Geometric version is 2.4.0
conda install pyg -c pyg

conda clean --all
```

### Run tests on the datasets 

```bash
python train.py --dataset=cora --model=gcn --dropout=0.85 --epochs=100 --hidden_dim=512 --layers=3 --lr=0.008 --weight_decay=0.009
python train.py --dataset=cora --model=gat --dropout=0.5 --epochs=100 --hidden_dim=512 --layers=3 --lr=0.007 --weight_decay=0.0015 --heads=32
python train.py --dataset=cora --model=gee --dropout=0.5 --epochs=100 --hidden_dim=256 --layers=1 --lr=0.005 --weight_decay=0.004

python train.py --dataset=pubmed --model=gcn --dropout=0.1 --epochs=100 --hidden_dim=128 --layers=4 --lr=0.001 --weight_decay=0.009
python train.py --dataset=pubmed --model=gat --dropout=0.3 --epochs=100 --hidden_dim=256 --layers=4 --lr=0.008 --weight_decay=0.004 --heads=16
python train.py --dataset=pubmed --model=gee --dropout=0.9 --epochs=100 --hidden_dim=256 --layers=3 --lr=0.005 --weight_decay=0.009

python train.py --dataset=wisconsin --model=gcn --dropout=0.9 --epochs=40 --hidden_dim=512 --layers=4 --lr=0.006 --weight_decay=0.009
python train.py --dataset=wisconsin --model=gat --dropout=0.75 --epochs=100 --hidden_dim=64 --layers=1 --lr=0.0015 --weight_decay=0.008 --heads=4
python train.py --dataset=wisconsin --model=gee --dropout=0.7 --epochs=100 --hidden_dim=256 --layers=2 --lr=0.009 --weight_decay=0.006

python train.py --dataset=texas --model=gcn --dropout=0.2 --epochs=100 --hidden_dim=512 --layers=2 --lr=0.008 --weight_decay=0.008
python train.py --dataset=texas --model=gat --dropout=0.2 --epochs=100 --hidden_dim=512 --layers=5 --lr=0.002 --weight_decay=0.001 --heads=32
python train.py --dataset=texas --model=gee --dropout=0.5 --epochs=100 --hidden_dim=512 --layers=4 --lr=0.008 --weight_decay=0.009

```

