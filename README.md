# scCAGN
*A self-supervised clustering method combining adversarial autoencoders with cross-attention-based graph convolutional networks (GCN) for cell type identification in scRNA-seq datasets.*

## Environment Configuration
The code runs under **Python 3.9.20** and **Pyorch 2.3.1+cu121**. Create a PyTorch environment and install required packages, such as "numpy", "pandas", "scikit-learn", "scanpy" and "cudatoolkit".
Please refer to requirements.txt for more details.

### Installation:
```python
pip install -r requirements.txt
```

## Data Preprocessing
Supports CSV/h5 formats. Processed features will be saved as output files. Examples:

**For CSV data (Yan dataset):**
```python
python preprocess.py --name Yan --file_format csv
```
**For h5 data (Adam dataset):**
```python
python preprocess.py --name  Adam --file_format h5
```

## Model Training

Optimal parameters are automatically saved during training:

```python
python Cluster.py --name Yan --pretain True --pretrain_epoch 50 --lr 0.001 --device cuda 
```

### Key Parameters
```python
--name: Dataset name (e.g., Yan/Adam)
--pretain: Enable pre-training (True/False)
--pretrain_epoch: Pre-training epochs (default: 50)
--lr: Learning rate (default: 0.001)
--device: `cuda` or `cpu`
```

### Output Files
**pre_embedding.txt**: Latent representations

**pre_label.csv**: Predicted cluster labels

## Code Structure

```python
preprocess.py: Data preprocessing pipelines
pretrain.py:   Adversarial autoencoder implementation
scCAGN.py:     Core clustering model
evaluation.py: Metrics (NMI, ARI) computation
GNN.py:        Custom GCN layers
KNN.py:        KNN graph construction & Louvain clustering
utils.py:      Data loading and graph tools
utilss.py:     Cell-type DAG operations
```
