# General Notes on Model

## Dataset

We use the QM9 dataset, which has over 130000 molecular graphs. There is a lot of data, but the only bits we worry about in this study are:

These notes come primarily from [this tutorial](notebooks/geometric_gnn_101.ipynb) but also from my data exploration in [this notebook](notebooks/01_explore_qm9.ipynb).

Note: $|V|$ and $|E|$ represent number of vertices and number of edges, respectively.

### Atom/node features `data.x`

Each node in the input graph represents an atom in a molecule.

$11|V|$-dimensional, but we only worry about the first **5 features**, which are one-hot encodings for **H, C, N, O, F**. $\implies 5|V|$-dimensional space.

### Edge indices `data.edge_index`

Each edge in the input graph represents a bond in a molecule. If the edge does not exist, there is no bonding. Since we are not assuming a complete graph, this is a useful feature.

$2|E|$-dimensional object that describes edge connectivity.

### Edge features `data.edge_attr`

$4|E|$-dimensional tensor. One-hot encoding for bond type: single, double, triple or aromatic.

### Atom positions `data.pos`

$3|V|$-dimensional tensor. 3D coordinates of each atom in the molecule. NOTE these are absolute coordinates, so we will need to process them somehow!

### Target `data.y`

$19$-dimensional vector. Each entry corresponds to a molecular property. Here we focus on the HOMO-LUMO gap, which is the fifth (index four).

### E.g. Water (H2O)

Third (i.e. index 2) data object in QM9

#### Node features
```
test_mol.x[:, :5] =
tensor([[0., 0., 0., 1., 0.],
        [1., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0.]])
```

i.e. first atom is O, second and third are H.

#### Edge index
```
tensor([[0, 0, 1, 2],
        [1, 2, 0, 0]])
```

i.e. index 0 and 1 are connected; 0 and 2 are connected.
     then so are 1 and 0; 2 and 0.
     Worth noting these are symmetric since bonds are not directed
     - maybe can make use of this?

#### Edge features

```
test_mol.edge_attr =
tensor([[1., 0., 0., 0.],
        [1., 0., 0., 0.],
        [1., 0., 0., 0.],
        [1., 0., 0., 0.]])
```

i.e. all of them are single bonds.

#### Atom positions

```
tensor([[-3.4400e-02,  9.7750e-01,  7.6000e-03],
        [ 6.4800e-02,  2.0600e-02,  1.5000e-03],
        [ 8.7180e-01,  1.3008e+00,  7.0000e-04]])
```

#### Target

```
test_mol.y[:,4] =
tensor([9.8369])
```

### Additional notes

In [the tutorial](notebooks/geometric_gnn_101.ipynb) we use fully connected graphs. However, I am not convinced this is suitable for a GraphVAE, as that can cause a lot of additional complexity and we are not really using the benefit of knowing bond type and enforcing physicality using edge attributes. Therefore, I think I should stick with a **sparse graph** representation. If this does not work well, maybe we can use fully connected graphs and add an additional edge feature to indicate no bond.

QM9 has several isomers. To avoid leakage, might make sense to split by molecular formula (or at least scaffold) rather than random rows. E.g. strip each molecule to its Bemis‑Murcko scaffold (ring system + linkers), group by scaffold and split data based on scaffolds.

## Plan of action

### Documenting progress

Keeping a log of different versions here, along with .yaml files for each version (v0.yaml, etc.). Store dataset options, model hyper-params, training params, random seed (for reproducibility). Recreate results like `python train.py --config conf/v0.yaml`. I will make each additional version from optional keyword arguments to the models.

All the while writing unit tests using pytest, including one where I simply reconstruct one molecule with an overfit model. Tiny sanity loader test, single-molecule overfit, gradient-flow check (no names, params norms < 1e3).

Document git hashes and use git tags for when each model version is done.

### v0

[v0-structure](v0.pdf)

 - Primarily intended as a pipeline sanity check
 - Padded input with 29 nodes (largest molecule in QM9),
 - Only encode/decode with reconstruction loss,
 - Encoder: simple GCNConv
   - GCNConv x num_layers
   - Ignore edge_attr for now
 - Decoder: complete/dense graph -> 29*(29+1)/2 = 406 edges,
   - Dense, fixed-size tensors
   - Masked padded nodes/edges for cross entropy loss
 - unit tests!
   - [x] tensor dimension matches
   - [x] unit test training functions (a few epoches training, validate, test)
   - [x] NaN values in the model outputs or loss
   - [x] check forward pass for all models succeed
   - [x] fixtures
     - [x] QM9 dataset (with various transforms)
     - [x] dataloader
     - [x] single batch
     - [x] test transformations
       - [x] in particular, check that a molecule from a transformed dataset looks the way you expect
     - [x] single molecule
     - [ ] random dataset

After struggling for hours on a model only to realise the problem was in how the data was preprocessed, I think it is imperative to add, which I somehow overlooked:
 - [ ] unit tests for transforms
   - [ ] sanity checks (e.g. dimensions)
   - [ ] take a known molecule (e.g. water) and feed it in, you should know the exact output; verify that it matches expectations!
 - [x] no self-bonds
 - [x] no bonds involving padding nodes
 - [x] ensure forward pass succeeds with no errors
 - [x] weighted loss function
 - [ ] introduce valence constraints (e.g. O has 2 valence)


### v0.1

 - Variable-size graphs(!!)
 - GINEConv Encoder -- produce edge attributes (not just existence), introduces edge_attr
 - Reconstruct both edges and nodes, where in each case there is a padded "empty" one-hot encodding (i.e. no bond or no atom)
 - Unit tests
   - [ ] can overfit to recreate a molecule

### v0.2 - ?

 - order not decided yet:
   - Coordinate awareness
   - Sparse decoder
   - add KL sampling for regularisation
   - Structured latent space search (reinforcement learning)


# Previous approach's models
Jumped too deep I think. Using a simple approach now and increase complexity incrementally.

## Equivariant Message-Passing NN Layer

TODO

## Equivariant Graph MPNN Predictor

TODO

## MPNN Encoder

TODO

## MPNN Decoder

TODO

## Graph Message-Passing Variational Autoencoder

TODO
