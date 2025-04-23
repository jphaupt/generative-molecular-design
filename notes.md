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

### Atom positions `data.pos`

### Target `data.y`

### E.g. Water (H2O)

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


#### Atom positions

#### Target


### Additional notes

In [the tutorial](notebooks/geometric_gnn_101.ipynb) we use fully connected graphs. However, I am not convinced this is suitable for a GraphVAE, as that can cause a lot of additional complexity and we are not really using the benefit of knowing bond type and enforcing physicality using edge attributes. Therefore, I think I should stick with a **sparse graph** representation.

## Equivariant Message-Passing NN Layer


## Equivariant Graph MPNN Predictor

## MPNN Encoder

## MPNN Decoder

## Graph Message-Passing Variational Autoencoder
