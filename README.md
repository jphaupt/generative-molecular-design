# Generative Molecular Design

This is a toy problem for my own amusement and professional development to use graph neural networks for generative molecular design, analogous to how molecules might be generated for drug discovery.

For a more complete, but much more chaotic set of notes, see my [general notes](notes.md).

## Purpose

The goal of this project is to write a neural network that uses a graph representation of molecules and can generate new, physically reasonable molecules with desirable properties. Since this is a personal project and I am just using my own modest hardware, I will restrict myself to the QM9 dataset, and simply maximise the HOMO-LUMO gap of a molecule as a proxy for e.g. drug stability or protein binding affinities.

The eventual goal is to use a graph variational autoencoder (or a similar generative model, e.g. autoregressive models) with a structured latent space search (e.g. policy gradients). PySCF would be used to calculate the HOMO-LUMO gap for new molecules as the model explores the latent landscape.

## Project Structure

I have explorative notebooks in the `notebooks` directory as well as training for the models I define. My models and helper functions are defined in a package called `mygenai` I have in this repo. This already is thoroughly unit tested in the `tests` directory. It contains the model specification, training functions, PySCF helper functions, data transformations and more.

## Progress

### Initial Attempt

I admittedly underestimated the complexity of the task and thought I could finish this quickly by jumping straight into an **equivariant message-passing graph variational autoencoder with variable input and output**, and a basic reinforcement learning algorithm. This ended up being a bit of a mess. While I have more or less scrapped the actual model, some of the underlying helper functions are still there, e.g. `mygenai/utils/pyscf_utils.py`.

Once I backpedalled a bit, I also came across a tutorial for a course in Cambridge, for which I have included my (partial) solutions [here](notebooks/geometric_gnn_101.ipynb).

### v0

`notebooks/v0.ipynb`

I am working with a much simpler model, making sure it is robust, and then *slowly* scaling to more complex models. My v0 is a graph variational autoencoder designed as a proof-of-concept pipeline for molecular generation that:
- Works on the QM9 dataset
- Uses fixed-sized representation (pads all molecules to 29 nodes)
- Focuses only on bond predictions, which is represented in graphs by the edge attributes
- Enforces basic constraints such as no self-bonds and no bonds to padding nodes using a mask in the decoder and weighting in the loss function

#### Architecture

Encoder
- Takes molecular graphs, which are padded to 29 nodes by a data transform, as input
- Processes through 4 (by default) GCNConv layers
- Pools to graph-level embeddings and projects to latent space
- Currently doesn't use any regularisation

Decoder
- Projects latent vectors to node embeddings
- Predicts edge types between all nodes pairs using an MLP
- Applies basic constraints using masking

#### Key Issues

- Very basic architecture
  - It is able to overfit to a single molecule and correctly output the result, but as soon as there is a more diverse dataset, it tends to predict incorrect methods, e.g. H-H bonding in water
- Padding is extremely excessive
  - I would estimate over 90% of computation is just handling padding
  - Overwhelming number of "no bond" (and "no atom") edges distorts learning
- Can only predict edge properties, and only for fixed graph sizes

### v1+

v0 proves the basic pipeline works, but it leaves much to be desired. Since continuing with the current structure would add a lot of unnecessary complexity, I think the next step is **moving to variable size graphs!**
- Natural representation: each molecule is represented by the correct number of nodes and edges
- no wated computation on padding logic
- better learning
- PyG is designed for sparse reprentation; fighting against that design was purely for convenience, and now it has become inconvenient

**NOTE**: I will be putting it on hold here for now, as I try to finish off a project for work.

#### Further TODOs

- [ ] variable sized graphs
- [ ] Introduce valence contrains (e.g. oxygen has 2 valence)
- [ ] message-passing
- [ ] equivariant layers
- [ ] hyperparameter tweaking/optimisation to get good training results
- [ ] PySCF to generate new molecules and augment dataset
