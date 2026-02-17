# GraphFold
Graph-based trimodal architecture for protein structure embeddings with contrastive (CLIP-style) alignment.

# Overview
GraphFold is a trimodal learning framework for protein structure modeling that combines graph, image, and sequence representations into a shared embedding space.

The system integrates:
  1. Graph-based structural encoding derived from contact maps
  2. Pretrained sequence embeddings used as node features
  3. Vision-based structural features from contact map images

Transformer-based components are used for representation learning, and a contrastive (CLIP-style) objective is designed to align modalities into a unified embedding space. The framework also supports supervised classification for structural categories.

The architecture is still under works, and testing/abalations are yet to be conducted.
