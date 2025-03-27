# TensorNetworkQuantumSimulator

A package for simulating quantum circuits with near-arbitrary geometry using tensor networks (TNs). 

The main workhorses of the simulation are _belief propagation_ (BP) for gauging the TNs, _simple update_ for applying the gates, and BP with _loop corrections_ or _boundary MPS_ for estimating expectation values. This package is an experimental comilation of state-of-the-art features before some of them get integrated into [ITensorNetworks](https://github.com/ITensor/ITensorNetworks.jl) over time. The focus on simulating arbitrary quantum circuits will likely remain part of this package.

The workflow is that you pass a `NamedGraph` object to the tensor network constructor, which describes the connectivity of your quantum circuit. This will create a TN matching the topology. Then you easily define and apply gates, and estimate expectation values with any of the available techniques, each having different advantages.

## Upcoming Features
- Gates beyond Pauli rotations, for example, Clifford gates.
- Applying gates to distant nodes of the TN via SWAP gates.
- Sampling from loopy networks.

## Relevant Literature
- [Gauging tensor networks with belief propagation](https://www.scipost.org/SciPostPhys.15.6.222?acad_field_slug=chemistry)
- [Efficient Tensor Network Simulation of IBM’s Eagle Kicked Ising Experiment](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.5.010308)
- [Loop Series Expansions for Tensor Networks](https://arxiv.org/abs/2409.03108)
- [Dynamics of disordered quantum systems with two- and three-dimensional tensor networks](https://arxiv.org/abs/2503.05693)

## Authors
The TN methods were developed and written by Joseph Tindall ([JoeyT1994](https://github.com/JoeyT1994)), a Postdoctoral Researcher at the Center for Computational Quantum Physics, Flatiron Institute NYC.

The package was developed with Manuel S. Rudolph ([MSRudolph](https://github.com/MSRudolph)), a PhD Candidate at EPFL, Switzerland, during a research stay at the Center for Computational Quantum Physics, Flatiron Institute NYC.

