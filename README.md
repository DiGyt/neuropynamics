# Neuropynamics

This toolbox is the result of a student group project for the course "Neurodynamics (Lecture + Tutorial)" held in SS2020 by Prof. Gordon Pipa at the University of Osnabrück.

The goal of this project was to implement simple Python examples for specific topics covered in the lecture. These examples can be used by future neurodynamics students to practically experience and investigate certain phenomena occuring in neural dynamical systems. For this, we will use Python's [brian2](https://brian2.readthedocs.io/en/stable/) toolbox, which is an extensive library created to implement and simulate various neurodynamic processes. We try to give simple examples on how to use the brian2 workflow, and to give students the opportunity to interact with these simulations using ipywidgets for jupyter notebooks.

Each topic is presented as a Jupyter Notebook that should be run within Google Colaboratory. Thus, no pulling or installation of this repository is needed. To get started, check the description of the available topics below and click the links to open the corresponding notebooks. 

## Available notebooks

### Single neurons

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DiGyt/neuropynamics/blob/master/notebooks/Single_neurons.ipynb)

We suggest you start your journey with this notebook as it introduces you to the brian2 toolbox that will be used throughout our examples. 
In this notebook you can interactively examine the spiking and reset behavior of different types of single neurons. We will show the equations for the underlying dynamical systems for the neuron models and implement them using brian2. 

---

### Stability analysis for different neuron models
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DiGyt/neuropynamics/blob/master/notebooks/Stability_analysis.ipynb)

Expanding on the notebook above, this one provides an introduction to stability analysis of 1D and 2D dynamical systems.

---

### Multi-neuron networks

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DiGyt/neuropynamics/blob/master/notebooks/multineuron_networks.ipynb)

In this notebook, you will learn how to build multineuron networks with brian2 and visualize them using our neuropynamics plotting functions.

---

### Dendritic computation

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DiGyt/neuropynamics/blob/master/notebooks/dendritic_computation.ipynb)

Learn what difference dendritic computation can make in the localization of a sound stimulus and see how it changes depending on the timing of sound reaching the left ear and right ear.

---

### Bifurcation and chaotic behavior

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DiGyt/neuropynamics/blob/master/notebooks/bifurcation.ipynb)

Investigate complex and chaotic behavior resulting from the interaction of different parameters in different neuron models.

---

### How to do Stability Analysis - Examining Dynamical Systems
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DiGyt/neuropynamics/blob/master/notebooks/pplane.ipynb)

Stability analysis is widely done in a [Java based software](https://www.cs.unm.edu/~joel/dfield/). We have implemented it as a Python notebook. The notebook provides a clean interface to plug in all the parameters for Non-Linear Ordinary Differential Equations. The code is clean and reusable. It can be extended to include more types of differential equations and controls over the plotting.


---

