import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class BNN():
    """A biological neural network connecting multiple neuron models."""

    def __init__(self, neurons, connections):
        self.neurons = neurons
        self.connections = connections
        self.neuron_states = np.zeros(len(neurons))

    def __call__(self, inputs=[0]):
        """Simulates one timestep in our BNN, while allowing additional external
        input being passed as a list of max length = len(BNN.neurons), where
        one inputs[i] corresponds to an action potential entered into BNN.neurons[i]
        at this timestep."""

        # add the external inputs to the propagated neuron inputs
        padded_inputs = np.pad(
            inputs, (0, len(self.neurons) - len(inputs)), 'constant')
        neuron_inputs = self.neuron_states + padded_inputs

        # process all the neurons
        # TODO: neuron outputs are atm represented as the deviation from their respective V0 value
        neuron_outputs = [neuron(i) - neuron.c for neuron,
                          i in zip(self.neurons, neuron_inputs)]

        # update the future neuron inputs by propagating them through the connections
        neuron_states = np.zeros(len(self.neurons))
        for (afferent, efferent, connection) in self.connections:
            neuron_states[efferent] += connection(neuron_outputs[afferent])

        # we need to round in order to prevent rounding errors
        #neuron_states = np.round(neuron_states, 9)
        self.neuron_states = neuron_states

        return neuron_outputs

    # TODO: The plotting function is really ugly and should be redone.
    def plot(self, pos_func=nx.circular_layout, **kwargs):
        """A crude way of plotting the network, by transforming it to a networkX graph."""

        graph = nx.MultiDiGraph()
        graph.add_nodes_from([0, len(self.neurons) - 1])
        graph.add_edges_from([(eff, aff, connection.temp_delay)
                              for aff, eff, connection in self.connections])

        pos = pos_func(graph)
        nx.draw_networkx_nodes(graph, pos, **kwargs)
        ax = plt.gca()
        for e in graph.edges:
            ax.annotate("",
                        xy=pos[e[0]], xycoords='data',
                        xytext=pos[e[1]], textcoords='data',
                        arrowprops=dict(arrowstyle="->", color="0.5",
                                        shrinkA=10, shrinkB=10,
                                        patchA=None, patchB=None,
                                        connectionstyle="arc3,rad=rrr".replace('rrr', str(0.05 + 0.1*e[2])),),)
        plt.axis('off')
        plt.show()
