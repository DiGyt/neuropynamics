import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def plot_signals(data, times, ch_names, **plotkwargs):
  """Show multiple signals in a stacked plot."""

  fig = plt.figure()

  yprops = dict(rotation=0,
                horizontalalignment='right',
                verticalalignment='center',
                x=-0.01)
  
  high = np.round(np.mean(np.max(np.array(data), axis=1)), 2)
  low = np.round(np.mean(np.min(np.array(data), axis=1)), 2)
  
  axprops = dict(yticks=np.linspace(high + 0.1 * low, low + 0.1 * high, 3))
  axes = []
  step = 0.2
  ax_locs = np.arange(0.1 + step * data.shape[0], 0.1, -step)

  for ind, channel in enumerate(data):

    axes.append(fig.add_axes([0.1, ax_locs[ind], 0.8, 0.2], **axprops))

    axes[ind].plot(times, channel, **plotkwargs)
    axes[ind].set_ylabel(ch_names[ind], **yprops)
    
    axprops['sharex'] = axes[0]
    axprops['sharey'] = axes[0]

    # turn off x ticklabels for all but the lower axes
    if ind != data.shape[0] - 1:
        plt.setp(ax.get_xticklabels(), visible=False)

  plt.xlabel("Time (seconds)")

  plt.show()
  
  
def plot_cmesh(data, times, ch_names):
  """Show multiple signals in a colormap."""

  data, times = np.array(data), np.array(times)

  plt.pcolormesh(data[::-1])
  labels = ch_names[::-1]
  ax = plt.gca()

  x_idx = np.arange(0, len(times)+1, 100)
  plt.xticks(x_idx)
  ax.set_xticklabels(np.round(plt.xticks()[0] * times[1] - times[0], 13))
  plt.xlabel("Time (seconds)")
  
  plt.yticks(np.arange(5)+0.5)
  ax.set_yticklabels(labels)

  plt.colorbar()
  plt.show()


def plot_spikes(spikes, times, ch_names=None, time_unit=None):
  """Plot spiking times as given by brian2 spikemonitors."""

  spikes, times = np.array(spikes), np.array(times)
  max_i = np.max(spikes)

  if ch_names == None:
    ch_names = np.linspace(max_i, 0, max_i+1).astype(int)
  else:
    ch_names = ch_names[::-1]

  if time_unit == None:
    label_x = "Time"
  else:
    label_x = "Time ({})".format(time_unit)

  plt.figure()
  plt.scatter(times, max_i-spikes, marker=".", color="k", linewidths=0.1)
  axes = plt.gca()
  axes.set_yticks(np.linspace(0.5, max_i-0.5, max_i), minor=True)
  axes.set_yticks(np.linspace(0, max_i, max_i+1))
  axes.set_yticklabels(ch_names)
  axes.yaxis.grid(which="minor", alpha=0.4)

  plt.xlabel(label_x)
  plt.show()


def plot_synapses(neuron_groups, synapse_groups, pos_func=nx.circular_layout,
                  color_cycle = ["r", "g", "b", "y"], legend=False,
                  node_size=200):
  """Plot Neural Network Graphs defined by brian2 NeuronGroups and Synapses."""

  def prev_nodes(node_list, n_idx):
    return np.sum(node_list[:n_idx])

  # define some often used values
  node_list = [len(neurons) for neurons in neuron_groups]
  n_groups = len(node_list)

  # add all neurons from the neuron groups to the graph
  graph = nx.MultiDiGraph()
  for nodes in node_list:
    graph.add_nodes_from([1, nodes])

  # go through the synapse, find the matching neuron groups
  # and create network edges from them
  for synapses in synapse_groups:
    for n_idx, group in enumerate(neuron_groups):
      
      if synapses.source == group:
        source_nodes = prev_nodes(node_list, n_idx) + synapses.i
      if synapses.target == group:
        target_nodes = prev_nodes(node_list, n_idx) + synapses.j

    edge_list = np.vstack([source_nodes, target_nodes, synapses.delay]).T
    graph.add_edges_from(edge_list)
  
  # draw the different neuron groups as plot, using different colors each time
  color_map = [color_cycle[idx % len(color_cycle)] for idx in range(n_groups)]
  pos = pos_func(graph)
  for idx in range(n_groups):
    cur_idx = prev_nodes(node_list, idx)
    nx.draw_networkx_nodes(graph, pos,
                           nodelist=np.arange(cur_idx, cur_idx + node_list[idx]),
                           node_color=color_map[idx], node_size=node_size)
    
  # draw the weights to the network graph, based on our synapses
  ax = plt.gca()
  for e in graph.edges:
    ax.annotate("",
                xy=pos[e[1]], xycoords='data',
                xytext=pos[e[0]], textcoords='data',
                arrowprops=dict(arrowstyle="->", color="0.5",
                                shrinkA=10, shrinkB=10,
                                patchA=None, patchB=None,
                                connectionstyle="arc3,rad=rrr".replace('rrr',str(0.05 + 10 * e[2])),),)
  
  # add legend
  if legend == True:
    plt.legend([neurons.name for neurons in neuron_groups], 
               prop={'size': 8})
  elif isinstance(legend, str):
    plt.legend([neurons.name for neurons in neuron_groups],
               prop={'size': 8}, loc=legend)

  # plot it
  plt.axis('off')
  plt.show()
