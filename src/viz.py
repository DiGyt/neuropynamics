import numpy as np
import matplotlib.pyplot as plt


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
  """Show multiple signals in a colormap

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
