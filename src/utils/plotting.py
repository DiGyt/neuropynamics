import matplotlib.pyplot as plt
import seaborn as sns

# Plotting function
def plot_single_neuron(x, neuron_data, neuron_labels, neuron_colors, spikes = None, spike_color = 'steelblue', input_current = None, input_label = 'Input Current', input_color = 'gold', y_range = None, title = '', x_axis_label = '', y_axis_label = '', input_axis_label = 'Input Current (A)', hline = None):
    
    # Use seaborn style
    with sns.axes_style("dark"):

            # Create first y axis and set size of figure
            fig, ax1 = plt.subplots(figsize=(14,6))

            # Set axis labels
            ax1.set_xlabel(x_axis_label)
            ax1.set_ylabel(y_axis_label)

            # Set y range if given
            if y_range is not None:
                ax1.set_ylim(y_range)

            # Add horizontal line
            if hline is not None:
                ax1.axhline(y = hline, linestyle = '--', linewidth = 1, color = 'gray', label = 'Spike Cutoff')

            # Plot a line for each neuron datapoint
            for idx, y in enumerate(neuron_data):
                # We ignore the first datapoint as these are always 0 and result in weird looking lines
                ax1.plot(x[1:], y[1:], color = neuron_colors[idx], label = neuron_labels[idx])

            # Plot spikes if given
            if spikes is not None and len(spikes) > 0:
                # Ignore the first spike if it is before 100ms (as no current was given)
                if spikes[0] >= 10:
                    ax1.axvline(spikes[0], linestyle = ':', color = spike_color, linewidth = 1, zorder = 0)
                for t in spikes[1:-1]:
                    ax1.axvline(t, linestyle = ':', color = spike_color, linewidth = 1, zorder = 0)
                # Add line for last spike with label
                ax1.axvline(spikes[-1], linestyle = ':', color = spike_color, linewidth = 1, label = 'Spikes', zorder = 0)

            # Show input current if given
            if input_current is not None:
                # Create 2nd y-axis
                ax2 = ax1.twinx()
                # Set axis label
                ax2.set_ylabel(input_axis_label)
                # Draw input line
                ax2.plot(x, input_current, color = input_color, label = input_label)   
            
            # Add legend
            fig.legend(loc="upper right", bbox_to_anchor=(0.825, 0.8))

            # Set title
            plt.title(title)