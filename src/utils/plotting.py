import matplotlib.pyplot as plt
import seaborn as sns

# Apply default seaborn settings
sns.set()
sns.set_style({'axes.grid' : False})

# Plotting function
def create_default_plot(x, neuron_data, neuron_labels, neuron_colors, spikes = None, spike_color = 'steelblue', input_current = None, input_label = 'Input Current', input_color = 'gold', y_range = None, title = '', x_axis_label = '', y_axis_label = '', input_axis_label = 'Input Current (A)', zeroline = True):
    
    # Create first y axis and set size of figure
    fig, ax1 = plt.subplots(figsize=(14,6))

    # Set axis labels
    ax1.set_xlabel(x_axis_label)
    ax1.set_ylabel(y_axis_label)

    # Set y range if given
    if y_range is not None:
        ax1.set_ylim(y_range)

    # Add horizontal 0 line
    if zeroline:
        ax1.axhline(y = 0, linestyle = '--', linewidth = 1, color = 'gray')

    # Plot a line for each neuron datapoint
    for idx, y in enumerate(neuron_data):
        ax1.plot(x, y, color = neuron_colors[idx], label = neuron_labels[idx])

    # Plot spikes if given
    if spikes is not None:
        ax1.scatter(spikes[0], spikes[1], color = spike_color, label = 'Spikes')

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