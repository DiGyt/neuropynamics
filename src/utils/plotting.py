import matplotlib.pyplot as plt
import seaborn as sns

# Apply default seaborn settings
sns.set()

def create_default_plot(x, y, title = '', x_label = '', y_label = ''):
    # Define plotting area and set size
    plt.figure(figsize=(14,8))
    # Create lineplot of the data
    sns.lineplot(x, y)
    # Set labels
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # Set title
    plt.title(title)