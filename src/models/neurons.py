from brian2 import NeuronGroup

def create_izhikevich_neuron(Vmax = 35):
    """Creates a brian2 NeuronGroup that contains a single izhikevich neuron"""
    # Define differential equation for izhikevich neuron
    eqs = '''   dvm/dt = (0.04/ms/mV)*vm**2+(5/ms)*vm+140*mV/ms-w + I : volt
                dw/dt = a*(b*vm-w) : volt/second
                I : volt/second '''
    # Define reset function
    reset = ''' vm = c
                w = w + d '''    
    # Define threshold
    threshold = 'vm>{}*mV'.format(Vmax)
    # Return NeuronGroup object
    return NeuronGroup(1, eqs, threshold = threshold, reset = reset, method = 'euler')