from brian2 import mV, ms, volt, second
from brian2 import NeuronGroup

def create_izhikevich_neuron(a,b,c,d,Vmax):
    """Creates a brian2 NeuronGroup that contains a single izhikevich neuron with the given parameters"""
    # Define differential equation for izhikevich neuron
    eqs = '''dvm/dt = (0.04/ms/mV)*vm**2+(5/ms)*vm+140*mV/ms-w + I : volt
            dw/dt = a*(b*vm-w) : volt/second
            I : volt/second'''
    # Define reset function
    reset = '''vm = c
                w = w + d'''
    # Set parameters
    a = a/ms; b = b/ms; c = c * mV; d = d * volt/second
    # Define threshold
    threshold = 'vm>{}*mV'.format(Vmax)
    # Return NeuronGroup object
    return NeuronGroup(1, eqs, threshold = threshold, reset = reset, method = 'euler')