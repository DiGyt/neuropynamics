from brian2 import NeuronGroup

def create_perfect_integrator_neuron(Vmax):
    """Creates a brian2 NeuronGroup that contains a single perfect integrator neuron"""
    # Define differential equation for perfect integrator neuron
    eqs = '''   
        dvm/dt = I/tau : volt
        I : volt
        '''
    # Define reset function
    reset = 'vm = {}*mV'.format(Vmax-20)
    # Define threshold
    threshold = 'vm > {}*mV'.format(Vmax)
    # Return NeuronGroup object
    return NeuronGroup(1, eqs, threshold = threshold, reset = reset, method = 'euler')

def create_lif_neuron(Vmax):
    """Creates a brian2 NeuronGroup that contains a single leaky integrate-and-fire neuron"""
    # Define differential equation for leaky integrate-and-fire neuron
    eqs = '''   
        dvm/dt = ((El - vm) + I)/tau : volt
        I : volt
        '''
    # Define reset function
    reset = 'vm = {}*mV'.format(Vmax-20)
    # Define threshold
    threshold = 'vm > {}*mV'.format(Vmax)
    # Return NeuronGroup object
    return NeuronGroup(1, eqs, threshold = threshold, reset = reset, method = 'euler')

def create_izhikevich_neuron(Vmax):
    """Creates a brian2 NeuronGroup that contains a single izhikevich neuron"""
    # Define differential equation for izhikevich neuron
    eqs = '''   
        dvm/dt = (0.04/ms/mV)*vm**2+(5/ms)*vm+140*mV/ms-w + I : volt
        dw/dt = a*(b*vm-w) : volt/second
        I : volt/second 
        '''
    # Define reset function
    reset = ''' 
        vm = c
        w = w + d
        '''    
    # Define threshold
    threshold = 'vm > {}*mV'.format(Vmax)
    # Return NeuronGroup object
    return NeuronGroup(1, eqs, threshold = threshold, reset = reset, method = 'euler')

def create_hodgkin_huxley_neuron(Vmax):
    """Creates a brian2 NeuronGroup that contains a single hodgkin-huxley neuron"""    
    # Define differential equation for hodgkin-huxley neuron
    eqs = '''
        dv/dt = (gl*(El-v) - g_na*(m*m*m)*h*(v-ENa) - g_kd*(n*n*n*n)*(v-EK) + I)/Cm : volt
        dm/dt = 0.32*(mV**-1)*(13.*mV-v+VT)/
            (exp((13.*mV-v+VT)/(4.*mV))-1.)/ms*(1-m)-0.28*(mV**-1)*(v-VT-40.*mV)/
            (exp((v-VT-40.*mV)/(5.*mV))-1.)/ms*m : 1
        dn/dt = 0.032*(mV**-1)*(15.*mV-v+VT)/
            (exp((15.*mV-v+VT)/(5.*mV))-1.)/ms*(1.-n)-.5*exp((10.*mV-v+VT)/(40.*mV))/ms*n : 1
        dh/dt = 0.128*exp((17.*mV-v+VT)/(18.*mV))/ms*(1.-h)-4./(1+exp((40.*mV-v+VT)/(5.*mV)))/ms*h : 1
        I : amp
        '''   
    # Define threshold and refractory period
    threshold = 'v > {}*mV'.format(Vmax)
    refractory = 'v > {}*mV'.format(Vmax)
    # Return NeuronGroup object
    return NeuronGroup(1, eqs, threshold = threshold, refractory = refractory, method='exponential_euler')