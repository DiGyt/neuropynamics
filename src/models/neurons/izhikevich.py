class IzhikevichNeuron():
    """Implementation of an Izhikevich Neuron."""

    def __init__(self, a=0.02, b=0.2, c=-65, d=8,
                 dt=0.5, Vmax=35, V0=-65, u0=-14):
        # Initialize starting parameters for our neuron
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.dt = dt
        self.Vmax = Vmax
        self.V = V0
        self.u = u0
        self.I = 0

    def __call__(self, I):
        """Simulate one timestep of our Izhikevich Model."""

        if self.V < self.Vmax:  # build up spiking potential
            # calculate the membrane potential
            dv = (0.04 * self.V + 5) * self.V + 140 - self.u
            V = self.V + (dv + self.I) * self.dt
            # calculate the recovery variable
            du = self.a * (self.b * self.V - self.u)
            u = self.u + self.dt * du

        else:  # spiking potential is reached
            V = self.c
            u = self.u + self.d

        # limit the spikes at Vmax
        V = self.Vmax if V > self.Vmax else V

        # assign the t-1 states of the model
        self.V = V
        self.u = u
        self.I = I
        return V
