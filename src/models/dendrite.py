class Dendrite():
    """A dendrite-axon model capable of storing multiple action potentials over a
    course of time steps."""

    def __init__(self, weight=1, temp_delay=1):
        self.weight = weight
        self.temp_delay = temp_delay
        self.action_potentials = []

    def __call__(self, ap_input):
        """Simulate one time step for this dendrite."""

        # simulate the next timestep in the dendrite
        new_ap_state = []
        ap_output = 0
        for ap, t in self.action_potentials:
            # if the AP has travelled through the dendrite, return output
            if t == 0:
                ap_output += ap * self.weight
            # else countdown the timesteps for remaining APs in the dendrite
            else:
                new_ap_state.append((ap, t - 1))

        self.action_potentials = new_ap_state

        # enter a new AP into the dendrite
        if ap_input != 0:
            self.action_potentials.append((ap_input, self.temp_delay))

        return ap_output
