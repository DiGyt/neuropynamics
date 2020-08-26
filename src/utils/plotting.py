# Plotting imports
import matplotlib.pyplot as plt
from matplotlib.pyplot import quiver
import seaborn as sns

# Imports for displaying and solving dynamical systems
import numpy as np
from sympy import symbols, solve, lambdify, sympify, dsolve, Eq, solveset, linear_eq_to_matrix, nonlinsolve, Matrix, diff, sqrt
from scipy import integrate
from neuropynamics.src.utils.dynamical_systems import rk4, plot_equation, system

# Imports for integration with widgets
from ipywidgets import widgets
from IPython.display import clear_output, display, HTML

# Plotting function for a dynamical system with two expressions
def plot_dynamical_system(expr1, expr2, x, y, start_x, end_x, start_y, end_y, stepsize, numsteps):
    '''
        expr1: sympy expression - non linear ode
        expr2: sympy expression - non linear ode
        x: sympy variable used in expr1 and/or expr2
        y: sympy variable used in expr1 and/or expr2
        start_x: starting value for the x range we work in (also used as starting value for range-kutta4)
        end_x: ending value for the x range we work in
        start_y: starting value for the y range we work in (also used as starting value for range-kutta4)
        end_y: ending value for the y range we work in
        stepsize: step size for runge-kutta4
        numsteps: the number of steps runge-kutta4 should take
    '''
    # converting the equations to functions
    f1 = lambdify((x, y), expr1)
    f2 = lambdify((x, y), expr2)
    
    # phase space
    sx, sy = system(start_x, start_y, f1, f2, stepsize, numsteps)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,10))
    ax1.plot(sx, label='X in Time')
    ax1.plot(sy, label='Y in Time')
    ax2.plot(sx, sy, label='X against Y')
    ax1.legend()
    ax2.legend()
    
    # compute ranges to work in    
    xrange = np.linspace(start_x, end_x)
    yrange = np.linspace(start_y, end_y)
    
    # a plot for the quiver plot and nullclines
    fig, ax = plt.subplots(figsize=(20,20))
    ax.set_xlim([start_x, end_x])
    ax.set_ylim([start_y, end_y])
    
    # plotting the 4 nullclines from range-kutta4
    rkx, rky = rk4( f1, start_x, -1000, .25, end_x )
    ax.scatter(rkx, rky, marker='+', label='Solution with Runge-Kutta 4', s=150)
    
    rky, rkx = rk4( f1, start_x, -1000, .25, end_x )
    ax.scatter(rkx, rky, marker='+', label='Solution with Runge-Kutta 4', s=150)
    
    rkx, rky = rk4( f2, start_y, 0, .25, end_y )
    ax.scatter(rkx, rky, marker='+', label='Solution with Runge-Kutta 4', s=150)
    
    rky, rkx = rk4( f2, start_y, 0, .25, end_y )
    ax.scatter(rkx, rky, marker='+', label='Solution with Runge-Kutta 4', s=150)
    
    # compute quivers
    f1_val = [[f1(i, j) for i in xrange] for j in yrange];
    f2_val = [[f2(i, j) for i in xrange] for j in yrange];

    # plot quiver plot
    q = ax.quiver(xrange, yrange, f1_val, f2_val, alpha=.5)
    
    # solve analytically using sympy
    solutions = solve((Eq(expr1, 0), Eq(expr2, 0)), x, y)
    
    # compute jacobian and eigen values
    equationMatrix = Matrix([ expr1, expr2 ])
    varMat = Matrix([ x, y ])
    jacobian = equationMatrix.jacobian(varMat)
    display(HTML('''<h2>Jacobian Matrix</h2> <br /> The Jacobian matrix has single order derivatives
                    of a System of Equations.</h2>'''))
    display(jacobian)
    
    # display eigen values
    display(HTML('<br /> <br /> <br /> <h2>Eigen Values</h2> <br /> <img src="2d-stability.png" />'))
    op = '''<table>
                     <th>
                         <td>Stable Point</td>
                         <td>Eigen Values</td>
                         <td>Type of Stability Point</td>
                     </th>'''
    for s in solutions:
        eqmat = jacobian.subs([ (x, s[0]), (y, s[1]) ])
        ev = list(eqmat.eigenvals().keys())
        
        if ev[0].is_real:
            if ev[0] > 0 and ev[1] > 0:
                t = 'Unstable Node'
            elif ev[0] < 0 and ev[1] < 0:
                t = 'Stable Node'
            elif (ev[0] < 0 and ev[1] > 0) or (ev[0] > 0 and ev[1] < 0):
                t = 'Saddle Point'
        else:
            if ev[0].args[0] > 0:
                t = 'Unstable Focus'
            if ev[0].args[0] < 0:
                t = 'Stable Focus'
        
        op = op + '''<tr>
                            <td>%s</td>
                            <td>%s, %s</td>
                            <td>%s</td>
                        </tr>''' % (str(s), ev[0], ev[1], t)
    op = op + '</table>'
    display(HTML(op))
    
    display(HTML('''<br /> <br /> <br /> <h2>Phase Potrait</h2> <br />
                 Phase Potrait shows the change in the two quantities with respect to each other
                 and idependantly.<br />'''))
    
    # plot the analytical solution
    try:
        [ax.plot(yrange, nc, c='b', alpha=.7, label='Analytical Solution') for nc in plot_equation(expr1, y, x, yrange)]
        [ax.plot(nc, yrange, c='b', alpha=.7, label='Analytical Solution') for nc in plot_equation(expr1, x, y, xrange)]
        [ax.plot(nc, xrange, c='r', alpha=.7, label='Analytical Solution') for nc in plot_equation(expr2, x, y, xrange)]
        [ax.plot(xrange, nc, c='r', alpha=.7, label='Analytical Solution') for nc in plot_equation(expr2, y, x, yrange)]
    except:
        print('Some nullcline values are complex')
    
    # plot the roots
    try:
        [ax.scatter(i[0], i[1], marker='x', label='Stable Point', s=150) for i in solutions]
    except:
        print('Some roots are complex')

    # a legend for our plot
    display(HTML('''<h2>Quiver Plot and Null Clines</h2><br />
                    A quiver plot shows the direction the system is moving in at each co-ordinate. <br />
                    Null Clines are lines where one of the two variables are zero. The Points 
                    where Null Clines intersect are where the system is stable because both variables are 
                    zero at that point.<br />
                    Null Clines can be obtained analytically (using calculus) and numerically.'''))
    fig.legend(framealpha=1, fancybox=True, fontsize='large', loc=1)
    fig.show()

# Plotting function for single neuron activity
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