import jax.numpy as jnp

def continuous_dynamics(state, control, wheel_base, dt):
    """
    Define the bicycle dynamics equations here. (derived from https://thomasfermi.github.io/Algorithms-for-Automated-Driving/Control/BicycleModel.html)

    State: [x, y, theta, v]
    Control: [a, delta]
    Wheel base: wheel_base
    Time step: dt
    """
    x, y, theta, v = state
    a, delta = control
    L = wheel_base

    # Calculate the longitudinal and lateral velocities
    x_dot = v * jnp.cos(theta)
    y_dot = v * jnp.sin(theta)
    omega = v * jnp.tan(delta) / L
    v_dot = a # TODO: Add noise

    return jnp.array([x_dot, y_dot, omega, v_dot])


def rk4_step(state, control, wheel_base, dt):
    """
    Implement Runge-Kutta integration for discrete-time dynamics.

    Args:
        state: The current state of the system.
        control: The current control input.
        wheel_base: The wheel base of the vehicle.
        dt: The time step for integration.
    
    Returns:
        The next state of the system.
    """
    half_dt = 0.5 * dt
    f1 = continuous_dynamics(state, control, wheel_base, dt)
    f2 = continuous_dynamics(state + half_dt * f1, control, wheel_base, half_dt)
    f3 = continuous_dynamics(state + half_dt * f2, control, wheel_base, half_dt)
    f4 = continuous_dynamics(state + dt * f3, control, wheel_base, half_dt)
    state_next = state + (dt/6) * (f1 + (2 * f2) + (2 * f3) + f4)
    
    return state_next