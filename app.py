import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Function for the system of ODEs
def f(d, t, omega, k1, k2, m1, m2, P):
    return np.array([
        d[1],
        (-(1 + k1 / k2) * d[0] + k2 / k1 + (P / k1) * np.sin(omega * t)) * (k1 / m1),
        d[3],
        -(d[2] - d[0]) * (k2 / m2)
    ])

# Function for the system without the tuned mass
def f_n(d, t, omega, k1, m1, P):
    return np.array([
        d[1],
        -d[0] * k1 / m1 + (P / k1) * np.sin(omega * t)
    ])

# Runge-Kutta 4th order method
def rk(f, T, N, d0, omega, k1, k2=None, m1=None, m2=None, P=None, is_tuned=True):
    dt = T / N
    t = np.zeros(N + 1)
    d = np.zeros((N + 1, 4)) if is_tuned else np.zeros((N + 1, 2))
    d[0] = d0

    for n in range(N):
        w1 = f(d[n], t[n], omega, k1, k2, m1, m2, P) if is_tuned else f(d[n], t[n], omega, k1, m1, P)
        w2 = f(d[n] + (dt / 2) * w1, t[n] + dt / 2, omega, k1, k2, m1, m2, P) if is_tuned else f(d[n] + (dt / 2) * w1, t[n] + dt / 2, omega, k1, m1, P)
        w3 = f(d[n] + (dt / 2) * w2, t[n] + dt / 2, omega, k1, k2, m1, m2, P) if is_tuned else f(d[n] + (dt / 2) * w2, t[n] + dt / 2, omega, k1, m1, P)
        w4 = f(d[n] + dt * w3, t[n] + dt, omega, k1, k2, m1, m2, P) if is_tuned else f(d[n] + dt * w3, t[n] + dt, omega, k1, m1, P)
        t[n + 1] = t[n] + dt
        d[n + 1] = d[n] + (dt / 6) * (w1 + 2 * w2 + 2 * w3 + w4)

    return t, d

# Main function for plotting
def plot_graphs(k1, k2, m1, m2, P, omega):
    t, d_tuned = rk(f, 10, 10000, [0, 0, 0, 0], omega, k1, k2, m1, m2, P, is_tuned=True)
    t_n, d_non_tuned = rk(f_n, 60, 10000, [0, 0], omega, k1, m1=m1, P=P, is_tuned=False)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

    # First plot: d[0] vs t (Tuned)
    ax1.plot(t, d_tuned[:, 0], label="Tuned d[0] vs t")
    ax1.set_xlabel("Time (t)")
    ax1.set_ylabel("d[0]")
    ax1.legend()
    ax1.grid(True)

    # Second plot: d[2] vs t (Tuned)
    ax2.plot(t, d_tuned[:, 2], label="Tuned d[2] vs t", color='r')
    ax2.set_xlabel("Time (t)")
    ax2.set_ylabel("d[2]")
    ax2.legend()
    ax2.grid(True)

    # Third plot: d[0] vs t (Non-Tuned)
    ax3.plot(t_n, d_non_tuned[:, 0], label="Non-Tuned d[0] vs t", color='g')
    ax3.set_xlabel("Time (t)")
    ax3.set_ylabel("Non-Tuned d[0]")
    ax3.legend()
    ax3.grid(True)

    st.pyplot(fig)

# Streamlit layout
st.title("Dynamic System Simulation")

# Create columns for sliders and graphs
col1, col2 = st.columns([1, 3])  # Adjust the size ratio as needed

with col1:
    # Sliders for user input
    k1 = st.slider("k1", 0.0, 200.0, 125.0, 0.01)
    k2 = st.slider("k2", 0.0, 200.0, 1.25, 0.01)
    m1 = st.slider("m1", 0.0, 200.0, 5.0, 0.01)
    m2 = st.slider("m2", 0.0, 200.0, 0.05, 0.01)
    P = st.slider("P", 0.0, 200.0, 100.0, 0.01)
    omega = st.slider("omega", 0.0, 200.0, 5.0, 0.01)

with col2:
    # Plot graphs with the current slider values
    plot_graphs(k1, k2, m1, m2, P, omega)

