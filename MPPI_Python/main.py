import numpy as np
import matplotlib.pyplot as plt
from quadrotor_dynamics import quadrotor

"""
Functions needed:
- dynamics
- Cost
- MPPI (Forward pass, Backward pass, Generate sample, Calculate input)
- Graph
"""

"""
Definition
x: current state vector (12) (x, y, z, roll, pitch, yaw, xdot, ydot, zdot, rolldot, pitchdot, yawdot)
u: control input vector (4)
xd: goal state vector (6) [x, y, z, roll, pitch, yaw]

output: next state vector
"""

# Cost function definitions

def cost(x, u):
    x_flat = np.array(x)
    u = np.array(u).reshape(-1)

    Q = np.eye(len(x_flat))
    R = 0*np.eye(len(u))

    return x_flat.T @ Q @ x_flat + u.T @ R @ u

def final_cost(x, xd):
    x_flat = np.array(x)
    xd_flat = np.array(xd)

    Qf = 1000 * np.eye(len(x_flat))
    return (x_flat - xd_flat) @ Qf @ (x_flat - xd_flat).T

"""
Forward pass or roll-out: shoot the dynamics

xtraj: trajectory ((step + 1) x 12).
The first position is initial condition. Last is desired target
utraj: control input vector (step x 4)

output: new xtraj ((step + 1) x 12)

"""

# MPPI: Dynamics & Cost

def forward_pass(x0, xtraj, utraj, dt):
    xtraj[0] = x0
    for i in range(len(utraj)):
        xdot = quadrotor(xtraj[i], utraj[i])
        result = np.array(xtraj[i]) + dt * np.array(xdot)  
        xtraj[i+1] = result.tolist()

    return xtraj

"""
Backward pass: calculate the trajectory cost
xtraj: trajectory (step+1 x 12)
The first position is initial condition. Last is desired target
utraj: control inputs (step x 4)

output: cost of trajectory
"""

def backward_pass(xtraj, utraj, xd):
    traj_cost = 0
    for i in range(len(utraj)):
        traj_cost += cost(xtraj[i], utraj[i])
    traj_cost += final_cost(xtraj[-1], xd)
    return traj_cost

# Sampling and Control

"""
Generate sample: randomly generate control inputs
N: number of samples
time_step: discretized time step
nu: number of inputs

output: rollouts of utraj (step x nu x N)
"""

def gen_sample(N, scale, nu, step):
    dutraj =  scale * np.random.rand(step, nu, N)
    
    return dutraj

"""
Update u: update inputs based on cost 
input: rollouts of utraj (step x nu x N) and cost(1xN)

output: optimal utraj(step x nu x 1)
"""

def optimal_u(rutraj, cost):
    lamda = 100000
    weights = np.exp(-np.array(cost) / lamda)
    weights_sum = np.sum(weights)
    u = (rutraj @ weights) / weights_sum
    return u

# Main

# Boundary conditions
x0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

xd = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Control input params
steps = 50        # Number of time steps
dt = 0.1          # Time step in second
N = 1000         # Number of MPPI samples
nu = 4            # 4 ang vel^2 inputs: w1, w2, w3, w4
scale = 5         # Force scale

u_opt = []

for i in range(steps):
    print("solving time step #", i+1)
    # Reset trajectory
    xtraj = [x0 for _ in range(steps + 1)]

    # Generate force input samples
    rutraj = gen_sample(N, scale, nu, steps)

    # Use previously optimized values for already-computed steps
    if len(u_opt) > 0:
        for j in range(min(len(u_opt), rutraj.shape[0])):
            rutraj[j] = np.tile(u_opt[j].reshape(4, 1), (1, N))

    # Evaluate each sample trajectory
    traj_cost = []
    for ru in rutraj.transpose(2, 0, 1):  # (N x steps x 4)
        new_xtraj = forward_pass(x0, xtraj, ru, dt)
        traj_cost.append(backward_pass(new_xtraj, ru, xd))

    # Choose best u
    all_u = optimal_u(rutraj, traj_cost)
    u_opt.append(all_u[i])

# Final roll-out
print("Calculating optimal u...")
final_u = u_opt
print("Optimal u: ", final_u)

final_x = forward_pass(x0, xtraj, final_u, dt)
optimal_cost = backward_pass(final_x, final_u, xd)

print("Final cost:", optimal_cost)
