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

def cost(x, u, Q, R):
    x_flat = np.array(x)
    u = np.array(u).reshape(-1)

    return x_flat.T @ Q @ x_flat + u.T @ R @ u

def final_cost(x, xd, Qf):
    x_flat = np.array(x)
    xd_flat = np.array(xd)

    return (x_flat - xd_flat) @ Qf @ (x_flat - xd_flat).T

"""
roll_out: shoot the dynamics

xtraj: trajectory ((step + 1) x 12).
The first position is initial condition. Last is desired target
utraj: control input vector (step x 4)

output: new xtraj ((step + 1) x 12)

"""

# MPPI: Dynamics & Cost

def roll_out(x0, xtraj, utraj, dt):
    xtraj[0] = x0
    for i in range(len(utraj)):
        xdot = quadrotor(xtraj[i], utraj[i])
        result = np.array(xtraj[i]) + dt * np.array(xdot)  
        xtraj[i+1] = result.tolist()

    return xtraj


"""
Backward pass:

xtraj: trajectory ((step + 1) x 12).
The first position is initial condition. Last is desired target
utraj: control input vector (step x 4)

output: new xtraj ((step + 1) x 12)

"""


"""
total_cost: calculate the trajectory cost
xtraj: trajectory (step+1 x 12)
The first position is initial condition. Last is desired target
utraj: control inputs (step x 4)

output: cost of trajectory
"""

def total_cost(xtraj, utraj, xd, Q, R, Qf):
    traj_cost = 0
    for i in range(len(utraj)):
        traj_cost += cost(xtraj[i], utraj[i], Q, R)
    traj_cost += final_cost(xtraj[-1], xd, Qf)
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

HYPER-PARAMETER: LAMDA

input: rollouts of utraj (step x nu x N) and cost(1xN)

output: optimal utraj(step x nu x 1)
"""

def optimal_u(rutraj, cost):
    lamda = 10000
    weights = np.exp(-np.array(cost) / lamda)
    weights_sum = np.sum(weights)
    u = (rutraj @ weights) / weights_sum
    return u

# Main

# iLQR params
Q = np.eye(len(x_flat))
R = 0*np.eye(len(u))

Qf = np.diag([10000, 10000, 10000,    # e.g. position x, y, z
   10, 10, 10,          # e.g. velocity x, y, z
   100, 100, 100,       # e.g. orientation roll, pitch, yaw
   1, 1, 1])             # e.g. bias or other states

# MPPI Params
N = 1000         # Number of MPPI samples
nu = 4            # 4 ang vel^2 inputs: w1, w2, w3, w4
scale = 10         # Force scale

# Trajectories params
x0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

xd = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

t = 0.1
steps = 10        # Number of time steps
dt = t/steps         # Time step in second

u_base = iLQR(x0, xd, dt, Q, R, Qf)

u_opt = []



