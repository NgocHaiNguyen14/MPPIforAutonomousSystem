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
# Assume all inputs are array
def runningCost(x, xd, u, du, Q, R, nuy):
    qx = (x-xd).T @ Q @ (x-xd)

    return qx + 1/2*u.T @ R @ u + (1-1/nuy)/2*du.T @ R @ du + u.T @ R @ du

def finalCost(x, xd, Qf):
    return (x - xd).T @ Qf @ (x - xd)

"""
roll_out: shoot the dynamics

xtraj: trajectory ((step + 1) x 12).
The first position is initial condition. Last is desired target
utraj: control input vector (step x 4)

output: new xtraj ((step + 1) x 12)

"""

# MPPI: Dynamics & Cost

def roll_out(x0, xtraj, utraj, dt):
    xtraj[:,0] = x0
    for i in range(len(utraj)):
        xdot = quadrotor(xtraj[i], utraj[i])
        result = np.array(xtraj[i]) + dt * np.array(xdot)  
        xtraj[i+1] = result.tolist()

    return xtraj


"""
total_cost: calculate the trajectory cost
xtraj: trajectory (step+1 x 12)
The first position is initial condition. Last is desired target
utraj: control inputs (step x 4)

output: cost of trajectory
"""

def total_cost(xtraj, utraj, dutraj, xd, Q, R, Qf, nuy):
    traj_cost = 0
    for i in range(len(utraj)):
        traj_cost += runningCost(xtraj[i], xd, utraj[i], dutraj[i], Q, R, nuy)
    traj_cost += finalCost(xtraj[-1], xd, Qf)
    return traj_cost


# Sampling and Control

"""
Update u: update inputs based on cost 

HYPER-PARAMETER: LAMDA

input: utraj (nU x T), rollouts of dutraj (nU x T x N), and cost(N)

output: optimal utraj(nU x T)
"""

def optimal_u(utraj, dU, Straj, lamda):
    
    for t in range(utraj.shape[1]):
        ss = 0
        su = 0
        minS = min(Straj)

        for k in range(len(Straj)):
            weight = np.exp(-1.0 / lamda * (Straj[k] - minS))
            ss += weight
            su += weight * dU[k][:, t]

        utraj[:,t] += su/ss
    
    return utraj

## Initialize new utraj after update

def updateutraj(utraj):
    nU, time_steps = utraj.shape
    for i in range(1, time_steps):
        utraj[:, i-1] = utraj[:, i]
    utraj[:, -1] = np.zeros(nU)

    return utraj

# Main
nX = 12 # Number of states
nU = 4 # Number of inputs

# Boundary Conditions
x0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
xd = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# MPPI Params
N = 1000         # Number of MPPI samples
T = 150          # Max number of time steps
dt = 0.02 # delta time

covU = np.diag([2.5,5*1e-3,5*1e-3,5*1e-3])
lamda = 10
nuy = 1000

# Cost matrices
Q = np.eye(nX)
R = lamda*np.linalg.inv(covU)

Qf = np.diag([2.5, 2.5, 10,    # e.g. position x, y, z
            1, 1, 10,        # e.g. orientation roll, pitch, yaw
            0, 0, 0,         # e.g. velocity x, y, z
            0, 0, 0])        # e.g. bias or other states

utraj = np.zeros((nU, T-1))
u_opt = []
xf = []

x = x0

for iter in range(500):
    xf = xf.append(x)
    Straj = np.zeros(N)
    dU = [None] * N

    for k in range(N):
        dutraj = covU @ np.random.randn(nU, T-1)
        dU[k] = dutraj

        xtraj = np.zeros((nX, T))
        xtraj[:,0] = x

        xtraj = roll_out(x, xtraj, utraj+dutraj, dt)
        Straj[k] = total_cost(xtraj, utraj, dutraj, xd, Q, R, Qf, nuy)
    
    utraj = optimal_u(utraj, dU, Straj, lamda)

    # Excecute utraj(0)
    x += quadrotor(x, utraj[:,0])*dt
    u_opt = u_opt.append(utraj[:,0])

    utraj = updateutraj(utraj)
