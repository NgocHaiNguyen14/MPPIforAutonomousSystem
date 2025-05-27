import numpy as np
import matplotlib.pyplot as plt
from quadrotor_dynamics import quadrotor
from quadrotor_grad import quadrotor_grad

"""
DEFINE EVERYTHING IN ARRAY
"""

# Cost function definitions

def cost(x, u, Q, R):
    cost = 0.5 * (x.T @ Q @ x + u.T @ R @ u)
    return cost

def final_cost(x, xd, Qf):
    final_cost = 0.5 * ((x - xd).T @ Qf @ (x - xd))
    return final_cost

def cost_grad(x, u, Q, R):
    gx = Q @ x
    gu = R @ u
    gxx = Q
    gux = np.zeros((gu.shape[0], gx.shape[0]))
    guu = R

    return gx, gu, gxx, gux, guu

def final_cost_grad(x, u, xd, Qf):
    gx = Qf @ (x - xd)
    gu = np.zeros((u.shape[0],1))
    gxx = Qf
    gux = np.zeros((u.shape[0], x.shape[0]))
    guu = np.zeros((u.shape[0], u.shape[0]))
    return gx, gu, gxx, gux, guu

def Qterms(gx, gu, gxx, gux, guu, fx, fu, Vx, Vxx):
    Qx = gx + fx.T @ Vx
    Qu = gu + fu.T @ Vx
    Qxx = gxx + fx.T @ Vxx @ fx
    Qux = 2*gux + fu.T @ Vxx @ fx
    Quu = guu + fu.T @ Vxx @ fu

    return Qx, Qu, Qxx, Qux, Quu

def gains(Qx, Qu, Qxx, Qux, Quu):
    v = -np.linalg.inv(Quu) @ Qu
    K = -np.linalg.inv(Quu) @ Qux

    return v, K

def Vterms(Qx, Qu, Qxx, Qux, Quu, K, v):
    Vx = (Qx.T + Qu.T @ K + v.T @ Qux + v.T @ Quu @ K).T
    Vxx = Qxx + K.T @ Qux + Qux.T @ K + K.T @ Quu @ K

    return Vx, Vxx

"""
Forward pass: shoot the dynamics

xtraj0: old trajectory (12 x steps).
utraj0: old control law (4 x steps-1)
vtraj: offset 
Ktraj: gain
steps
dt
Q
R
Qf
xd
J0

output: new xtraj (12 x steps)
        new utraj (4 x steps-1)

"""
def forward_pass(x0, xtraj0, utraj0, vtraj, Ktraj, steps, dt, Q, R, Qf, xd, J0):
    J = 1e7
    alpha = 10
    xtraj = xtraj0.copy()
    utraj = utraj0.copy()
    while J0 < J:
        x = x0
        J = 0
        for i in range(steps-1):
            xtraj[:, :, i] = x
            utraj[:, :, i] = utraj0[:, :, i] + alpha*vtraj[:, :, i] + Ktraj[:, :, i] @ (x - xtraj0[:, :, i])
            J += cost(x, utraj[:, :, i], Q, R)
            xdot = quadrotor(x, utraj[:, :, i])
            x = x + xdot*dt
        
        xtraj[:, :, steps-1] = x
        J += final_cost(x, xd, Qf)
        alpha /= 2
    return xtraj, utraj, J


"""
Backward pass:

xtraj: trajectory (12 x steps).
utraj: control input vector (4 x steps-1)

output: vtraj (4 x steps)
        Ktraj (4 x 12 x steps)

"""
def backward_pass(xtraj, utraj, vtraj, Ktraj, Q, R, Qf, xd, steps, dt):
    # Final time step gradient (at x_N)
    gxN, guN, gxxN, guxN, guuN = final_cost_grad(xtraj[:, :,steps - 1], utraj[:, :,steps - 2], xd, Qf)
    
    Vx = gxN
    Vxx = gxxN
    
    for i in range(steps - 2, -1, -1):  # include i = 0
        # Get cost gradients at current timestep
        gx, gu, gxx, gux, guu = cost_grad(xtraj[:, :, i], utraj[:, :, i], Q, R)
        
        # Dynamics linearization
        fx, fu = quadrotor_grad(xtraj[:, :, i], utraj[:, :, i])
        fx = fx * dt + np.eye(fx.shape[0])
        fu = fu * dt

        # Compute Q-function derivatives
        Qx, Qu, Qxx, Qux, Quu = Qterms(gx, gu, gxx, gux, guu, fx, fu, Vx, Vxx)

        # Compute gains
        vtraj[:, :, i], Ktraj[:, :, i] = gains(Qx, Qu, Qxx, Qux, Quu)

        # Update value function
        Vx, Vxx = Vterms(Qx, Qu, Qxx, Qux, Quu, Ktraj[:, :, i], vtraj[:, :, i])

    return Ktraj, vtraj

"""
iLQR

xtraj: trajectory (12 x 1x steps)
utraj: control input vector (4 x 1 x steps-1)
vtraj: offset (4 x 1 x steps)
Ktraj: gain   (4 x 12 x steps)

output: base utraj (4 x 1 x steps-1)

"""

def iLQR(x0, xd, steps, dt, Q, R, Qf):
    J = 1e6
    Jlast = J
    nx = len(x0)
    nu = 4
    vtraj = np.zeros((nu, 1, steps))
    Ktraj = np.zeros((nu, nx, steps))
    xtraj = np.zeros((nx, 1, steps))
    utraj = np.zeros((nu, 1, steps-1))
    for i in range(1000):
        print("Iteration: ", i)
        xtraj, utraj, J = forward_pass(x0, xtraj, utraj, vtraj, Ktraj, steps, dt, Q, R, Qf, xd, J)
        Ktraj, vtraj = backward_pass(xtraj, utraj, vtraj, Ktraj, Q, R, Qf, xd, steps, dt)
        if abs(J - Jlast) < 1e-6:
            break
        Jlast = J

    return xtraj, utraj, Jlast

def main():
    # Simulation parameters
    steps = 80
    dt = 0.01  # 20 Hz

    # System dimensions
    nx = 12  # state dimension
    nu = 4   # input dimension

    # Initial and desired state
    x0 = np.zeros((nx,1))
    x0[0] = 1.0  # Start at x = 1m
    x0[2] = 0.0  # Start at z = 2m
    xd = np.zeros((nx,1))  # Hover at origin

    # Cost matrices
    Q = np.diag([10, 10, 100, 1, 1, 10, 1, 1, 1, 1, 1, 1]) * 0
    R = np.eye(nu) * 0
    Qf = np.diag([100, 100, 100, 10, 10, 100, 10, 10, 10, 10, 10, 10])

    # Run iLQR
    xtraj, utraj, Jlast = iLQR(x0, xd, steps, dt, Q, R, Qf)
    print(xtraj.shape)
    print(Jlast)
    # Plot position trajectories
    time = np.linspace(0, dt * (steps - 1), steps)
    labels = ['x', 'y', 'z']
    plt.figure(figsize=(10, 6))
    for i in range(3):
        plt.plot(time, xtraj[i, 0, :], label=labels[i])
    plt.title('Final Position Trajectory from iLQR')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
