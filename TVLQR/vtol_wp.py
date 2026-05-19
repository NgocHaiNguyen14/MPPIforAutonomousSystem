"""
Python / PyTorch port of vtol_wp.m.

CEM trajectory optimisation through a sequence of waypoints for a 13-state
quaternion VTOL.  All `num_samples` candidate control sequences in a CEM
iteration are rolled out in parallel as a single batched tensor op, so the
hot loop is one Euler-integration scan of length N over a (num_samples,)
batch dimension -- not a Python-level loop over samples.

Run:
    python vtol_wp.py
"""

import os
import time

import numpy as np
import torch
import matplotlib.pyplot as plt

from vtol_quaternion import vtol_dynamics, quat_multiply


# ============================================================================
# Device / dtype
# ============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float32
torch.set_default_dtype(DTYPE)
print(f"Using device: {DEVICE}")


# ============================================================================
# Helpers
# ============================================================================
def build_diag_cost(w_pos, w_vel, w_quat, w_omega):
    """13x13 block-diagonal cost matrix from weight vectors.

    Error vector ordering: [pos(3); vel(3); q_err(4); omega(3)].
    """
    diag = list(w_pos) + list(w_vel) + list(w_quat) + list(w_omega)
    return torch.diag(torch.tensor(diag, device=DEVICE, dtype=DTYPE))


def state_error(x, xd):
    e_pos   = x[:, 0:3]   - xd[0:3]
    e_vel   = x[:, 3:6]   - xd[3:6]
    e_omega = x[:, 10:13] - xd[10:13]

    q = x[:, 6:10]
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp(min=1e-12)

    qd = xd[6:10]
    qd = qd / qd.norm().clamp(min=1e-12)

    qd_conj = torch.tensor(
        [qd[0], -qd[1], -qd[2], -qd[3]],
        device=x.device,
        dtype=x.dtype
    )

    q_err = quat_multiply(
        qd_conj.unsqueeze(0).expand_as(q),
        q
    )

    # shortest rotation
    q_err = torch.where(
        q_err[:, 0:1] < 0,
        -q_err,
        q_err
    )

    # minimal 3D attitude error
    e_rot = 2.0 * q_err[:, 1:4]

    return torch.cat([e_pos, e_vel, e_rot, e_omega], dim=-1)


def rollout(x0, U_batch, dt, N):
    """Batched forward Euler rollout with quaternion renormalisation.

    x0      : [13]          shared initial state
    U_batch : [B, 4, N]     candidate control sequences

    Returns
        X : [B, 13, N+1]
    """
    B = U_batch.shape[0]
    X = torch.empty(B, 13, N + 1, device=DEVICE, dtype=DTYPE)
    X[:, :, 0] = x0.unsqueeze(0).expand(B, -1)

    for k in range(N):
        sdot = vtol_dynamics(X[:, :, k], U_batch[:, :, k])
        xk1 = X[:, :, k] + sdot * dt
        # Renormalise the quaternion slot.
        q = xk1[:, 6:10]
        q = q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp(min=1e-12)
        xk1 = torch.cat([xk1[:, 0:6], q, xk1[:, 10:13]], dim=-1)
        X[:, :, k + 1] = xk1

    return X


def trajectory_cost(X, U_batch, xd, Q, Qf, R_ctrl, dt, N):
    """Batched trajectory cost.  X: [B,13,N+1], U_batch: [B,4,N]. Returns [B]."""
    B = X.shape[0]
    J = torch.zeros(B, device=DEVICE, dtype=DTYPE)

    # Running state cost: sum_{k=0..N-1} e_k^T Q e_k * dt
    for k in range(N):
        e = state_error(X[:, :, k], xd)            # [B, 13]
        J = J + (e @ Q * e).sum(dim=-1) * dt       # batched e^T Q e
        u = U_batch[:, :, k]
        J = J + (u @ R_ctrl * u).sum(dim=-1) * dt

    # Terminal cost
    eT = state_error(X[:, :, N], xd)
    J = J + (eT @ Qf * eT).sum(dim=-1)
    return J


# ============================================================================
# Problem definition (matches vtol_wp.m)
# ============================================================================
nX = 13
nU = 4

waypoints = [
    torch.tensor([0,0,0,    0,0,0, 0.7071,0,0.7071,0, 0,0,0], device=DEVICE, dtype=DTYPE),  # x0
    torch.tensor([0,0,5,    0,0,0, 0.7071,0,0.7071,0, 0,0,0], device=DEVICE, dtype=DTYPE),  # x1
    torch.tensor([0,0,5,    0,0,0, 1,0,0,0,           0,0,0], device=DEVICE, dtype=DTYPE),  # x2
    torch.tensor([3,3,6,    0,0,0, 1,0,0,0,           0,0,0], device=DEVICE, dtype=DTYPE),  # x3
    torch.tensor([6,6,8,    0,0,0, 1,0,0,0,           0,0,0], device=DEVICE, dtype=DTYPE),  # x4
    torch.tensor([9,9,9,    0,0,0, 1,0,0,0,           0,0,0], device=DEVICE, dtype=DTYPE),  # x5
    torch.tensor([9,9,9,    0,0,0, 0.7071,0,0.7071,0, 0,0,0], device=DEVICE, dtype=DTYPE),  # x6
    torch.tensor([9,9,0,    0,0,0, 0.7071,0,0.7071,0, 0,0,0], device=DEVICE, dtype=DTYPE),  # xd
]
num_waypoints = len(waypoints)

# (N, dt) per segment.
seg_params = [
    (50, 0.01),    # x0 -> x1
    (50,  0.01),   # x1 -> x2
    (150, 0.01),   # x2 -> x3
    (150, 0.01),   # x3 -> x4 
    (150, 0.01),   # x4 -> x5
    (50,  0.01),   # x5 -> x6
    (75, 0.01),   # x6 -> xd
]

# CEM parameters
num_samples = 250000
num_elites  = 100
iterations  = 1000

alpha_mean  = 0.5
alpha_sigma = 0.2

sigma2_init = 20.0
sigma2_min  = 1e-1
sigma2_max  = 20.0

# ---- per-segment cost matrices (same numbers as the MATLAB script) ----
# seg_costs = [
#     # 1: x0 -> x1   takeoff
#     dict(Q=build_diag_cost([25, 25, 25], [1, 1, 1], [1, 1, 1],          [1, 1, 1]),
#          Qf=build_diag_cost([100, 100, 100], [1, 1, 1], [1000, 1000, 1000], [1, 1, 1]),
#          R=0.0 * torch.eye(nU, device=DEVICE, dtype=DTYPE)),
#     # 2: x1 -> x2   rotate to upright
#     dict(Q=build_diag_cost([1.0, 1.0, 1.0], [0.1, 0.1, 0.1], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5]),
#          Qf=build_diag_cost([5, 5, 5],      [1, 1, 1],        [100, 100, 100], [10, 10, 10]),
#          R=0.0 * torch.eye(nU, device=DEVICE, dtype=DTYPE)),
#     # 3: x2 -> x3   diagonal cruise
#     dict(Q=build_diag_cost([25, 25, 25],   [1, 1, 1],    [1, 1, 1],      [1, 1, 1]),
#          Qf=build_diag_cost([1000, 1000, 1000], [10, 10, 10], [50, 50, 50], [10, 10, 10]),
#          R=0.0 * torch.eye(nU, device=DEVICE, dtype=DTYPE)),
#     # 4: x3 -> x4
#     dict(Q=build_diag_cost([25, 25, 25],   [1, 1, 1],    [1, 1, 1],      [1, 1, 1]),
#          Qf=build_diag_cost([1000, 1000, 1000], [10, 10, 10], [50, 50, 50], [10, 10, 10]),
#          R=0.0 * torch.eye(nU, device=DEVICE, dtype=DTYPE)),
#     # 5: x4 -> x5
#     dict(Q=build_diag_cost([25, 25, 25],   [1, 1, 1],    [1, 1, 1],      [1, 1, 1]),
#          Qf=build_diag_cost([1000, 1000, 1000], [10, 10, 10], [50, 50, 50], [10, 10, 10]),
#          R=0.0 * torch.eye(nU, device=DEVICE, dtype=DTYPE)),
#     # 6: x5 -> x6   rotate to landing attitude
#     dict(Q=build_diag_cost([1.0, 1.0, 1.0], [0.1, 0.1, 0.1], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5]),
#          Qf=build_diag_cost([1, 1, 1],      [1, 1, 1],        [100, 100, 100], [10, 10, 10]),
#          R=0.0 * torch.eye(nU, device=DEVICE, dtype=DTYPE)),
#     # 7: x6 -> xd   landing
#     dict(Q=build_diag_cost([10, 10, 10],     [1, 1, 1],    [1, 1, 1],            [1, 1, 1]),
#          Qf=build_diag_cost([100, 100, 100], [1, 1, 1], [1000, 1000, 1000], [1, 1, 1]),
#          R=0.0 * torch.eye(nU, device=DEVICE, dtype=DTYPE)),
# ]

seg_costs = [
    # 1: x0 -> x1   takeoff
    dict(Q=build_diag_cost([0, 0, 0], [0, 0, 0], [0, 0, 0],          [0, 0, 0]),
         Qf=build_diag_cost([100, 100, 100], [1, 1, 1], [1000, 1000, 1000], [1, 1, 1]),
         R=0.0 * torch.eye(nU, device=DEVICE, dtype=DTYPE)),
    # 2: x1 -> x2   rotate to upright
    dict(Q=build_diag_cost([0, 0, 0], [0, 0, 0], [0, 0, 0],          [0, 0, 0]),
         Qf=build_diag_cost([10, 10, 10],      [1, 1, 1],        [100, 100, 100], [10, 10, 10]),
         R=0.0 * torch.eye(nU, device=DEVICE, dtype=DTYPE)),
    # 3: x2 -> x3   diagonal cruise
    dict(Q=build_diag_cost([0, 0, 0], [0, 0, 0], [0, 0, 0],          [0, 0, 0]),
         Qf=build_diag_cost([100, 100, 100], [0, 0, 0], [50, 50, 50], [0, 0, 0]),
         R=0.0 * torch.eye(nU, device=DEVICE, dtype=DTYPE)),
    # 4: x3 -> x4
    dict(Q=build_diag_cost([0, 0, 0], [0, 0, 0], [0, 0, 0, 0],          [0, 0, 0]),
         Qf=build_diag_cost([100, 100, 100], [0, 0, 0], [50, 50, 50, 50], [0, 0, 0]),
         R=0.0 * torch.eye(nU, device=DEVICE, dtype=DTYPE)),
    # 5: x4 -> x5
    dict(Q=build_diag_cost([0, 0, 0], [0, 0, 0], [0, 0, 0, 0],          [0, 0, 0]),
         Qf=build_diag_cost([100, 100, 100], [10, 10, 10], [50, 50, 50, 50], [10, 10, 10]),
         R=0.0 * torch.eye(nU, device=DEVICE, dtype=DTYPE)),
    # 6: x5 -> x6   rotate to landing attitude
    dict(Q=build_diag_cost([0, 0, 0], [0, 0, 0], [0, 0, 0, 0],          [0, 0, 0]),
         Qf=build_diag_cost([10, 10, 10],      [10, 10, 10],        [100, 100, 100, 100], [10, 10, 10]),
         R=0.0 * torch.eye(nU, device=DEVICE, dtype=DTYPE)),
    # 7: x6 -> xd   landing
    dict(Q=build_diag_cost([0, 0, 0], [0, 0, 0], [0, 0, 0, 0],          [0, 0, 0]),
         Qf=build_diag_cost([1000, 1000, 1000], [10, 10, 10], [1000, 1000, 1000, 1000], [10, 10, 10]),
         R=0.1 * torch.eye(nU, device=DEVICE, dtype=DTYPE)),
]

# ============================================================================
# CEM optimisation loop
# ============================================================================
@torch.no_grad()
def run_cem():
    x_full = waypoints[0].unsqueeze(-1).clone()   # [13, 1] -> grows
    u_full_list = []                              # accumulate [4, N] per seg
    seg_breaks = [0]                              # column indices in x_full
    J_histories = []
    sigma_histories = []

    # Mean control for warm-starting the next segment.
    U = None

    for seg in range(num_waypoints - 1):
        N, dt = seg_params[seg]
        x0_seg = waypoints[0] if seg == 0 else x_full[:, -1].clone()
        xd_seg = waypoints[seg + 1]

        Q      = seg_costs[seg]["Q"]
        Qf     = seg_costs[seg]["Qf"]
        R_ctrl = seg_costs[seg]["R"]

        print(f"\n=== Segment {seg+1} -> {seg+2}  (N={N}, dt={dt:.4f}) ===")

        # Initialise / reshape mean for this segment.
        if U is None:
            U = torch.zeros(nU, N, device=DEVICE, dtype=DTYPE)
        else:
            cur_N = U.shape[1]
            if N > cur_N:
                U = torch.cat([U, torch.zeros(nU, N - cur_N, device=DEVICE, dtype=DTYPE)], dim=1)
            elif N < cur_N:
                U = U[:, :N]

        sigma2 = sigma2_init * torch.ones(N, device=DEVICE, dtype=DTYPE)

        J_hist = torch.empty(iterations, device=DEVICE, dtype=DTYPE)
        sig_hist = torch.empty(iterations, device=DEVICE, dtype=DTYPE)

        t_seg_start = time.time()

        for i in range(iterations):
            # --- Sample du ~ N(0, sigma2)  -> shape [num_samples, nU, N] ----
            std = sigma2.sqrt().view(1, 1, N)          # [1,1,N]
            du = std * torch.randn(num_samples, nU, N, device=DEVICE, dtype=DTYPE)
            U_batch = U.unsqueeze(0) + du              # [num_samples, nU, N]

            # --- Parallel rollout & cost -----------------------------------
            X_batch = rollout(x0_seg, U_batch, dt, N)
            J = trajectory_cost(X_batch, U_batch, xd_seg, Q, Qf, R_ctrl, dt, N)

            # --- Elites ----------------------------------------------------
            _, elite_idx = torch.topk(J, num_elites, largest=False)
            elites = U_batch[elite_idx]                # [num_elites, nU, N]

            U_new = elites.mean(dim=0)                 # [nU, N]
            diffs = elites - U_new.unsqueeze(0)        # [num_elites, nU, N]
            sigma2_new = diffs.pow(2).mean(dim=(0, 1)) # [N] (averaged over elites and ctrl dims)

            # --- Momentum smoothing on mean and variance -------------------
            U_new      = alpha_mean  * U_new      + (1.0 - alpha_mean)  * U
            sigma2_new = alpha_sigma * sigma2_new + (1.0 - alpha_sigma) * sigma2

            sigma2_new = sigma2_new.clamp(min=sigma2_min, max=sigma2_max)

            # Cost of the new mean trajectory (for the history plot).
            X_mean = rollout(x0_seg, U_new.unsqueeze(0), dt, N)
            J_mean = trajectory_cost(X_mean, U_new.unsqueeze(0), xd_seg,
                                     Q, Qf, R_ctrl, dt, N).item()

            J_hist[i] = J_mean
            sig_hist[i] = sigma2_new.min()

            if (i + 1) % 25 == 0 or i == 0:
                print(f"  Iter {i+1:3d} | J_mean = {J_mean:10.4f} | "
                      f"sigma2 min/mean/max = {sigma2_new.min().item():.2e} / "
                      f"{sigma2_new.mean().item():.2e} / "
                      f"{sigma2_new.max().item():.2e}")

            U = U_new
            sigma2 = sigma2_new

        elapsed = time.time() - t_seg_start
        print(f"  Segment {seg+1} finished in {elapsed:.2f}s "
              f"({elapsed/iterations*1000:.1f} ms/iter)")

        # --- Final segment rollout with the converged mean ----------------
        X_seg = rollout(x0_seg, U.unsqueeze(0), dt, N).squeeze(0)  # [13, N+1]
        print(X_seg[:, 0])
        print(X_seg[:, -1])

        # Accumulate (skip the first state to avoid duplicates).
        x_full = torch.cat([x_full, X_seg[:, 1:]], dim=1)
        u_full_list.append(U.clone())
        seg_breaks.append(x_full.shape[1] - 1)

        J_histories.append(J_hist.cpu().numpy())
        sigma_histories.append(sig_hist.cpu().numpy())

        # Warm-start the next segment's mean.
        U = torch.cat([U[:, 1:], torch.zeros(nU, 1, device=DEVICE, dtype=DTYPE)], dim=1)

    u_full = torch.cat(u_full_list, dim=1)   # [4, sum_N]

    return (x_full.cpu().numpy(),
            u_full.cpu().numpy(),
            np.array(seg_breaks),
            J_histories,
            sigma_histories)


# ============================================================================
# Plotting
# ============================================================================
def plot_convergence(J_histories, sigma_histories):
    for seg, (J_hist, sig_hist) in enumerate(zip(J_histories, sigma_histories)):
        fig, axs = plt.subplots(2, 1, figsize=(8, 6))
        axs[0].semilogy(np.arange(1, len(J_hist) + 1), J_hist, "b-", linewidth=1.5)
        axs[0].set_xlabel("Iteration")
        axs[0].set_ylabel("Cost J")
        axs[0].set_title(f"Segment {seg+1}->{seg+2}: Cost convergence")
        axs[0].grid(True, which="both")

        axs[1].semilogy(np.arange(1, len(sig_hist) + 1), sig_hist, "r-", linewidth=1.5)
        axs[1].set_xlabel("Iteration")
        axs[1].set_ylabel("min(sigma^2)")
        axs[1].set_title(f"Segment {seg+1}->{seg+2}: Min sigma^2 convergence")
        axs[1].grid(True, which="both")
        fig.tight_layout()
        fig.savefig(f"convergence_seg{seg+1}.png", dpi=120)
        plt.close(fig)


def plot_trajectory(x_full, u_full, seg_breaks, waypoints_list):
    num_wp = len(waypoints_list)
    wp_pos = np.stack([wp.cpu().numpy()[:3] for wp in waypoints_list], axis=1)  # [3, num_wp]
    colors = plt.cm.tab10(np.linspace(0, 1, num_wp - 1))

    fig = plt.figure(figsize=(13, 9))

    # --- 3D ---
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    for seg in range(num_wp - 1):
        a, b = seg_breaks[seg], seg_breaks[seg + 1]
        ax3d.plot(x_full[0, a:b+1], x_full[1, a:b+1], x_full[2, a:b+1],
                  "-", color=colors[seg], linewidth=2,
                  label=f"Seg {seg+1}->{seg+2}")
    ax3d.scatter(wp_pos[0], wp_pos[1], wp_pos[2], c="k", s=40, label="Waypoints")
    for k in range(num_wp):
        ax3d.text(wp_pos[0, k], wp_pos[1, k], wp_pos[2, k] + 0.3, f"  x{k}",
                  fontsize=8, fontweight="bold")
    ax3d.set_xlabel("X (m)"); ax3d.set_ylabel("Y (m)"); ax3d.set_zlabel("Z (m)")
    ax3d.set_title("3D Trajectory")
    ax3d.legend(loc="best", fontsize=7)
    ax3d.view_init(elev=30, azim=45)

    # --- XY ---
    axxy = fig.add_subplot(2, 2, 2)
    for seg in range(num_wp - 1):
        a, b = seg_breaks[seg], seg_breaks[seg + 1]
        axxy.plot(x_full[0, a:b+1], x_full[1, a:b+1], "-", color=colors[seg], linewidth=2)
    axxy.scatter(wp_pos[0], wp_pos[1], c="k", s=40)
    for k in range(num_wp):
        axxy.text(wp_pos[0, k] + 0.2, wp_pos[1, k] + 0.2, f"x{k}", fontsize=8)
    axxy.set_xlabel("X (m)"); axxy.set_ylabel("Y (m)")
    axxy.set_title("Top View (XY)"); axxy.grid(True)

    # --- XZ ---
    axxz = fig.add_subplot(2, 2, 4)
    for seg in range(num_wp - 1):
        a, b = seg_breaks[seg], seg_breaks[seg + 1]
        axxz.plot(x_full[0, a:b+1], x_full[2, a:b+1], "-", color=colors[seg], linewidth=2)
    axxz.scatter(wp_pos[0], wp_pos[2], c="k", s=40)
    for k in range(num_wp):
        axxz.text(wp_pos[0, k] + 0.2, wp_pos[2, k] + 0.2, f"x{k}", fontsize=8)
    axxz.set_xlabel("X (m)"); axxz.set_ylabel("Z (m)")
    axxz.set_title("Side View (XZ)"); axxz.grid(True)

    fig.suptitle("VTOL Waypoint Trajectory (CEM)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig("trajectory.png", dpi=130)
    plt.close(fig)

    # --- Controls ---
    fig2, axs = plt.subplots(nU, 1, figsize=(10, 8), sharex=True)
    labels = ["u_1 (Tr)", "u_2 (Tl)", "u_3 (dr)", "u_4 (dl)"]
    T_u = u_full.shape[1]
    for k in range(nU):
        axs[k].plot(np.arange(T_u), u_full[k], linewidth=1.0)
        axs[k].set_ylabel(labels[k])
        axs[k].grid(True)
        for sb in seg_breaks[1:-1]:
            axs[k].axvline(sb, color="k", linestyle="--", alpha=0.4)
    axs[0].set_title("Control Inputs (dashed lines = segment boundaries)")
    axs[-1].set_xlabel("Time step")
    fig2.tight_layout()
    fig2.savefig("controls.png", dpi=130)
    plt.close(fig2)


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    t0 = time.time()
    x_full, u_full, seg_breaks, J_histories, sigma_histories = run_cem()
    print(f"\nTotal optimisation time: {time.time() - t0:.2f}s")

    print("\nFinal state:")
    print(x_full[:, -1])
    print("Optimisation complete.")

    np.savez("vtol_trajectory.npz",
             x_full=x_full,
             u_full=u_full,
             seg_breaks=seg_breaks,
             waypoints=np.stack([wp.cpu().numpy() for wp in waypoints], axis=1),
             seg_params=np.array(seg_params))
    print("Trajectory saved to vtol_trajectory.npz")

    plot_convergence(J_histories, sigma_histories)
    plot_trajectory(x_full, u_full, seg_breaks, waypoints)
    print("Plots saved: trajectory.png, controls.png, convergence_seg*.png")
