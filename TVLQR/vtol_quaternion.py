"""
Batched PyTorch port of vtol_quaternion.m.

All functions operate on a leading batch dimension B so that an entire
population of CEM samples can be rolled out in one tensor op.

State vector s (B, 13):
    s[:,  0: 3]  position in world frame (NED)
    s[:,  3: 6]  body-frame linear velocity [u; v; w]
    s[:,  6:10]  attitude quaternion [w; x; y; z]   (body wrt. world)
    s[:, 10:13]  body-frame angular velocity [p; q; r]

Input vector u (B, 4):
    u[:, 0]  Tr  right propeller thrust
    u[:, 1]  Tl  left  propeller thrust
    u[:, 2]  dr  right elevon deflection (deg)
    u[:, 3]  dl  left  elevon deflection (deg)
"""

import torch

# --- Environment -----------------------------------------------------------
G_ACCEL = 9.81
RHO     = 1.2551

# --- VTOL specs ------------------------------------------------------------
M_MASS    = 2.23
IX        = 0.16017
IY        = 0.04085
IZ        = 0.19866
IXZ       = 0.00008
PROP_DIST = 0.2
B_SPAN    = 1.2
C_CHORD   = 0.438
AW_AREA   = 0.478
PROP_R    = 0.2
PROP_A    = 0.1


def _make_J(device, dtype):
    """Inertia tensor as a [3, 3] constant on a given device/dtype."""
    return torch.tensor(
        [[IX,   0.0, -IXZ],
         [0.0,  IY,   0.0],
         [-IXZ, 0.0,  IZ ]],
        device=device, dtype=dtype,
    )


# ---------------------------------------------------------------------------
# Quaternion helpers (all batched on leading dim)
# ---------------------------------------------------------------------------
def quat2rotmat(q):
    """Body -> world rotation matrix. q: [..., 4] (w,x,y,z) -> [..., 3, 3]."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)
    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - w * x)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)

    row0 = torch.stack([r00, r01, r02], dim=-1)
    row1 = torch.stack([r10, r11, r12], dim=-1)
    row2 = torch.stack([r20, r21, r22], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def quaternion_derivative(q, omega):
    """qdot = 0.5 * Q(q) * omega.  q: [..., 4], omega: [..., 3] -> [..., 4]."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Q has shape [..., 4, 3]
    row0 = torch.stack([-x, -y, -z], dim=-1)
    row1 = torch.stack([ w, -z,  y], dim=-1)
    row2 = torch.stack([ z,  w, -x], dim=-1)
    row3 = torch.stack([-y,  x,  w], dim=-1)
    Q = torch.stack([row0, row1, row2, row3], dim=-2)

    return 0.5 * torch.matmul(Q, omega.unsqueeze(-1)).squeeze(-1)


def quat_multiply(a, b):
    """Hamilton product. a, b: [..., 4] -> [..., 4]."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


# ---------------------------------------------------------------------------
# Aerodynamic coefficients (vectorise naturally on tensors)
# ---------------------------------------------------------------------------
def coeff_L(delta):
    return -0.0004 * delta**3 - 0.0006 * delta**2 - 0.0007 * delta + 0.1629


def coeff_D(delta):
    return 0.0008 * delta**2 - 0.0007 * delta + 0.0059


# ---------------------------------------------------------------------------
# Speed / angle of attack at the wing (with prop wash, batched)
# ---------------------------------------------------------------------------
def _speed_angles(thrust_total, vbody, vwind_world, R):
    """
    thrust_total : [B]      total scalar thrust magnitude (= Tl + Tr)
    vbody        : [B, 3]   body-frame velocity
    vwind_world  : [B, 3]   wind in WORLD frame
    R            : [B, 3, 3] body -> world rotation

    Returns
        v     : [B, 3]   airspeed vector at the wing in body frame
        alpha : [B]      angle of attack (rad)
        beta  : [B]      sideslip (rad)
    """
    vinf = 0.0
    lp = 0.2  # disk-to-wing distance along prop axis

    # Momentum-theory induced velocity through the disk, clamped to T>=0.
    t_eff = torch.clamp(thrust_total, min=0.0)
    factor = 1.0 + (lp / PROP_R) / torch.sqrt(torch.tensor(
        1.0 + (lp / PROP_R) ** 2, device=vbody.device, dtype=vbody.dtype))
    u0 = torch.sqrt(2.0 * t_eff / (RHO * PROP_A) + vinf * vinf) * factor

    # Prop wash contribution in body frame: +x (slipstream blows past wing in -x,
    # so aircraft sees +x airspeed contribution).
    vprop_contrib = torch.zeros_like(vbody)
    vprop_contrib[..., 0] = u0

    # World-frame wind -> body frame via R.T
    # R is [B,3,3], vwind_world is [B,3]; we need R^T @ vwind_world.
    vwind_body = torch.matmul(R.transpose(-1, -2), vwind_world.unsqueeze(-1)).squeeze(-1)

    v = vbody - vwind_body + vprop_contrib

    speed_norm = torch.linalg.norm(v, dim=-1)
    safe = speed_norm > 1e-8

    alpha = torch.where(safe, torch.atan2(v[..., 2], v[..., 0]),
                        torch.zeros_like(speed_norm))
    # Clamp the asin argument for numerical safety.
    sin_beta = torch.where(safe, v[..., 1] / speed_norm.clamp(min=1e-8),
                           torch.zeros_like(speed_norm))
    beta = torch.asin(sin_beta.clamp(-1.0, 1.0))

    return v, alpha, beta


def _aero_forces(alpha, dr, dl):
    """Wind-frame [-D; 0; -L] rotated by alpha into body axes. Returns [B, 3]."""
    L = coeff_L(dl) + coeff_L(dr)
    D = coeff_D(dl) + coeff_D(dr)
    ca, sa = torch.cos(alpha), torch.sin(alpha)
    fx = -D * ca + L * sa
    fy = torch.zeros_like(fx)
    fz = -D * sa - L * ca
    return torch.stack([fx, fy, fz], dim=-1)


def _aero_moments(dr, dl, b_span, c_chord):
    """Roll from differential elevons; pitch/yaw zero. Returns [B, 3]."""
    L_roll = -coeff_L(dl) + coeff_L(dr)
    pitch = torch.zeros_like(L_roll)
    yaw   = torch.zeros_like(L_roll)
    return torch.stack([L_roll * b_span, pitch * c_chord, yaw * b_span], dim=-1)


# ---------------------------------------------------------------------------
# Main batched dynamics
# ---------------------------------------------------------------------------
def vtol_dynamics(state, control, vwind=None):
    """
    state   : [B, 13]
    control : [B, 4]     (Tr, Tl, dr, dl)
    vwind   : [B, 3] or None    wind in WORLD frame (defaults to zero)

    Returns
        sdot : [B, 13]
    """
    device, dtype = state.device, state.dtype

    # --- Unpack state ------------------------------------------------------
    vbody = state[:, 3:6]
    q     = state[:, 6:10]
    omega = state[:, 10:13]

    # Renormalise the quaternion before use.
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp(min=1e-12)

    # --- Unpack control ----------------------------------------------------
    Tr = control[:, 0]
    Tl = control[:, 1]
    dr = control[:, 2]
    dl = control[:, 3]

    # --- Wind --------------------------------------------------------------
    if vwind is None:
        vwind = torch.zeros_like(vbody)

    # --- Rotation matrix (body -> world) -----------------------------------
    R = quat2rotmat(q)

    # --- Thrust force in body frame: [Tl+Tr; 0; 0] ------------------------
    T_total = Tl + Tr
    Ft = torch.zeros_like(vbody)
    Ft[:, 0] = T_total

    # --- Local airspeed at the wing ---------------------------------------
    v_air, alpha, _beta = _speed_angles(T_total, vbody, vwind, R)

    # --- Dynamic pressure from LOCAL airspeed ------------------------------
    q_dyn = 0.5 * RHO * (v_air * v_air).sum(dim=-1)   # [B]

    # --- Gravity in body frame: R^T @ [0; 0; m*g] -------------------------
    g_world = torch.zeros_like(vbody)
    g_world[:, 2] = M_MASS * G_ACCEL
    Fg = torch.matmul(R.transpose(-1, -2), g_world.unsqueeze(-1)).squeeze(-1)

    # --- Aerodynamic forces and moments -----------------------------------
    Faero = _aero_forces(alpha, dr, dl) * (q_dyn * AW_AREA).unsqueeze(-1)
    Maero = _aero_moments(dr, dl, B_SPAN, C_CHORD) * (q_dyn * AW_AREA).unsqueeze(-1)

    # Differential-thrust yaw moment.
    Mt = torch.zeros_like(vbody)
    Mt[:, 2] = (Tr - Tl) * PROP_DIST

    F_total = Faero + Fg + Ft
    M_total = Maero + Mt

    # --- Kinematics / dynamics --------------------------------------------
    # pdot = R @ vbody
    pdot = torch.matmul(R, vbody.unsqueeze(-1)).squeeze(-1)

    # vbodydot = -omega x vbody + F/m
    vbodydot = -torch.linalg.cross(omega, vbody, dim=-1) + F_total / M_MASS

    # qdot = 0.5 * Q(q) * omega
    qdot = quaternion_derivative(q, omega)

    # omegadot = J^{-1} (cross(-omega, J@omega) + M)
    J = _make_J(device, dtype)
    Jomega = torch.matmul(omega, J.T)   # [B, 3]; equivalent to (J @ omega^T)^T
    rhs = -torch.linalg.cross(omega, Jomega, dim=-1) + M_total
    # Solve J @ x = rhs   <=>  x = (J^{-T} @ rhs^T)^T; but J is symmetric, so:
    omegadot = torch.linalg.solve(J, rhs.unsqueeze(-1)).squeeze(-1)

    return torch.cat([pdot, vbodydot, qdot, omegadot], dim=-1)
