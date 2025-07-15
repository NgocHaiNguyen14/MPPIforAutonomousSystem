%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Kinematic bicycle model for autonomous vehicle
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function xdot = vehicle_dynamics(t, x, u, param)
    L = param.L;
    x_pos = x(1);
    y_pos = x(2);
    psi = x(3);
    v = max(min(x(4), 20), 0); % Bound velocity
    delta = min(max(u(1), -0.2), 0.2); % Allow both left and right turns
    a = min(max(u(2), -1), 1); % Limit acceleration

    xdot = zeros(4,1);
    xdot(1) = v * cos(psi);
    xdot(2) = v * sin(psi);
    xdot(3) = (v / L) * tan(delta);
    xdot(4) = a;

    if any(isnan(xdot) | isinf(xdot))
        warning('Invalid xdot at t=%.2f: %s', t, mat2str(xdot));
    end
end