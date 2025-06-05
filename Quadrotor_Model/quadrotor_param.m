function params = quadrotor_param()
    % QUADROTOR_PARAM Returns a struct of fixed physical parameters for a quadrotor

    % Inertia around x-axis (kg·m²)
    params.Ixx = 7*1e-3;

    % Inertia around y-axis (kg·m²)
    params.Iyy = 7*1e-3;

    % Inertia around z-axis (kg·m²)
    params.Izz = 12*1e-3;

    % Lift constant (N·s²)
    % Relates rotor speed squared to upward thrust
    params.k = 8.55*(1e-6)*91.61;

    % Distance from rotor to center of mass (m)
    params.l = 0.17;

    % Mass of the quadrotor (kg)
    params.m = 0.716;

    % Drag constant (N·m·s²)
    % Relates rotor speed squared to torque around the z-axis
    params.b = 1.6*(1e-2)*91.61;

    % Gravitational acceleration (m/s²)
    params.g = 9.81;
end
