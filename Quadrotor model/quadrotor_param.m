function params = quadrotor_param()
    % QUADROTOR_PARAM Returns a struct of fixed physical parameters for a quadrotor

    % Inertia around x-axis (kg·m²)
    params.Ixx = 1.2;

    % Inertia around y-axis (kg·m²)
    params.Iyy = 1.2;

    % Inertia around z-axis (kg·m²)
    params.Izz = 2.3;

    % Lift constant (N·s²)
    % Relates rotor speed squared to upward thrust
    params.k = 1;

    % Distance from rotor to center of mass (m)
    params.l = 0.25;

    % Mass of the quadrotor (kg)
    params.m = 2;

    % Drag constant (N·m·s²)
    % Relates rotor speed squared to torque around the z-axis
    params.b = 0.2;

    % Gravitational acceleration (m/s²)
    params.g = 9.81;
end
