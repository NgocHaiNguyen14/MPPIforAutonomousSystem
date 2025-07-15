%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameters for autonomous vehicle simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function param = init_vehicle_params()
    param.L = 2.5;
    param.v_des = 10;

    param.road_width = 6;

    param.obstacles = struct(...
        'x', {30, 30, 40}, ... % Moved further
        'y', {1.5, -1.5, 0}, ...
        'r', {1, 0.8, 1.2});
    param.k_obst = 1000; % Reduced
    param.sigma_obst = 1.5;

    param.k_lane = 500;
end