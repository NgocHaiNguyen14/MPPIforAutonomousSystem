function param = init_vehicle_params()
    param.L = 2.5;
    param.v_des = 10;

    param.road_width = 10; % Expanded road width

    % Static obstacles removed
    param.obstacles = struct('x', {}, 'y', {}, 'r', {});
    param.k_obst = 200;
    param.sigma_obst = 1.5;

    % Moving vehicles (adjusted y0 for lane centers)
    param.moving_vehicles = struct(...
        'x0', {10, 15}, ...
        'y0', {2.5, -2.5}, ...
        'vx', {5, 3}, ...
        'length', {4, 4}, ...
        'width', {1.8, 1.8}, ...
        'r', {2.2, 2.2});

    param.k_lane = 200;
end