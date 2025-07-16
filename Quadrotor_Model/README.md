# This folder contains quadrotor and VTOL simulations with different methods
## VTOL simulation with MPPI
- The dynamics model: vtol2.m
- The MPPI simulation model: MPPI_vtol2.m - to see how the dynamics model changes over iterations.
- The MPPI simulation model with visualizations: MPPI_vtol2_visualization - to see the trajectory of VTOL and other parameters of the dynamics model.

## VTOL simulation with iLQR
- The dynamics model: vtol3_quaternion.m - dynamics rotation transferred to dynamics with quaternion
- The iLQR simulation (with visualization): iLQR_vtol3 - to see the trajectory of VTOL and other parameters of the dynamics model.

## Usage
(comment of Kiet)

params = quadrotor_param();

then run

generateQuadrotorModel(params)

## Reference

Dynamical model

Controller Design for Quadrotor-Slung Load
System with Swing Angle Constraints Using
Particle Swarm Optimization

Passivity Based Control for a Quadrotor UAV Transporting a
Cable-Suspended Payload with Minimum Swing

On the Effect of Slung Load on Quadrotor Performance
