""" 
Runs the simulation under a rocket avoidance test scenario:
- 18 satellites at 500 km altitude.
- A rocket launches and performs a GTO burn at t = 4 hours (14,400 seconds).
- The rocket crosses the constellation ring at t = 4.08 hours.
- Satellites have 0.17 N EP thrusters.
- The rocket trajectory is passed as a foreignTrajectory to all satellites.
- Plots and telemetry are saved to the results directory.
"""

import sys                                                                                                                                               
sys.path.append("/Users/danielludlow/Documents/Constellation-MPC")    

import numpy as np
from src.constellation import Constellation
from src.simulation.simulation import Simulation
import src.simulation.config as config
from src.physics.rocket import Rocket
from src.simulation.visualization import save_telemetry_to_csv, see_spacing_plots, print_summary_metrics

# Configure restrictive gimbal limit in the config module before initialization
config.CROSSLINK_GIMBAL_RANGE_RAD = np.deg2rad(20.0)

# Create 80 satellites at 350 km altitude
sampleConstellation = Constellation(number_of_satellites=80, altitude=350000)

# Create the rocket GTO trajectory (burn at 4 hours / 14,400 seconds)
rocket = Rocket(GTO_burn_time=14400.0)
rocket_traj = rocket.trajectory  # Get the trajectory object for the rocket directly

# Configure the rocket trajectory as a foreign obstacle for all satellites
for sat in sampleConstellation.satellites:
    sat.controller.foreignTrajectory = rocket_traj

print(f"Created constellation with {len(sampleConstellation.satellites)} satellites.")
print(f"Configured rocket trajectory with GTO burn at 4.0 hours (14,400 s).")
print(f"Assigning rocket trajectory to all satellite controllers.")

# Set up simulation for 6 hours (21600 seconds)
samepleSimulation = Simulation(
    constellation=sampleConstellation, 
    duration_sec=config.SIMULATION_DURATION_S, 
    dt=config.SIMULATION_TIME_STEP_S
)

print(f"Running simulation for {samepleSimulation.duration} seconds with dt = {samepleSimulation.dt} seconds...")
samepleSimulation.run()

print("Simulation complete. Generating visualization...")
see_spacing_plots(samepleSimulation)
print("Visualization complete.")

print("Generating telemetry csv...")
save_telemetry_to_csv(samepleSimulation)
print("CSV complete.")

print_summary_metrics(samepleSimulation)
