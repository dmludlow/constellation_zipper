""" 
Runs the simulation under a coordinated drift stress test:
- 1 satellite (Sat 5) experiences a thruster failure and starts with a -2 m/s velocity deficit.
- All satellites have a restrictive but physically feasible gimbal range of 13 degrees.
- Plots and telemetry are saved to the results directory.
"""

import numpy as np
from src.constellation import Constellation
from src.simulation.simulation import Simulation
import src.simulation.config as config
from src.simulation.visualization import see_globe, save_telemetry_to_csv, see_spacing_plots, see_ring_animation, print_summary_metrics

# Configure restrictive gimbal limit in the config module before initialization
config.CROSSLINK_GIMBAL_RANGE_RAD = np.deg2rad(13.0)

# Create 18 satellites at 500 km altitude
sampleConstellation = Constellation(number_of_satellites=18, altitude=500000)

# Configure the thruster out scenario on Sat 5
sat5 = sampleConstellation.satellites[5]
sat5.thruster.MAX_THRUST_N = 0.0  # Engine failure
vel_unit = sat5.velocityECI / np.linalg.norm(sat5.velocityECI)
sat5.velocityECI -= 2.0 * vel_unit  # Inject 2 m/s drift velocity deficit

print(f"Created constellation with {len(sampleConstellation.satellites)} satellites.")
print(f"Configured Sat 5 with Thruster Out Failure and -2.0 m/s velocity deficit.")
print(f"Configured all crosslinks with a restrictive gimbal limit of 13.0 degrees.")

# Set up simulation
samepleSimulation = Simulation(
    constellation=sampleConstellation, 
    duration_sec=config.SIMULATION_DURATION_S, 
    dt=config.SIMULATION_TIME_STEP_S
)

print(f"Running simulation for {samepleSimulation.duration} seconds with dt = {samepleSimulation.dt} seconds...")
samepleSimulation.run()

print("Simulation complete. Generating visualization...")
see_globe(samepleSimulation)
see_spacing_plots(samepleSimulation)
see_ring_animation(samepleSimulation)
print("Visualization complete.")

print("Generating telemetry csv...")
save_telemetry_to_csv(samepleSimulation)
print("CSV complete.")

print_summary_metrics(samepleSimulation)
