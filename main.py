""" 
Runs the simulation 
Sample for now, just need to proove that orbits and attitude propogate correctly.
"""

from src.constellation import Constellation
from src.simulation.simulation import Simulation
import src.simulation.config as config
from src.simulation.visualization import see_globe, save_telemetry_to_csv, see_spacing_plots, see_ring_animation, print_summary_metrics

# 10 Sats in 500 km orbit
sampleConstellation = Constellation(number_of_satellites=18, altitude=500000)
print(f"Created constellation with {len(sampleConstellation.satellites)} satellites.")

samepleSimulation = Simulation(constellation=sampleConstellation, 
                               duration_sec= config.SIMULATION_DURATION, 
                               dt=config.SIMULATION_TIME_STEP)

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