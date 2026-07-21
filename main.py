""" 
Runs the simulation 
Sample for now, just need to proove that orbits and attitude propogate correctly.
"""

import argparse
from src.constellation import Constellation
from src.simulation.simulation import Simulation
import src.simulation.config as config
from src.simulation.visualization import see_globe, save_telemetry_to_csv, see_spacing_plots, see_ring_animation, print_summary_metrics


# Should use __name__ == "__main__" like this to prevent code from running on import.
# Use argparse for command line arguments to specify number of satellites, altitude_m, duration_s, and dt.
if __name__ == "__main__":
    # Now we can run it like this: python3 main.py --num_sats 10 --altitude_m 500000 --duration_s 7200 --dt 1.0 and change the parameters as needed.
    parser = argparse.ArgumentParser(description="Run a satellite constellation simulation.")
    parser.add_argument("--num_sats", type=int, default=18, help="Number of satellites in the constellation (default: 15)")
    parser.add_argument("--altitude_m", type=int, default=500000, help="Altitude of the satellites in meters (default: 500000)")
    parser.add_argument("--duration_s", type=int, default=config.SIMULATION_DURATION_S, help="Duration of the simulation in seconds (default: 5650)")
    parser.add_argument("--dt", type=float, default=config.SIMULATION_TIME_STEP_S, help="Time step for the simulation in seconds (default: 1.0)")
    args = parser.parse_args()
    # 10 Sats in 500 km orbit
    sampleConstellation = Constellation(number_of_satellites=args.num_sats, altitude_m=args.altitude_m)
    print(f"Created constellation with {len(sampleConstellation.satellites)} satellites.")

    # 2 hours, dt = 1 second
    samepleSimulation = Simulation(constellation=sampleConstellation, duration_s=args.duration_s, dt=args.dt)

    print(f"Running simulation for {samepleSimulation.duration_s} seconds with dt = {samepleSimulation.dt} seconds...")
    samepleSimulation.run()
    print("Simulation complete. Generating visualization...")

    see_globe(samepleSimulation)
    print("Visualization complete. Check 'orbits.gif' in the root directory.")