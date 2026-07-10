""" 
Runs the simulation 
Sample for now, just need to proove that orbits and attitude propogate correctly.
"""

from src.constellation import Constellation
from src.simulation.simulation import Simulation
from src.simulation.visualization import see_globe

# 10 Sats in 500 km orbit
sampleConstellation = Constellation(number_of_satellites=15, altitude=500000)
print(f"Created constellation with {len(sampleConstellation.satellites)} satellites.")

# 2 hours, dt = 1 second
samepleSimulation = Simulation(constellation=sampleConstellation, duration_sec= (5650), dt=1.0)

print(f"Running simulation for {samepleSimulation.duration} seconds with dt = {samepleSimulation.dt} seconds...")
samepleSimulation.run()
print("Simulation complete. Generating visualization...")

see_globe(samepleSimulation)
print("Visualization complete. Check 'orbits.gif' in the root directory.")