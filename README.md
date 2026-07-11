# The Accordion Protocol: Distributed MPC for Active Launch Corridor Creation

This repository contains a high-fidelity satellite mega-constellation simulator designed to test **Distributed Model Predictive Control (dMPC)** for proactive along-track phasing—creating active, localized "accordion waves" to open safety corridors for transiting launch vehicles without breaking Inter-Satellite Laser Links (ISLLs).

---

## The Problem
As mega-constellations grow to tens of thousands of satellites, low Earth orbit (LEO) becomes highly congested. Current Space Traffic Management (STM) relies on:
1. **Launch COLA (Collision Avoidance) Holds**: Grounding rockets on the launch pad if their launch path intersects a satellite's safety envelope. With dense fleets, finding natural gaps becomes statistically impossible, paralyzing launch cadences.
2. **Reactive Altitude Jumps**: Forcing satellites to make emergency out-of-plane altitude adjustments. This is fuel-expensive and breaks local cross-satellite laser links (ISLLs), disrupting regional internet routing and global network performance.

---

## The Solution
Instead of the rocket dodging the constellation, **the constellation actively parts to open a path for the rocket.**

Because launch trajectories are known hours in advance, a string of satellites in the transiting orbital plane can coordinate a proactive "along-track phasing ripple." The satellites directly in the path slow down or speed up slightly, sliding forward and backward within their existing orbital lane like an accordion. 

This creates a temporary, 50-kilometer safety corridor exactly when and where the rocket transits the plane. Crucially, the dMPC optimization mathematically couples relative orbital mechanics with communication graph topology constraints. This ensures that the movement burden is distributed down the line, keeping the satellites within the distance and angular tracking limits of their ISLLs so data routing never drops. Once the vehicle passes, state-restoration weights in the MPC smoothly compress the fleet back into its default configuration.

---

## Repository Structure

```
├── results/                  # Simulation output artifacts
│   ├── orbits.gif            # 3D animated visualization of orbit propagation
│   └── telemetry.csv         # Exported CSV telemetry data (positions, thrust, links)
├── src/                      # Source Code
│   ├── config.py             # Global physical, satellite, and simulation constants
│   ├── constellation.py      # Fleet initialization and ring network topology setup
│   ├── physics/
│   │   └── orbit.py          # Orbit propagation (2-body gravity + J2 perturbation + control force)
│   ├── simulation/
│   │   ├── simulation.py     # Main step/run loop and telemetry logging database
│   │   └── visualization.py  # 3D animated GIF generator and dynamic CSV exporter
│   └── vehicle/
│       ├── controller.py     # Spacecraft dMPC controller (current consensus stub)
│       ├── satellite.py      # Satellite state vector and connection status
│       └── thruster.py       # Thruster actuator (force limits & testing profiles)
├── main.py                   # Entry point to run the simulation
└── .gitignore                # Excludes python caches and results outputs
```

---

## Current Verification Status
* **Physics Engine**: Verified and numerically stable. Evaluates absolute 3D Keplerian gravity + J2 equatorial perturbations via Runge-Kutta 4th Order (RK4) integration.
* **Actuators**: Thruster force integration ($F = ma$) is fully mapped to ECI acceleration in the propagator.
* **Crosslink Connection Checks**: Implemented stateless geometry verification. Calculates pointing pitch angle deviations (using ECI position and velocity vector dot products) to check if adjacent gimbals remain locked within their field-of-view limits.
* **Output Pipeline**: Tidy telemetry is compiled dynamically and exported to `results/telemetry.csv` and `results/orbits.gif`.

---

## How to Run

1. Clone or navigate to the repository directory.
2. Run the simulation driver script:
   ```bash
   python3 main.py
   ```
3. Check the `results/` directory for the telemetry logs and animated orbit path outputs.
