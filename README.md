# The Accordion Protocol: Distributed MPC for Active Launch Corridor Creation

This repository contains a high-fidelity satellite mega-constellation simulator designed to test **Distributed Model Predictive Control (dMPC)** for proactive along-track phasing—creating active, localized "accordion waves" to open safety corridors for transiting launch vehicles without breaking Inter-Satellite Laser Links (ISLLs).

---

## The Problem
As mega-constellations grow to tens of thousands of satellites, low Earth orbit (LEO) becomes highly congested. Current Space Traffic Management (STM) relies on:
1. **Launch COLA (Collision Avoidance) Holds**: Grounding rockets on the launch pad if their launch path intersects a satellite's safety envelope. With dense fleets, finding natural gaps becomes statistically impossible, paralyzing launch cadences.
2. **Reactive Altitude Jumps**: Forcing satellites to make emergency out-of-plane altitude adjustments. This is fuel-expensive and breaks local cross-satellite laser links (ISLLs), disrupting regional internet routing and global network performance.

---

## The Solution: Just-In-Time (JIT) Phasing
Instead of the rocket dodging the constellation, **the constellation actively parts to open a path for the rocket.**

Because launch trajectories are known in advance, a string of satellites in the transiting plane can coordinate a proactive along-track phasing ripple. Satellites ahead of the crossing point speed up slightly, and satellites behind slow down, sliding open a temporary $50\text{ km}$ safety corridor like an accordion.

### Key Operational Clarifications:
1. **Just-In-Time (JIT) Corridor**: Satellites do *not* hold the safety corridor open for the entire 1-hour launch window, which would exhaust fuel and break links. Instead, the constellation remains nominal until liftoff occurs, locking in the exact transit time. The dMPC then coordinates the corridor to open and close within a tight **3-to-5 minute window** centered exactly on the rocket's transit.
2. **Launch Window Dynamics**:
   * **Low-Thrust Electric Propulsion (e.g., $170\text{ mN}$)**: Requires a warning of $\sim 4\text{ hours}$, starting a slow, fuel-efficient drift during the terminal countdown sequence. 
   * **High-Thrust Chemical Propulsion (e.g., $10\text{ N}$)**: Requires a warning of $\sim 30\text{ minutes}$, allowing the constellation to react and open a new corridor in the event of a launch hold or clock recycle.
3. **Resilience to Scrubs**: If a launch is scrubbed at the last second (e.g., at $T-10\text{ seconds}$), the dMPC immediately detects the removal of the keep-out constraint, reverses thrust, and slides the satellites back to their nominal slots with negligible fuel waste ($< 0.1\text{ m/s}$ of $\Delta V$).
4. **The Zipper Wave**: As the rocket ascends through multiple shells, the safety corridors open and shut sequentially plane-by-plane like a zipper, minimizing regional communication network footprint.

---

## Repository Structure

```
├── results/                  # Simulation output artifacts
│   ├── orbits.gif            # 3D animated visualization of orbit propagation
│   ├── ring_orbits.gif       # 2D animated visualization of constellation links
│   ├── spacing_metrics.png   # Plots of inter-satellite spacing and connectivity %
│   └── telemetry.csv         # Exported CSV telemetry data (positions, thrust, links)
├── src/                      # Source Code
│   ├── config.py             # Global physical, satellite, and simulation constants
│   ├── constellation.py      # Fleet initialization and ring network topology setup
│   ├── physics/
│   │   └── orbit.py          # Orbit propagation (rk4 math & satellite step wrapper)
│   ├── simulation/
│   │   ├── simulation.py     # Main step/run loop and telemetry logging database
│   │   └── visualization.py  # 3D/2D animation generators & telemetry CSV exporter
│   └── vehicle/
│       ├── controller.py     # Spacecraft dMPC controller (GNC interface)
│       ├── crosslink.py      # Physical laser link (pointing checks & transmission)
│       ├── satellite.py      # Satellite state vector and local hardware attributes
│       ├── thruster.py       # Thruster actuator (force limits & testing profiles)
│       └── trajectory.py     # Trajectory representation and interpolation helper
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
