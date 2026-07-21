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

## Implementation Details

### 1. Distributed Model Predictive Control (dMPC)
Each satellite runs an independent local MPC solver formulated using **Disciplined Parameterized Programming (DPP)**. 
* **Clohessy-Wiltshire (CW) Dynamics**: Relative satellite motion is linearized in local Hill/LVLH frames.
* **Compile-Once Architecture**: Optimization parameters, decision variables, objective functions, and constraints are instantiated once in the `Controller.__init__` constructor.
* **Millisecond Solves**: During runtime, parameters (initial state `x0_param`, neighbor ECI coordinates, and max thrust) are updated dynamically, bypassing the CVXPY compilation pipeline and solving in **$< 1\text{ ms}$** via the `OSQP` solver.
* **Information Isolation**: Each controller maintains a unique instance dictionary `neighboringSatTrajectories` to isolate plan-sharing information flow, preventing data leaks and representing realistic on-board GNC flight computers.

### 2. Physical constraints under J2 Perturbations
* **Laser Link Pointing (Gimbal Cone)**: Spacing separation must satisfy the geometry of a 3D gimbal limit $\phi_{max}$. Since radial separation $\Delta x$ and along-track separation $\Delta y$ couple, the absolute value constraint:
  $$\Delta y \cdot \sin(\phi_{max}) \geq |\Delta x| \cdot \cos(\phi_{max})$$
  is split into two linear inequalities to maintain full DPP compliance:
  1. $\Delta y \cdot \sin(\phi_{max}) \geq \Delta x \cdot \cos(\phi_{max})$
  2. $\Delta y \cdot \sin(\phi_{max}) \geq -\Delta x \cdot \cos(\phi_{max})$
* **The Feasibility Boundary**: Due to Earth's equatorial bulge (J2), orbits naturally breathe (eccentricity wiggles). Because satellites are equipped only with along-track thrusters, they cannot directly control radial breathing. If the gimbal limit is too strict (e.g. $10.5^\circ$), the radial separation wiggles exceed the pointing bounds, making the optimization problem mathematically infeasible. A moderate limit of **$13.0^\circ$** is verified to be 100% stable and feasible.

### 3. Multi-Rate Caching
The physical constellation simulation propagates at a time step of $10\text{ s}$ (`SIMULATION_TIME_STEP_S`). However, the MPC optimization is executed at a coarser rate of $5\text{ minutes}$ (`MPC_TIME_STEP_S`). On intermediate simulation steps, cached optimal control forces are rotated from the nominal reference frame to the current ECI frame, reducing solver calls by 30x without losing tracking fidelity.

---

## How to Run

### 1. Baseline Station Keeping Simulation
Runs the constellation simulation under normal flight conditions (J2 active, $20^\circ$ gimbal range) and exports telemetry plots.
```bash
python3 main.py
```

### 2. Coordinated Drift / Thruster-Out Stress Test
Runs a simulation where Sat 5 experiences a total thruster failure (`MAX_THRUST_N = 0.0`) and drifts with a $-2.0\text{ m/s}$ along-track delta-V deficit under a tight $13.0^\circ$ gimbal limit. The adjacent satellites coordinate and actively adjust their orbits to keep the drifting satellite within their pointing cones.
```bash
python3 testing_thurster_out.py
```

Check the `results/` directory for the output animated orbit paths (`orbits.gif`), 2D cluster views (`ring_orbits.gif`), and individual spacing plots (`spacing_metrics.png`).
