import os
import shutil
import tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid GUI display errors
import matplotlib.pyplot as plt
import csv

def see_globe(simulation, target_orbit_time_sec=1.5):
    """
    Skipped for speed optimization.
    """
    print(" -> 3D Globe animation generation skipped.")
    return

def see_ring_animation(simulation, target_orbit_time_sec=1.5):
    """
    Skipped for speed optimization.
    """
    print(" -> 2D Radar Ring animation generation skipped.")
    return

def save_telemetry_to_csv(simulation, filename="telemetry.csv"):
    """
    Saves simulation telemetry to results directory
    """
    telemetry = simulation.telemetry
    if not telemetry:
        print("No telemetry data to save.")
        return
        
    workspace_dir = "/Users/danielludlow/Documents/Constellation-MPC"
    results_dir = os.path.join(workspace_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)
    
    sat_ids = sorted(list(telemetry.keys()))
    header = ["time_sec", "sat_id", "x_eci_m", "y_eci_m", "z_eci_m", "vx_eci_m_s", "vy_eci_m_s", "vz_eci_m_s",
              "thrust_x_N", "thrust_y_N", "thrust_z_N", "solver_status"]
              
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        n_steps = len(telemetry[sat_ids[0]]['time'])
        for step_idx in range(n_steps):
            for sat_id in sat_ids:
                row = [
                    telemetry[sat_id]['time'][step_idx],
                    sat_id,
                    telemetry[sat_id]['r'][step_idx][0],
                    telemetry[sat_id]['r'][step_idx][1],
                    telemetry[sat_id]['r'][step_idx][2],
                    telemetry[sat_id]['v'][step_idx][0],
                    telemetry[sat_id]['v'][step_idx][1],
                    telemetry[sat_id]['v'][step_idx][2],
                    telemetry[sat_id]['applied_thrust'][step_idx][0],
                    telemetry[sat_id]['applied_thrust'][step_idx][1],
                    telemetry[sat_id]['applied_thrust'][step_idx][2],
                    telemetry[sat_id]['solver_status'][step_idx]
                ]
                writer.writerow(row)
    print(f"Telemetry saved successfully to {filepath}")

def see_spacing_plots(simulation):
    """
    Generates a slot-free 4-panel data visualization dashboard focusing on spacing,
    safety clearance, and collective behavior:
    1. Space-Time Corridor Waterfall: Along-track positions in the average rotating frame (km) with the shaded rocket keep-out channel.
    2. Spacing Accordion Chain: Actual distances between adjacent satellites.
    3. Commanded Thrust Envelope: Max/Min thrust applied across the entire fleet.
    4. Distance to Rocket: Safety margin of the closest satellite to the rocket.
    """
    telemetry = simulation.telemetry
    if not telemetry:
        print("No telemetry data for spacing plots.")
        return

    import src.simulation.config as config

    sat_ids = sorted(list(telemetry.keys()))
    n_steps = len(telemetry[sat_ids[0]]['time'])
    time_sec = np.array(telemetry[sat_ids[0]]['time'])
    time_hr = time_sec / 3600.0
    dt = simulation.dt

    # Get orbit geometry
    r_earth = config.EARTH_EQUATORIAL_RADIUS_M
    r_orbit = r_earth + 350000.0
    pos_sat0 = telemetry[sat_ids[0]]['r']
    if len(pos_sat0) > 0:
        r_orbit = np.mean(np.linalg.norm(pos_sat0, axis=1))
    
    # Calculate spacing limits
    nominal_spacing_km = (2 * r_orbit * np.sin(np.pi / len(sat_ids))) / 1000.0

    # Detect rocket trajectory
    has_rocket = False
    foreign_traj = None
    intersect_long_rad = 0.0
    
    sat_ctrl = simulation.constellation.satellites[sat_ids[0]].controller
    if hasattr(sat_ctrl, 'foreignTrajectory') and sat_ctrl.foreignTrajectory is not None:
        foreign_traj = sat_ctrl.foreignTrajectory
        has_rocket = True
        
        # Find where rocket crosses the constellation ring (altitude equals r_orbit)
        cross_step = 0
        min_alt_diff = float('inf')
        for idx, t in enumerate(time_sec):
            r_pos, _ = foreign_traj.get_state_at_time(t)
            alt = np.linalg.norm(r_pos)
            diff = abs(alt - r_orbit)
            if diff < min_alt_diff:
                min_alt_diff = diff
                cross_step = idx
                
        # Find longitude at crossing
        pos_cross, _ = foreign_traj.get_state_at_time(time_sec[cross_step])
        intersect_long_rad = np.arctan2(pos_cross[1], pos_cross[0])

    # Import physics propagator
    from src.physics.orbit import propogate_orbit_rk4

    # Pre-propagate unperturbed J2 free-drift trajectory for all satellites
    unperturbed_pos = {}
    for sat_id in sat_ids:
        unperturbed_pos[sat_id] = np.zeros((n_steps, 3))
        r_temp = telemetry[sat_id]['r'][0].copy()
        v_temp = telemetry[sat_id]['v'][0].copy()
        mass = config.SATELLITE_MASS_KG
        for step_idx in range(n_steps):
            unperturbed_pos[sat_id][step_idx] = r_temp
            r_temp, v_temp = propogate_orbit_rk4(r_temp, v_temp, mass, np.zeros(3), dt)

    # Initialize data matrices
    rel_along_track_matrix = np.zeros((len(sat_ids), n_steps))
    spacing_chain_matrix = np.zeros((len(sat_ids), n_steps))
    thrust_matrix = np.zeros((len(sat_ids), n_steps))
    fleet_min_rocket_dist = np.zeros(n_steps)
    rocket_rel_pos_km = np.zeros(n_steps)

    for step_idx in range(n_steps):
        t = time_sec[step_idx]
        
        # 1. Find mean angle of the fleet to define the slot-free average rotating frame
        angles = []
        for sat_id in sat_ids:
            pos = telemetry[sat_id]['r'][step_idx]
            angles.append(np.arctan2(pos[1], pos[0]))
        
        # Unwrap angles to prevent jump discontinuities in mean calculation
        angles = np.array(angles)
        mean_angle = np.mean(np.unwrap(angles))
        
        # Calculate rocket position relative to average rotating frame
        if has_rocket:
            pos_r, _ = foreign_traj.get_state_at_time(t)
            angle_r = np.arctan2(pos_r[1], pos_r[0])
            diff_r = (angle_r - mean_angle + np.pi) % (2 * np.pi) - np.pi
            rocket_rel_pos_km[step_idx] = diff_r * r_orbit / 1000.0

            # Find closest satellite to rocket at this step
            dists = []
            for sat_id in sat_ids:
                pos = telemetry[sat_id]['r'][step_idx]
                dists.append(np.linalg.norm(pos - pos_r) / 1000.0)
            fleet_min_rocket_dist[step_idx] = min(dists)

        for i, sat_id in enumerate(sat_ids):
            pos = telemetry[sat_id]['r'][step_idx]
            vel = telemetry[sat_id]['v'][step_idx]
            vel_unit = vel / np.linalg.norm(vel)
            
            # Position relative to average rotating frame (along-track coordinate)
            angle_sat = np.arctan2(pos[1], pos[0])
            diff_sat = (angle_sat - mean_angle + np.pi) % (2 * np.pi) - np.pi
            rel_along_track_matrix[i, step_idx] = diff_sat * r_orbit / 1000.0

            # Chord distance to leading neighbor (actual vs unperturbed spacing)
            next_sat_id = sat_ids[(i + 1) % len(sat_ids)]
            pos_next = telemetry[next_sat_id]['r'][step_idx]
            d_actual = np.linalg.norm(pos_next - pos) / 1000.0  # km
            
            unperturbed_curr = unperturbed_pos[sat_id][step_idx]
            unperturbed_next = unperturbed_pos[next_sat_id][step_idx]
            d_unperturbed = np.linalg.norm(unperturbed_next - unperturbed_curr) / 1000.0  # km
            
            spacing_chain_matrix[i, step_idx] = d_actual - d_unperturbed  # Detrended spacing deviation!

            # Signed along-track thrust
            applied_f = telemetry[sat_id]['applied_thrust'][step_idx]
            thrust_matrix[i, step_idx] = applied_f.dot(vel_unit)

    # Set up dark style
    plt.style.use('dark_background')
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), facecolor='#121212')
    line_colors = plt.cm.cool(np.linspace(0, 1, len(sat_ids)))

    def style_axis(ax, title, ylabel):
        ax.set_facecolor('#1A1A1A')
        ax.grid(True, color='#333333', linestyle=':', alpha=0.5)
        ax.set_title(title, color='#FFFFFF', fontsize=12, pad=10)
        ax.set_ylabel(ylabel, color='#CCCCCC', fontsize=10)
        ax.tick_params(colors='#888888', labelsize=9)

    # --- Plot 1: Space-Time Corridor Waterfall ---
    ax1 = axs[0, 0]
    style_axis(ax1, "Collective Space-Time Avoidance Corridor", "Along-Track relative to Fleet Center [km]")
    for i, sat_id in enumerate(sat_ids):
        ax1.plot(time_hr, rel_along_track_matrix[i], color=line_colors[i], alpha=0.6, lw=1.0)
    
    if has_rocket:
        # Plot rocket trajectory relative to fleet center
        ax1.plot(time_hr, rocket_rel_pos_km, color='#FF5252', linestyle='-', lw=1.8, label="Rocket Path")
        # Shade safety corridor
        ax1.fill_between(time_hr, rocket_rel_pos_km - config.SAFETY_DISTANCE_M/1000.0, 
                         rocket_rel_pos_km + config.SAFETY_DISTANCE_M/1000.0, 
                         color='red', alpha=0.08, label="Keep-Out Zone (100km)")
        # Zoom Y-axis on the crossing corridor
        y_cross = rocket_rel_pos_km[cross_step]
        ax1.set_ylim([y_cross - 400.0, y_cross + 400.0])
        ax1.legend(loc="upper right", framealpha=0.2, fontsize=8)

    # --- Plot 2: Relative Spacing Chain (Accordion Plot) ---
    ax2 = axs[0, 1]
    style_axis(ax2, "Detrended Inter-Satellite Spacing (J2 Subtracted)", "Spacing Deviation from Free-Drift [km]")
    for i, sat_id in enumerate(sat_ids):
        ax2.plot(time_hr, spacing_chain_matrix[i], color=line_colors[i], alpha=0.6, lw=1.0)
    ax2.axhline(0.0, color='#00FFCC', linestyle='--', alpha=0.7, lw=1.2, label="Nominal spacing (J2 Free)")
    ax2.legend(loc="upper right", framealpha=0.2, fontsize=8)

    # --- Plot 3: Commanded Thrust Envelope ---
    ax3 = axs[1, 0]
    style_axis(ax3, "Actuator Thrust Envelope (Fleet Limits)", "Thrust Force [N]")
    
    # Calculate envelope
    max_thrust_fleet = np.max(thrust_matrix, axis=0)
    min_thrust_fleet = np.min(thrust_matrix, axis=0)
    mean_thrust_fleet = np.mean(thrust_matrix, axis=0)
    
    ax3.fill_between(time_hr, min_thrust_fleet, max_thrust_fleet, color='#00E5FF', alpha=0.2, label="Thrust Envelope")
    ax3.plot(time_hr, mean_thrust_fleet, color='#00E5FF', lw=1.2, linestyle='-', label="Fleet Mean")
    ax3.axhline(config.MAX_THRUST_N, color='crimson', linestyle=':', alpha=0.5, lw=1)
    ax3.axhline(-config.MAX_THRUST_N, color='crimson', linestyle=':', alpha=0.5, lw=1)
    ax3.set_xlabel("Time [hours]", color='#CCCCCC')
    ax3.legend(loc="upper right", framealpha=0.2, fontsize=8)

    # --- Plot 4: Fleet-to-Obstacle Safety Margin ---
    ax4 = axs[1, 1]
    style_axis(ax4, "Fleet Safety Margin (Distance to Rocket)", "Relative Separation [km]")
    if has_rocket:
        ax4.plot(time_hr, fleet_min_rocket_dist, color='#00E676', lw=1.8, label="Closest Satellite")
        safety_floor = config.SAFETY_DISTANCE_M / 1000.0
        ax4.axhline(safety_floor, color='#FF1744', linestyle='--', alpha=0.9, lw=1.5, label=f"Safety Floor ({int(safety_floor)} km)")
        ax4.legend(loc="upper right", framealpha=0.2, fontsize=8)
        ax4.set_yscale('log')
    else:
        ax4.text(0.5, 0.5, "No Rocket in Scenario", color='#666666', ha='center', va='center')
    ax4.set_xlabel("Time [hours]", color='#CCCCCC')

    # Save
    workspace_dir = "/Users/danielludlow/Documents/Constellation-MPC"
    results_dir = os.path.join(workspace_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, "spacing_metrics.png")
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, facecolor='#121212')
    plt.close(fig)
    plt.style.use('default')
    print(f"Spacing plots dashboard saved successfully to {plot_path}")

def print_summary_metrics(simulation):
    """
    Prints a detailed console summary detailing:
    Sat ID | MPC Success Rate | Min Link Spacing (km) | Max Link Spacing (km) | Safety Min (km) | Fuel (N-s)
    """
    telemetry = simulation.telemetry
    if not telemetry:
        print("No telemetry data to summarize.")
        return

    import src.simulation.config as config

    sat_ids = sorted(list(telemetry.keys()))
    n_steps = len(telemetry[sat_ids[0]]['time'])
    dt = simulation.dt

    print("\n" + "="*95)
    print("                      MPC PERFORMANCE & SAFETY SUMMARY (SLOT-FREE)")
    print("="*95)
    print(f"{'Sat ID':<8} | {'MPC Success Rate':<18} | {'Min Link Space (km)':<20} | {'Max Link Space (km)':<20} | {'Safety Min (km)':<15} | {'Fuel (N-s)':<12}")
    print("-"*95)

    total_solves = 0
    total_failures = 0
    total_gimbal_violations = 0
    total_safety_violations = 0
    
    gimbal_limit_deg = np.rad2deg(config.CROSSLINK_GIMBAL_RANGE_RAD)
    safety_distance_km = config.SAFETY_DISTANCE_M / 1000.0

    # Determine rocket properties
    has_rocket = False
    foreign_traj = None
    sat_ctrl = simulation.constellation.satellites[sat_ids[0]].controller
    if hasattr(sat_ctrl, 'foreignTrajectory') and sat_ctrl.foreignTrajectory is not None:
        foreign_traj = sat_ctrl.foreignTrajectory
        has_rocket = True

    for sat_id in sat_ids:
        sat_data = telemetry[sat_id]
        times = sat_data['time']
        
        # Calculate solver stats
        statuses = sat_data['solver_status']
        actual_solves = 0
        successful_solves = 0
        last_solve_time = -9999.0
        
        for step_idx, t in enumerate(times):
            if t == 0.0 or (t - last_solve_time) >= config.MPC_TIME_STEP_S:
                actual_solves += 1
                status = statuses[step_idx]
                if status in ["optimal", "optimal_inaccurate"]:
                    successful_solves += 1
                last_solve_time = t
        
        solver_success_pct = (successful_solves / actual_solves) * 100.0 if actual_solves > 0 else 100.0
        total_solves += actual_solves
        total_failures += (actual_solves - successful_solves)

        # Calculate link spacing bounds to leading neighbor
        spacing_vals = []
        next_sat_id = sat_ids[(sat_ids.index(sat_id) + 1) % len(sat_ids)]
        for step_idx in range(n_steps):
            pos = sat_data['r'][step_idx]
            pos_next = telemetry[next_sat_id]['r'][step_idx]
            spacing_vals.append(np.linalg.norm(pos_next - pos) / 1000.0)
            
        min_spacing_km = min(spacing_vals)
        max_spacing_km = max(spacing_vals)

        # Calculate safety minimum distance to rocket
        min_rocket_dist_km = float('inf')
        if has_rocket:
            for step_idx, t in enumerate(times):
                pos = sat_data['r'][step_idx]
                pos_r, _ = foreign_traj.get_state_at_time(t)
                dist = np.linalg.norm(pos - pos_r) / 1000.0
                if dist < min_rocket_dist_km:
                    min_rocket_dist_km = dist
        
        safety_min_str = f"{min_rocket_dist_km:.2f}" if min_rocket_dist_km != float('inf') else "N/A"

        # Calculate fuel usage
        thrusts = np.array(sat_data['applied_thrust'])
        thrust_mags = np.linalg.norm(thrusts, axis=1)
        fuel_impulse = np.sum(thrust_mags) * dt

        print(f"S{sat_id:<7} | {solver_success_pct:>16.2f}% | {min_spacing_km:>20.2f} | {max_spacing_km:>20.2f} | {safety_min_str:>15} | {fuel_impulse:>12.2f}")

        # Check safety and gimbal violations
        for step_idx in range(n_steps):
            pos = sat_data['r'][step_idx]
            vel = sat_data['v'][step_idx]
            
            # Safety checks
            pos_next = telemetry[next_sat_id]['r'][step_idx]
            dist = np.linalg.norm(pos_next - pos) / 1000.0
            if dist < safety_distance_km:
                total_safety_violations += 1

            # Pointing checks
            rel_pos = pos_next - pos
            rel_unit = rel_pos / np.linalg.norm(rel_pos)
            vel_unit = vel / np.linalg.norm(vel)
            dot_product = rel_unit.dot(vel_unit)
            viewing_angle = np.rad2deg(np.arccos(np.clip(abs(dot_product), -1.0, 1.0)))
            if viewing_angle > gimbal_limit_deg:
                total_gimbal_violations += 1

    print("-"*95)
    print("Constellation Fleet Summary:")
    print(f" * Total Optimization Cycles : {total_solves} solves")
    print(f" * Solver Success Rate       : {((total_solves - total_failures)/total_solves)*100.0:.3f}%")
    print(f" * Gimbal Range Violations   : {total_gimbal_violations} steps")
    print(f" * Safety Distance Breaches  : {total_safety_violations} steps")
    print("="*95 + "\n")
