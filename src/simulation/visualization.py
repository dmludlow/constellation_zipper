import os
import shutil
import tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid GUI display errors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

def see_globe(simulation, target_orbit_time_sec=1.5):
    """
    Generates a 3D animated GIF of the satellites orbiting the Earth based on simulation telemetry.
    The GIF is saved as 'orbits.gif' in the results folder.
    """
    # Extract telemetry
    telemetry = simulation.telemetry
    if not telemetry:
        print("No telemetry data found in simulation.")
        return
    
    # Get satellite IDs
    sat_ids = list(telemetry.keys())
    if not sat_ids:
        print("No satellites found in telemetry.")
        return
    
    # Number of steps
    n_steps = len(telemetry[sat_ids[0]]['time'])
    if n_steps == 0:
        print("Telemetry contains no steps.")
        return
    
    # Downsample steps to keep GIF generation reasonably fast and file size small
    frame_step = max(1, n_steps // 100)
    frame_indices = list(range(0, n_steps, frame_step))
    if len(frame_indices) == 0 or frame_indices[-1] != n_steps - 1:
        frame_indices.append(n_steps - 1)
        
    r_earth = 6378137.0  # Earth equatorial radius
    try:
        import src.config as config
        r_earth = config.EARTH_EQUATORIAL_RADIUS
    except Exception:
        pass

    # Calculate orbital period dynamically to set perfect playback speed (target_orbit_time_sec per orbit)
    r_orbit = r_earth + 500000.0
    for sat_id in sat_ids:
        pos = telemetry[sat_id]['r']
        if len(pos) > 0:
            r_orbit = np.mean(np.linalg.norm(pos, axis=1))
            break
    GM = 3.986004418e14
    orbital_period = 2 * np.pi * np.sqrt(r_orbit**3 / GM)
    steps_per_orbit = orbital_period / simulation.dt
    frames_per_orbit = steps_per_orbit / frame_step
    duration_ms = int((target_orbit_time_sec * 1000.0) / frames_per_orbit)
    duration_ms = max(20, min(2000, duration_ms))
        
    # Set dark style
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(10, 8), facecolor='#121212')
    ax = fig.add_subplot(111, projection='3d', facecolor='#121212')
    
    max_r = 0.0
    for sat_id in sat_ids:
        pos = telemetry[sat_id]['r']
        if len(pos) > 0:
            max_r = max(max_r, np.max(np.linalg.norm(pos, axis=1)))
    if max_r == 0.0:
        max_r = r_earth + 500000.0
    
    limit = max_r * 1.1
    
    frames = []
    temp_dir = tempfile.mkdtemp()
    
    try:
        for frame_idx, step_idx in enumerate(frame_indices):
            ax.clear()
            ax.set_facecolor('#121212')
            
            # Set limits and aspect ratio
            ax.set_xlim([-limit, limit])
            ax.set_ylim([-limit, limit])
            ax.set_zlim([-limit, limit])
            ax.set_xlabel("X (ECI) [m]", color='#888888')
            ax.set_ylabel("Y (ECI) [m]", color='#888888')
            ax.set_zlabel("Z (ECI) [m]", color='#888888')
            ax.tick_params(colors='#666666')
            
            # Hide grid lines for a cleaner space radar look
            ax.grid(False)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            
            # Draw the Earth as a glowing blue sphere
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x_earth = r_earth * np.outer(np.cos(u), np.sin(v))
            y_earth = r_earth * np.outer(np.sin(u), np.sin(v))
            z_earth = r_earth * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x_earth, y_earth, z_earth, color='#102E5C', alpha=0.5, edgecolor='#00E5FF', lw=0.3)
            
            # Draw orbital paths and current satellite positions
            for sat_id in sat_ids:
                r_hist = telemetry[sat_id]['r']
                
                # Draw complete path in thin neon purple
                ax.plot(r_hist[:, 0], r_hist[:, 1], r_hist[:, 2], color='#8A2BE2', linestyle=':', alpha=0.3, lw=1.0)
                
                # Current position
                curr_pos = r_hist[step_idx]
                
                # Highlight Sat 5 in red, others in cyan
                color = '#FF1744' if sat_id == 5 else '#00E5FF'
                size = 40 if sat_id == 5 else 20
                ax.scatter(curr_pos[0], curr_pos[1], curr_pos[2], color=color, marker='o', s=size, zorder=5)
                ax.text(curr_pos[0], curr_pos[1], curr_pos[2], f" S{sat_id}", color=color, fontsize=8)
            
            time_sec = telemetry[sat_ids[0]]['time'][step_idx]
            ax.set_title(f"Constellation Orbit Simulation\nTime: {time_sec:.1f} s", color='#FFFFFF', fontsize=12, pad=15)
            
            # Save frame
            frame_path = os.path.join(temp_dir, f"frame_{frame_idx:04d}.png")
            plt.savefig(frame_path, dpi=100, bbox_inches='tight', facecolor='#121212')
            img = Image.open(frame_path)
            img.load()
            frames.append(img)
            
        results_dir = os.path.join("/Users/danielludlow/Documents/Constellation-MPC", "results")
        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(results_dir, "orbits.gif")
        
        if frames:
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration_ms,
                loop=0
            )
            print(f"GIF saved successfully to {output_path}")
    finally:
        plt.close(fig)
        shutil.rmtree(temp_dir)
        plt.style.use('default')


import csv

def save_telemetry_to_csv(simulation, filename="telemetry.csv"):
    """
    Generates a CSV file of the simulation telemetry in the 'results' directory.
    This method is general and will export whatever keys and dimensions are in the telemetry.
    """
    # Ensure results directory exists
    workspace_dir = "/Users/danielludlow/Documents/Constellation-MPC"
    results_dir = os.path.join(workspace_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, filename)

    telemetry = simulation.telemetry
    if not telemetry:
        print("No telemetry data to save.")
        return

    # 1. Determine all columns dynamically
    sample_sat_id = list(telemetry.keys())[0]
    sample_sat_data = telemetry[sample_sat_id]
    
    headers = ["satellite_id"]
    
    # Ensure time is always the second column
    has_time = 'time' in sample_sat_data
    if has_time:
        headers.append('time')
        
    for key, val in sample_sat_data.items():
        if key == 'time':
            continue
        
        # Safely check if the telemetry array has data
        if val is not None and getattr(val, 'size', 0) > 0:
            # If the array is 0-dimensional (scalar or dictionary wrapper), treat as scalar
            if hasattr(val, 'ndim') and val.ndim == 0:
                headers.append(key)
                continue
                
            first_elem = val[0]
            
            # Check if the element is an array or vector (has dimensions)
            is_vector = False
            if isinstance(first_elem, (list, np.ndarray)):
                is_vector = True
            elif hasattr(first_elem, 'ndim') and first_elem.ndim > 0:
                is_vector = True
                
            if is_vector:
                num_elements = len(first_elem)
                if num_elements == 3:
                    for label in ['x', 'y', 'z']:
                        headers.append(f"{key}_{label}")
                else:
                    for idx in range(num_elements):
                        headers.append(f"{key}_{idx}")
            else:
                # Scalar data (like booleans or simple floats)
                headers.append(key)

    # 2. Write rows to CSV
    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)
        
        for sat_id, sat_data in telemetry.items():
            n_steps = len(sat_data.get('time', []))
            
            for step_idx in range(n_steps):
                row = [sat_id]
                if has_time:
                    row.append(sat_data['time'][step_idx])
                    
                for key, val in sat_data.items():
                    if key == 'time':
                        continue
                    if len(val) > step_idx:
                        # Handle 0-dimensional arrays safely
                        if hasattr(val, 'ndim') and val.ndim == 0:
                            row.append(val.item())
                        else:
                            elem = val[step_idx]
                            if isinstance(elem, (list, np.ndarray)):
                                row.extend(elem)
                            else:
                                row.append(elem)
                    else:
                        row.append(None)
                writer.writerow(row)
                
    print(f"Telemetry saved successfully to {csv_path}")


def see_spacing_plots(simulation):
    """
    Generates a premium 4-panel dark-mode dashboard showing:
    1. Inter-satellite along-track spacing (with safety corridors).
    2. Link pointing angles vs. gimbal limit.
    3. Applied control thrust (signed along-track).
    4. Relative orbital altitude breathing (radial separation).
    """
    telemetry = simulation.telemetry
    if not telemetry:
        print("No telemetry data for spacing plots.")
        return

    import src.config as config

    sat_ids = sorted(list(telemetry.keys()))
    n_steps = len(telemetry[sat_ids[0]]['time'])
    time_sec = np.array(telemetry[sat_ids[0]]['time'])
    time_hr = time_sec / 3600.0

    # Initialize data matrices
    spacing_matrix = np.zeros((len(sat_ids), n_steps))
    angle_matrix = np.zeros((len(sat_ids), n_steps))
    thrust_matrix = np.zeros((len(sat_ids), n_steps))
    radial_sep_matrix = np.zeros((len(sat_ids), n_steps))

    for step_idx in range(n_steps):
        for i, sat_id in enumerate(sat_ids):
            # Current satellite state
            pos = telemetry[sat_id]['r'][step_idx]
            vel = telemetry[sat_id]['v'][step_idx]
            r_norm = np.linalg.norm(pos)
            
            # Leading neighbor state
            next_sat_id = sat_ids[(i + 1) % len(sat_ids)]
            pos_next = telemetry[next_sat_id]['r'][step_idx]
            r_next_norm = np.linalg.norm(pos_next)

            # 1. Along-track separation (chord distance in km)
            dist = np.linalg.norm(pos_next - pos)
            spacing_matrix[i, step_idx] = dist / 1000.0  # km

            # 2. Link Pointing Angle (deg)
            rel_pos = pos_next - pos
            rel_unit = rel_pos / np.linalg.norm(rel_pos)
            vel_unit = vel / np.linalg.norm(vel)
            dot_product = rel_unit.dot(vel_unit)
            # Clip for arccos bounds safety
            viewing_angle_rad = np.arccos(np.clip(abs(dot_product), -1.0, 1.0))
            angle_matrix[i, step_idx] = np.rad2deg(viewing_angle_rad)

            # 3. Signed along-track thrust
            applied_f = telemetry[sat_id]['applied_thrust'][step_idx]
            thrust_matrix[i, step_idx] = applied_f.dot(vel_unit)  # Newtons

            # 4. Radial separation (altitude difference in km)
            radial_sep_matrix[i, step_idx] = (r_next_norm - r_norm) / 1000.0  # km

    # Set up dark style
    plt.style.use('dark_background')
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), facecolor='#121212')
    
    # Custom colors
    line_colors = plt.cm.cool(np.linspace(0, 1, len(sat_ids)))
    
    # Helper to style axes
    def style_axis(ax, title, ylabel):
        ax.set_facecolor('#1A1A1A')
        ax.grid(True, color='#333333', linestyle=':', alpha=0.5)
        ax.set_title(title, color='#FFFFFF', fontsize=12, pad=10)
        ax.set_ylabel(ylabel, color='#CCCCCC', fontsize=10)
        ax.tick_params(colors='#888888', labelsize=9)
        
    # --- Plot 1: Spacing ---
    ax1 = axs[0, 0]
    style_axis(ax1, "Inter-Satellite Along-Track Spacing", "Spacing [km]")
    for i, sat_id in enumerate(sat_ids):
        ax1.plot(time_hr, spacing_matrix[i], color=line_colors[i], alpha=0.7, lw=1.2)
    
    # Highlight safe zone
    phi_max = config.CROSSLINK_GIMBAL_RANGE
    # Calculate nominal orbital radius to draw max/min spacing
    r_orbit = 6871e3
    max_spacing_km = (2 * r_orbit * np.sin(phi_max)) / 1000.0
    min_spacing_km = config.SAFETY_DISTANCE / 1000.0
    
    ax1.axhline(min_spacing_km, color='crimson', linestyle='--', alpha=0.9, lw=1.5, label=f"Safety Floor ({int(min_spacing_km)} km)")
    ax1.axhline(max_spacing_km, color='gold', linestyle='--', alpha=0.8, lw=1.5, label=f"Gimbal Ceiling ({int(max_spacing_km)} km)")
    
    # Color safe zone green shading
    ax1.axhspan(min_spacing_km, max_spacing_km, color='limegreen', alpha=0.03)
    ax1.legend(loc="upper right", framealpha=0.2, fontsize=8)
    
    # --- Plot 2: Pointing Angles ---
    ax2 = axs[0, 1]
    style_axis(ax2, "Laser Link Pointing Angles", "Viewing Angle [deg]")
    for i, sat_id in enumerate(sat_ids):
        ax2.plot(time_hr, angle_matrix[i], color=line_colors[i], alpha=0.7, lw=1.2)
        
    gimbal_limit_deg = np.rad2deg(config.CROSSLINK_GIMBAL_RANGE)
    ax2.axhline(gimbal_limit_deg, color='crimson', linestyle='-', alpha=0.9, lw=1.5, label=f"Gimbal Limit ({gimbal_limit_deg:.1f}°)")
    ax2.legend(loc="upper right", framealpha=0.2, fontsize=8)
    
    # --- Plot 3: Actuator Thrust ---
    ax3 = axs[1, 0]
    style_axis(ax3, "Commanded Along-Track Thrust", "Thrust Force [N]")
    for i, sat_id in enumerate(sat_ids):
        ax3.plot(time_hr, thrust_matrix[i], color=line_colors[i], alpha=0.7, lw=1.2)
        
    max_t = config.MAX_THRUST
    ax3.axhline(max_t, color='crimson', linestyle=':', alpha=0.5, lw=1)
    ax3.axhline(-max_t, color='crimson', linestyle=':', alpha=0.5, lw=1)
    ax3.set_xlabel("Time [hours]", color='#CCCCCC')

    # --- Plot 4: Radial Separation (Breathing) ---
    ax4 = axs[1, 1]
    style_axis(ax4, "Relative Altitude (Radial Breathing)", "Radial Separation [km]")
    for i, sat_id in enumerate(sat_ids):
        ax4.plot(time_hr, radial_sep_matrix[i], color=line_colors[i], alpha=0.7, lw=1.2)
    ax4.set_xlabel("Time [hours]", color='#CCCCCC')

    # Save
    workspace_dir = "/Users/danielludlow/Documents/Constellation-MPC"
    results_dir = os.path.join(workspace_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, "spacing_metrics.png")
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, facecolor='#121212')
    plt.close(fig)
    plt.style.use('default')  # Restore default style
    print(f"Spacing plots dashboard saved successfully to {plot_path}")


def see_ring_animation(simulation, target_orbit_time_sec=1.5):
    """
    Generates a premium 2D radar-style animated GIF showing the equatorial ring of satellites,
    their spacing, and their crosslink connection statuses:
    - Green = Active & Stable link
    - Gold = Warning (pointing angle within 1.5 degrees of limit)
    - Red = Broken link
    The failed thruster satellite (Sat 5) is highlighted in bright red.
    Saved as 'ring_orbits.gif' in the results folder.
    """
    telemetry = simulation.telemetry
    if not telemetry:
        print("No telemetry data for ring animation.")
        return

    import src.config as config

    sat_ids = sorted(list(telemetry.keys()))
    n_steps = len(telemetry[sat_ids[0]]['time'])

    # Downsample steps to keep file size small (~100 frames)
    frame_step = max(1, n_steps // 100)
    frame_indices = list(range(0, n_steps, frame_step))
    if len(frame_indices) == 0 or frame_indices[-1] != n_steps - 1:
        frame_indices.append(n_steps - 1)

    r_earth = 6378137.0
    try:
        r_earth = config.EARTH_EQUATORIAL_RADIUS
    except Exception:
        pass

    # Calculate orbital period dynamically to set perfect playback speed (target_orbit_time_sec per orbit)
    r_orbit = r_earth + 500000.0
    for sat_id in sat_ids:
        pos = telemetry[sat_id]['r']
        if len(pos) > 0:
            r_orbit = np.mean(np.linalg.norm(pos, axis=1))
            break
    GM = 3.986004418e14
    orbital_period = 2 * np.pi * np.sqrt(r_orbit**3 / GM)
    steps_per_orbit = orbital_period / simulation.dt
    frames_per_orbit = steps_per_orbit / frame_step
    duration_ms = int((target_orbit_time_sec * 1000.0) / frames_per_orbit)
    duration_ms = max(20, min(2000, duration_ms))

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='#121212')
    frames = []
    temp_dir = tempfile.mkdtemp()

    max_r = r_earth + 500000.0
    limit = max_r * 1.15
    gimbal_limit_rad = config.CROSSLINK_GIMBAL_RANGE
    gimbal_limit_deg = np.rad2deg(gimbal_limit_rad)

    try:
        for frame_idx, step_idx in enumerate(frame_indices):
            ax.clear()
            ax.set_facecolor('#121212')
            ax.set_aspect('equal')
            ax.set_xlim([-limit, limit])
            ax.set_ylim([-limit, limit])
            ax.axis('off')  # Hide axis spines and ticks for a radar-screen look

            # Draw Earth as a neon blue translucent circle
            earth_circle = plt.Circle((0, 0), r_earth, facecolor='#102E5C', alpha=0.7, edgecolor='#00E5FF', lw=1.5)
            ax.add_patch(earth_circle)

            # Draw nominal orbit path as a thin neon purple circle
            orbit_circle = plt.Circle((0, 0), max_r, color='#8A2BE2', linestyle=':', fill=False, alpha=0.3, lw=1.2)
            ax.add_patch(orbit_circle)

            # Store positions and velocities for this step
            step_positions = {}
            step_velocities = {}
            for sat_id in sat_ids:
                step_positions[sat_id] = telemetry[sat_id]['r'][step_idx]
                step_velocities[sat_id] = telemetry[sat_id]['v'][step_idx]

            # Draw crosslinks
            for i, sat_id in enumerate(sat_ids):
                pos = step_positions[sat_id]
                vel = step_velocities[sat_id]
                next_sat_id = sat_ids[(i + 1) % len(sat_ids)]
                pos_next = step_positions[next_sat_id]

                link_active = telemetry[sat_id]['leading_link_active'][step_idx]

                if not link_active:
                    color = '#FF3333'  # Crimson red for broken link
                    style = '--'
                    width = 1.0
                    alpha = 0.5
                else:
                    # Calculate current viewing angle to check if near warning threshold
                    rel_pos = pos_next - pos
                    rel_unit = rel_pos / np.linalg.norm(rel_pos)
                    vel_unit = vel / np.linalg.norm(vel)
                    dot_product = rel_unit.dot(vel_unit)
                    viewing_angle = np.rad2deg(np.arccos(np.clip(abs(dot_product), -1.0, 1.0)))
                    
                    # If within 1.5 degrees of breaking, draw as gold warning
                    if (gimbal_limit_deg - viewing_angle) < 1.5:
                        color = '#FFD700'  # Gold warning
                        style = '-'
                        width = 2.0
                        alpha = 0.9
                    else:
                        color = '#39FF14'  # Neon green for stable link
                        style = '-'
                        width = 1.8
                        alpha = 0.7

                ax.plot([pos[0], pos_next[0]], [pos[1], pos_next[1]], 
                        color=color, linestyle=style, lw=width, alpha=alpha)

            # Draw satellite dots and labels
            for sat_id in sat_ids:
                pos = step_positions[sat_id]
                
                # Check if it's the broken/drift satellite (Sat 5)
                if sat_id == 5:
                    # Highlight Sat 5 in neon red
                    ax.scatter(pos[0], pos[1], color='#FF1744', marker='o', s=70, edgecolors='#FFFFFF', lw=1.0, zorder=6)
                else:
                    # Active satellites in cyan
                    ax.scatter(pos[0], pos[1], color='#00E5FF', marker='o', s=35, zorder=5)

                # Shift text label slightly outward radially to prevent overlap
                angle = np.arctan2(pos[1], pos[0])
                label_offset = 1.07
                ax.text(pos[0] * label_offset, pos[1] * label_offset, f"S{sat_id}", 
                        color='#FFFFFF', fontsize=8, ha='center', va='center')

            time_sec = telemetry[sat_ids[0]]['time'][step_idx]
            ax.set_title(f"Constellation Ring Topology Radar\nTime: {time_sec:.1f} s | Green: Active, Gold: Warning, Red: Broken/Sat5", color='#FFFFFF', fontsize=10, pad=15)

            # Save frame
            frame_path = os.path.join(temp_dir, f"frame_{frame_idx:04d}.png")
            plt.savefig(frame_path, dpi=100, bbox_inches='tight', facecolor='#121212')
            img = Image.open(frame_path)
            img.load()
            frames.append(img)

        # Save GIF
        workspace_dir = "/Users/danielludlow/Documents/Constellation-MPC"
        results_dir = os.path.join(workspace_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(results_dir, "ring_orbits.gif")

        if frames:
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration_ms,
                loop=0
            )
            print(f"2D Ring animation saved successfully to {output_path}")

    finally:
        plt.close(fig)
        shutil.rmtree(temp_dir)
        plt.style.use('default')


def print_summary_metrics(simulation):
    """
    Analyzes telemetry data and prints a structured console report
    detailing the MPC success rate, fuel consumption, and constraint tracking.
    """
    telemetry = simulation.telemetry
    if not telemetry:
        print("No telemetry data to summarize.")
        return

    import src.config as config

    sat_ids = sorted(list(telemetry.keys()))
    n_steps = len(telemetry[sat_ids[0]]['time'])
    dt = simulation.dt

    print("\n" + "="*80)
    print("                      MPC PERFORMANCE & SAFETY SUMMARY")
    print("="*80)
    print(f"{'Sat ID':<8} | {'MPC Success Rate':<18} | {'Mean Space Error (km)':<22} | {'Fuel Impulse (N-s)':<18}")
    print("-"*80)

    total_solves = 0
    total_failures = 0
    total_gimbal_violations = 0
    total_safety_violations = 0
    gimbal_limit_deg = np.rad2deg(config.CROSSLINK_GIMBAL_RANGE)
    safety_distance_km = config.SAFETY_DISTANCE / 1000.0

    for sat_id in sat_ids:
        sat_data = telemetry[sat_id]
        
        # Calculate solver stats (only count actual solves, i.e. when step matches MPC update cycle)
        statuses = sat_data['solver_status']
        times = sat_data['time']
        
        # Filter to actual MPC run steps
        actual_solves = 0
        successful_solves = 0
        
        # Find when solver actually executes (every MPC time step)
        last_solve_time = -9999.0
        for step_idx, t in enumerate(times):
            # The controller solves at step 0, and then whenever (t - last_mpc_run_time) >= MPC_TIME_STEP
            if t == 0.0 or (t - last_solve_time) >= config.MPC_TIME_STEP:
                actual_solves += 1
                status = statuses[step_idx]
                if status in ["optimal", "optimal_inaccurate"]:
                    successful_solves += 1
                last_solve_time = t
        
        solver_success_pct = (successful_solves / actual_solves) * 100.0 if actual_solves > 0 else 100.0
        
        total_solves += actual_solves
        total_failures += (actual_solves - successful_solves)

        # Calculate tracking errors
        # To avoid class variable issues, we recreate a mock controller state for slot lookup
        from src.vehicle.controller import Controller
        ctrl = Controller(sat_data['r'][0], sat_data['v'][0])
        
        tracking_errors = []
        for step_idx, t in enumerate(times):
            pos_actual = sat_data['r'][step_idx]
            pos_nom, _ = ctrl.get_nominal_state(t)
            err = np.linalg.norm(pos_actual - pos_nom) / 1000.0  # km
            tracking_errors.append(err)
            
        mean_err_km = np.mean(tracking_errors)

        # Calculate fuel usage
        thrusts = np.array(sat_data['applied_thrust'])
        thrust_mags = np.linalg.norm(thrusts, axis=1)
        fuel_impulse = np.sum(thrust_mags) * dt

        print(f"S{sat_id:<7} | {solver_success_pct:>16.2f}% | {mean_err_km:>20.4f} | {fuel_impulse:>16.2f}")

        # Check safety and gimbal violations over the full telemetry history
        for step_idx in range(n_steps):
            pos = sat_data['r'][step_idx]
            vel = sat_data['v'][step_idx]
            
            # Find leading neighbor
            next_sat_id = sat_ids[(sat_ids.index(sat_id) + 1) % len(sat_ids)]
            pos_next = telemetry[next_sat_id]['r'][step_idx]

            # Spacing check
            dist = np.linalg.norm(pos_next - pos) / 1000.0
            if dist < safety_distance_km:
                total_safety_violations += 1

            # Pointing check
            rel_pos = pos_next - pos
            rel_unit = rel_pos / np.linalg.norm(rel_pos)
            vel_unit = vel / np.linalg.norm(vel)
            dot_product = rel_unit.dot(vel_unit)
            viewing_angle = np.rad2deg(np.arccos(np.clip(abs(dot_product), -1.0, 1.0)))
            if viewing_angle > gimbal_limit_deg:
                total_gimbal_violations += 1

    print("-"*80)
    print("Constellation Fleet Summary:")
    print(f" * Total Optimization Cycles : {total_solves} solves")
    print(f" * Solver Success Rate       : {((total_solves - total_failures)/total_solves)*100.0:.3f}%")
    print(f" * Gimbal Range Violations   : {total_gimbal_violations} steps")
    print(f" * Safety Distance Breaches  : {total_safety_violations} steps")
    print("="*80 + "\n")
