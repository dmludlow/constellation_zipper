import os
import shutil
import tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid GUI display errors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

def see_globe(simulation):
    """
    Generates a 3D animated GIF of the satellites orbiting the Earth based on simulation telemetry.
    The GIF is saved as 'orbits.gif' in the root directory of the workspace.
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
    # Aim for ~100 frames for smooth visualization without excessive file size
    frame_step = max(1, n_steps // 100)
    frame_indices = list(range(0, n_steps, frame_step))
    if len(frame_indices) == 0 or frame_indices[-1] != n_steps - 1:
        frame_indices.append(n_steps - 1)
        
    r_earth = 6378137.0  # Earth equatorial radius
    # Try to import from config, fallback if not possible
    try:
        import src.config as config
        r_earth = config.EARTH_EQUATORIAL_RADIUS
    except Exception:
        pass
        
    # Prepare figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Find max position magnitude to scale axis limits
    max_r = 0.0
    for sat_id in sat_ids:
        pos = telemetry[sat_id]['r']
        if len(pos) > 0:
            max_r = max(max_r, np.max(np.linalg.norm(pos, axis=1)))
    if max_r == 0.0:
        max_r = r_earth + 500000.0  # fallback
    
    limit = max_r * 1.1
    
    frames = []
    temp_dir = tempfile.mkdtemp()
    
    try:
        for frame_idx, step_idx in enumerate(frame_indices):
            ax.clear()
            
            # Set limits and aspect ratio
            ax.set_xlim([-limit, limit])
            ax.set_ylim([-limit, limit])
            ax.set_zlim([-limit, limit])
            ax.set_xlabel("X (ECI) [m]")
            ax.set_ylabel("Y (ECI) [m]")
            ax.set_zlabel("Z (ECI) [m]")
            
            # Draw the Earth
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x_earth = r_earth * np.outer(np.cos(u), np.sin(v))
            y_earth = r_earth * np.outer(np.sin(u), np.sin(v))
            z_earth = r_earth * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x_earth, y_earth, z_earth, color='royalblue', alpha=0.25, edgecolor='cornflowerblue', lw=0.3)
            
            # Draw orbital paths and current satellite positions
            for sat_id in sat_ids:
                r_hist = telemetry[sat_id]['r']
                
                # Draw complete path
                ax.plot(r_hist[:, 0], r_hist[:, 1], r_hist[:, 2], color='gray', linestyle=':', alpha=0.4, lw=1)
                
                # Current position
                curr_pos = r_hist[step_idx]
                ax.scatter(curr_pos[0], curr_pos[1], curr_pos[2], color='red', marker='o', s=30)
                ax.text(curr_pos[0], curr_pos[1], curr_pos[2], f" Sat {sat_id}", color='darkred', fontsize=8)
            
            time_sec = telemetry[sat_ids[0]]['time'][step_idx]
            ax.set_title(f"Constellation Orbit Simulation\nTime: {time_sec:.1f} s")
            
            # Save frame
            frame_path = os.path.join(temp_dir, f"frame_{frame_idx:04d}.png")
            plt.savefig(frame_path, dpi=100, bbox_inches='tight')
            img = Image.open(frame_path)
            img.load()
            frames.append(img)
            
        # Ensure results directory exists and save orbits.gif there
        results_dir = os.path.join("/Users/danielludlow/Documents/Constellation-MPC", "results")
        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(results_dir, "orbits.gif")
        
        # Save the frames as an animated GIF
        if frames:
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=80,  # ms per frame
                loop=0
            )
            print(f"GIF saved successfully to {output_path}")
    finally:
        # Clean up figure and temp directory
        plt.close(fig)
        shutil.rmtree(temp_dir)


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
    Generates static plots of inter-satellite spacing and connection statuses over time,
    saving the figure as 'spacing_metrics.png' in the results folder.
    """
    telemetry = simulation.telemetry
    if not telemetry:
        print("No telemetry data for spacing plots.")
        return

    sat_ids = sorted(list(telemetry.keys()))
    n_steps = len(telemetry[sat_ids[0]]['time'])
    time_sec = np.array(telemetry[sat_ids[0]]['time'])

    # Compute spacing (distance to leading neighbor) and connection status
    spacing_matrix = np.zeros((len(sat_ids), n_steps))
    connection_matrix = np.zeros((len(sat_ids), n_steps))

    for step_idx in range(n_steps):
        for i, sat_id in enumerate(sat_ids):
            pos = telemetry[sat_id]['r'][step_idx]
            next_sat_id = sat_ids[(i + 1) % len(sat_ids)]
            pos_next = telemetry[next_sat_id]['r'][step_idx]

            dist = np.linalg.norm(pos_next - pos)
            spacing_matrix[i, step_idx] = dist / 1000.0  # in km

            # Link status (using leading link active boolean)
            lead_active = telemetry[sat_id]['leading_link_active'][step_idx]
            connection_matrix[i, step_idx] = 1.0 if lead_active else 0.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot 1: Spacing (each satellite's spacing layered clearly)
    for i in range(len(sat_ids)):
        ax1.plot(time_sec, spacing_matrix[i], alpha=0.8, lw=1.2, label="Inter-satellite Spacing" if i == 0 else None)

    ax1.set_ylabel("Along-Track Spacing [km]")
    ax1.set_title("Inter-Satellite Spacing and Connection Status over Time")
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc="upper right")

    # Plot 2: Active Connection Percentage
    active_percentage = np.mean(connection_matrix, axis=0) * 100.0
    ax2.plot(time_sec, active_percentage, color='forestgreen', lw=2, label="Active Links %")
    ax2.set_ylabel("Constellation Connectivity [%]")
    ax2.set_xlabel("Simulation Time [s]")
    ax2.set_ylim([-5, 105])
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc="lower right")

    workspace_dir = "/Users/danielludlow/Documents/Constellation-MPC"
    results_dir = os.path.join(workspace_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, "spacing_metrics.png")
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Spacing plots saved successfully to {plot_path}")


def see_ring_animation(simulation):
    """
    Generates a 2D animated GIF showing the equatorial ring of satellites,
    their spacing, and their crosslink connection statuses (green = active, red = broken) over time.
    The GIF is saved as 'ring_orbits.gif' in the results folder.
    """
    telemetry = simulation.telemetry
    if not telemetry:
        print("No telemetry data for ring animation.")
        return

    sat_ids = sorted(list(telemetry.keys()))
    n_steps = len(telemetry[sat_ids[0]]['time'])

    # Downsample steps to keep file size small (~100 frames)
    frame_step = max(1, n_steps // 100)
    frame_indices = list(range(0, n_steps, frame_step))
    if len(frame_indices) == 0 or frame_indices[-1] != n_steps - 1:
        frame_indices.append(n_steps - 1)

    r_earth = 6378137.0
    try:
        import src.config as config
        r_earth = config.EARTH_EQUATORIAL_RADIUS
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(8, 8))
    frames = []
    temp_dir = tempfile.mkdtemp()

    # Find max position magnitude to scale axis
    max_r = r_earth + 500000.0
    limit = max_r * 1.15

    try:
        for frame_idx, step_idx in enumerate(frame_indices):
            ax.clear()
            ax.set_aspect('equal')
            ax.set_xlim([-limit, limit])
            ax.set_ylim([-limit, limit])
            ax.set_xlabel("X (ECI) [m]")
            ax.set_ylabel("Y (ECI) [m]")

            # Draw Earth as a 2D blue circle
            earth_circle = plt.Circle((0, 0), r_earth, facecolor='royalblue', alpha=0.3, edgecolor='cornflowerblue', lw=1.5)
            ax.add_patch(earth_circle)

            # Draw nominal orbit path as a thin dotted gray circle
            orbit_circle = plt.Circle((0, 0), max_r, color='gray', linestyle=':', fill=False, alpha=0.3, lw=1)
            ax.add_patch(orbit_circle)

            # Store positions for this step to draw links
            step_positions = {}
            for sat_id in sat_ids:
                step_positions[sat_id] = telemetry[sat_id]['r'][step_idx]

            # Draw crosslinks first (so they sit behind the satellite dots)
            for i, sat_id in enumerate(sat_ids):
                pos = step_positions[sat_id]
                next_sat_id = sat_ids[(i + 1) % len(sat_ids)]
                pos_next = step_positions[next_sat_id]

                # Link connection flag
                link_active = telemetry[sat_id]['leading_link_active'][step_idx]

                # Line properties: green solid for connected, red dashed for broken
                color = 'limegreen' if link_active else 'crimson'
                style = '-' if link_active else '--'
                width = 2.0 if link_active else 1.0
                alpha = 0.8 if link_active else 0.4

                ax.plot([pos[0], pos_next[0]], [pos[1], pos_next[1]], 
                        color=color, linestyle=style, lw=width, alpha=alpha)

            # Draw satellite dots and labels
            for sat_id in sat_ids:
                pos = step_positions[sat_id]
                ax.scatter(pos[0], pos[1], color='black', marker='o', s=35, zorder=5)
                # Shift text label slightly outward radially to prevent overlap
                angle = np.arctan2(pos[1], pos[0])
                label_offset = 1.07
                ax.text(pos[0] * label_offset, pos[1] * label_offset, f"S{sat_id}", 
                        color='black', fontsize=8, ha='center', va='center')

            time_sec = telemetry[sat_ids[0]]['time'][step_idx]
            ax.set_title(f"Constellation Ring Topology\nTime: {time_sec:.1f} s (Green = Connected, Red = Broken)")

            # Save frame
            frame_path = os.path.join(temp_dir, f"frame_{frame_idx:04d}.png")
            plt.savefig(frame_path, dpi=100, bbox_inches='tight')
            img = Image.open(frame_path)
            img.load()
            frames.append(img)

        # Ensure results directory exists and save
        workspace_dir = "/Users/danielludlow/Documents/Constellation-MPC"
        results_dir = os.path.join(workspace_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(results_dir, "ring_orbits.gif")

        if frames:
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=80,
                loop=0
            )
            print(f"2D Ring animation saved successfully to {output_path}")

    finally:
        plt.close(fig)
        shutil.rmtree(temp_dir)
