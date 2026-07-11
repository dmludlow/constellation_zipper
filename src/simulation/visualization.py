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
