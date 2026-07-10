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
        import src.simulation.config as config
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
            
        # Save to highest folder level (workspace directory)
        output_path = os.path.join("/Users/danielludlow/Documents/Constellation-MPC", "orbits.gif")
        
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
