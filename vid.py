import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import os
import glob # Import glob to find files
import numpy as np

# --- Configuration ---
FRAME_INTERVAL_MS = 100  # Speed of the animation (milliseconds per frame)
MARKER_SIZE = 10         # Size of the points in the plot


def create_single_visualization(csv_path, video_path):
    """
    Processes a single .csv file and saves the results to a .mp4 video.
    """
    csv_filename = os.path.basename(csv_path)
    print(f"  --- Processing {csv_filename} for animation ---")

    # Load the point cloud data
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"    Error reading CSV {csv_path}: {e}. Skipping.")
        return

    if 'frame' not in df.columns or df.empty:
        print(f"    CSV file {csv_path} is empty or missing 'frame' column. Skipping.")
        return

    print(f"    Total rows loaded: {len(df)}")

    # --- OPTIMIZATION 1: Pre-group data by frame ---
    print("    Pre-processing frames for fast lookup...")
    frames_dict = {}
    for frame_num, frame_data in df.groupby('frame', sort=True):
        frames_dict[frame_num] = {
            'x': frame_data['x'].values,
            'y': frame_data['y'].values,
            'z': frame_data['z'].values,
            'velocity': frame_data['velocity_mps'].values
        }
    
    total_frames = len(frames_dict)
    frame_numbers = sorted(frames_dict.keys())
    
    if total_frames == 0:
        print("    No frames found in CSV to animate. Skipping.")
        return
        
    print(f"    Pre-processing complete. Total frames: {total_frames}")

    # --- Set up the 3D Plot ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Find data ranges for consistent plot limits
    x_min, x_max = df['x'].min() - 0.5, df['x'].max() + 0.5
    y_min, y_max = df['y'].min() - 0.5, df['y'].max() + 0.5
    z_min, z_max = df['z'].min() - 0.5, df['z'].max() + 0.5
    vel_min, vel_max = df['velocity_mps'].min(), df['velocity_mps'].max()
    # Handle case where vel_min == vel_max
    if vel_min == vel_max:
        vel_min -= 1
        vel_max += 1

    # Initialize the scatter plot (empty at first)
    scatter = ax.scatter([], [], [], s=MARKER_SIZE, cmap='coolwarm', 
                        vmin=vel_min, vmax=vel_max)

    # Set plot labels and limits
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m) - Range')
    ax.set_zlabel('Z (m)')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_title(f'Point Cloud Animation: {csv_filename}')

    # Add color bar for velocity
    cbar = fig.colorbar(scatter, ax=ax, label='Velocity (m/s)')

    # --- OPTIMIZATION 2: Efficient update function with pre-grouped data ---
    def update(frame_idx):
        """Update function optimized for blitting"""
        frame_num = frame_numbers[frame_idx]
        frame_data = frames_dict[frame_num]
        
        # Update point positions and colors using pre-extracted numpy arrays
        scatter._offsets3d = (frame_data['x'], frame_data['y'], frame_data['z'])
        scatter.set_array(frame_data['velocity'])
        
        return (scatter,)

    # --- Create and Save Animation ---
    print("    Creating animation...")
    ani = animation.FuncAnimation(
        fig, 
        update, 
        frames=total_frames,
        interval=FRAME_INTERVAL_MS, 
        blit=True,  # OPTIMIZATION 3: Efficient blitting
        repeat=False # No need to repeat for a saved file
    )

    try:
        print(f"    Saving animation to {os.path.basename(video_path)}...")
        
        # --- OPTIMIZATION 4: Fast FFmpeg settings ---
        from matplotlib.animation import FFMpegWriter
        
        writer = FFMpegWriter(
            fps=1000/FRAME_INTERVAL_MS,
            metadata=dict(artist='Point Cloud Visualizer'),
            bitrate=2000,
            codec='libx264',
            extra_args=[
                '-preset', 'fast',      # Fast encoding (medium, slow for better quality)
                '-crf', '23',           # Constant quality (lower = better, 18-28 range)
                '-pix_fmt', 'yuv420p'   # Compatibility format
            ]
        )
        
        ani.save(video_path, writer=writer)
        print(f"    ✓ Save complete: {os.path.basename(video_path)}")
        
    except Exception as e:
        print(f"    Error saving video: {e}")
        # print("Attempting to show plot instead (close plot window to continue)...")
        # plt.show()
    finally:
        plt.close(fig) # Ensure the figure is closed
    
    # Clear memory
    del df, frames_dict, ani, fig, ax


def run_3d_making(csv_dir, vid_dir):
    """
    Finds all .csv files in csv_dir, processes them,
    and saves the resulting .mp4 videos in vid_dir.
    """
    
    # 1. --- Find all .csv files ---
    csv_files = glob.glob(os.path.join(csv_dir, "*_points.csv"))
    
    if not csv_files:
        print(f"No '*_points.csv' files found in directory: {csv_dir}")
        return
        
    print(f"Found {len(csv_files)} .csv files to process.")
    
    # --- Process each .csv file ---
    for csv_path in csv_files:
        # Get the base name of the csv file (e.g., "circular_object_center_points.csv")
        base_name = os.path.basename(csv_path)
        # Remove the .csv extension (e.g., "circular_object_center_points")
        file_name_no_ext = os.path.splitext(base_name)[0]
        # Create the output video file name
        output_video_name = f"{file_name_no_ext}_animation.mp4"
        # Create the full output path
        output_video_path = os.path.join(vid_dir, output_video_name)
        
        create_single_visualization(csv_path, output_video_path)


if __name__ == "__main__":
    """
    This block allows the script to be run directly.
    It will use the default directory structure relative to this script.
    """
    print("Running maker_3D.py as standalone script...")
    
    # 1. --- Define your default paths ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv_dir = os.path.join(script_dir, "csv")
    default_vid_dir = os.path.join(script_dir, "vid")
    
    # 2. --- Create vid directory if it doesn't exist ---
    os.makedirs(default_vid_dir, exist_ok=True)
    
    # 3. --- Run the main processing function ---
    if not os.path.exists(default_csv_dir):
        print(f"*** ERROR: CSV directory not found at {default_csv_dir} ***")
    else:
        run_3d_making(default_csv_dir, default_vid_dir)