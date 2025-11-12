import main_processing
import vid
import os
import multiprocessing

def main():
    """
    Main function to orchestrate the entire processing pipeline.
    1. Processes .bin files from 'bin files/' into .csv files in 'csv/'.
    2. Processes .csv files from 'csv/' into .mp4 videos in 'vid/'.
    """
    
    # --- Define Directory Paths ---
    # Get the absolute path of the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths relative to the script directory
    bin_dir = os.path.join(script_dir, "bin files")
    csv_dir = os.path.join(script_dir, "csv")
    vid_dir = os.path.join(script_dir, "vid")
    
    # Config file is assumed to be in the root, next to this main.py
    config_file = os.path.join(script_dir, "1843RangeDoppler.cfg")

    # --- 1. Create output directories if they don't exist ---
    print(f"Ensuring output directories exist...")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(vid_dir, exist_ok=True)
    print(f"  - {csv_dir}")
    print(f"  - {vid_dir}")

    # --- 2. Run .bin to .csv processing ---
    print("\n--- Starting Step 1: Processing .bin files to .csv ---")
    if not os.path.exists(config_file):
        print(f"*** ERROR: Config file not found at {config_file} ***")
        print("Cannot proceed with .bin processing.")
    else:
        main_processing.run_csv_processing(bin_dir, csv_dir, config_file)
        print("--- Finished Step 1: .csv processing complete ---")

    # --- 3. Run .csv to .mp4 processing ---
    print("\n--- Starting Step 2: Processing .csv files to .mp4 ---")
    vid.run_3d_making(csv_dir, vid_dir)
    print("--- Finished Step 2: .mp4 processing complete ---")
    
    print("\n--- All tasks finished. ---")

if __name__ == "__main__":
    # This is crucial for multiprocessing to work correctly on all platforms
    multiprocessing.freeze_support() 
    main()