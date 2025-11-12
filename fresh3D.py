import numpy as np
import os
import pandas as pd # Added for CSV output

# --- Configuration Constants ---
# These might need tuning based on your specific AWR1843Boost setup
N_CHIRPS = 128
N_ADC_SAMPLES = 512
N_TX_ANTENNAS = 3
N_RX_ANTENNAS = 4
N_FRAMES_READ_LIMIT = None # Set to a number (e.g., 10) to process only the first few frames for faster testing

# --- Radar Parameters (Crucial for calculations) ---
# Check these against your radar configuration!
START_FREQ_GHZ = 77.0
FREQ_SLOPE_MHZ_PER_US = 62.04 # From your previous code
ADC_SAMPLE_RATE_KSPS = 10000 # From your previous code
CHIRP_IDLE_TIME_US = 7.0 # Example value, check your config
CHIRP_RAMP_END_TIME_US = 60.0 # Example value, check your config

# Calculated parameters
LIGHT_SPEED_MPS = 299792458.0
ADC_SAMPLE_TIME_US = N_ADC_SAMPLES / ADC_SAMPLE_RATE_KSPS # In us
BANDWIDTH_HZ = FREQ_SLOPE_MHZ_PER_US * 1e12 * ADC_SAMPLE_TIME_US * 1e-6 # BW = FreqSlope * ADC_SamplingTime
RANGE_RESOLUTION_M = LIGHT_SPEED_MPS / (2 * BANDWIDTH_HZ)
CENTER_FREQ_HZ = (START_FREQ_GHZ * 1e9) + (BANDWIDTH_HZ / 2)
WAVELENGTH_M = LIGHT_SPEED_MPS / CENTER_FREQ_HZ
CHIRP_DURATION_US = CHIRP_IDLE_TIME_US + CHIRP_RAMP_END_TIME_US
# Doppler resolution depends on the number of chirps and the frame periodicity (or chirp duration if using TDM MIMO)
TOTAL_DOPPLER_TIME_S = N_CHIRPS * CHIRP_DURATION_US * 1e-6 * N_TX_ANTENNAS # Example assuming TDM MIMO
DOPPLER_RESOLUTION_MPS = WAVELENGTH_M / (2 * TOTAL_DOPPLER_TIME_S)

# Angle FFT parameters
N_ANGLE_BINS = 64 # FFT size for angle estimation
ANTENNA_SPACING_LAMBDAS = 0.5 # Assuming half-wavelength spacing for RX antennas
ANTENNA_SPACING_M = ANTENNA_SPACING_LAMBDAS * WAVELENGTH_M

# Peak Detection Threshold (tune this!)
DETECTION_THRESHOLD_DB = 15 # dB above the noise floor (adjust based on results)
MIN_PEAK_DISTANCE = 2 # Minimum separation between peaks in bins (range, doppler, angle)


# --- Core Data Processing Functions (Mostly Unchanged) ---

def frameReshape(frame, chirps, tx, rx, adcsamples):
    """Reshapes the 1D complex ADC data into a 4D frame."""
    expected_elements = chirps * tx * rx * adcsamples
    if frame.shape[0] != expected_elements:
        print(f"Warning: Frame data size mismatch. Expected {expected_elements}, got {frame.shape[0]}. Skipping frame.")
        return None
    try:
        frameWithChirp = np.reshape(frame, (chirps, tx, rx, adcsamples))
        return frameWithChirp.transpose(1, 2, 0, 3) # Output: [tx][rx][chirp][adc]
    except ValueError as e:
        print(f"Error reshaping frame: {e}. Skipping frame.")
        return None


def rangeFFT(reshapedFrame, numADCSamples):
    """Performs a 1D FFT along the ADC sample axis (axis 3)."""
    window = np.hanning(numADCSamples)
    windowedBins1D = reshapedFrame * window
    rangeFFTResult = np.fft.fft(windowedBins1D, n=numADCSamples, axis=3)
    return rangeFFTResult


def process_bin_file(bin_file_name, frame_limit=None):
    """Reads the raw .bin file and parses it into a list of complex 4D frames."""
    n_chirps = N_CHIRPS
    adc_samples = N_ADC_SAMPLES
    tx_antennas = N_TX_ANTENNAS
    rx_antennas = N_RX_ANTENNAS
    bytes_per_sample = 2
    frame_size = n_chirps * adc_samples * tx_antennas * rx_antennas * 2 * bytes_per_sample

    complex_frames = []
    print(f"Reading and processing '{os.path.basename(bin_file_name)}'...")
    frame_count = 0
    with open(bin_file_name, 'rb') as bin_file:
        while True:
            if frame_limit is not None and frame_count >= frame_limit:
                print(f"Reached frame limit ({frame_limit}). Stopping read.")
                break
            adcDataBytes = bin_file.read(frame_size)
            if len(adcDataBytes) < frame_size:
                print(f"Read incomplete frame ({len(adcDataBytes)}/{frame_size} bytes). End of file or error.")
                break

            adcData = np.frombuffer(adcDataBytes, dtype=np.int16)
            adcData = adcData.reshape(-1, 2)
            adcDataComplex = adcData[:, 0] + 1j * adcData[:, 1]

            reshaped_frame = frameReshape(adcDataComplex, n_chirps, tx_antennas, rx_antennas, adc_samples)

            if reshaped_frame is not None:
                complex_frames.append(reshaped_frame)
                frame_count += 1

    print(f"Successfully processed {len(complex_frames)} frames.")
    return complex_frames


def dopplerFFT(rangeFFTOutput, numChirps):
    """Performs a 1D FFT along the chirp axis (axis 2) after windowing."""
    window = np.hanning(numChirps).reshape(1, 1, numChirps, 1)
    windowedChirps = rangeFFTOutput * window
    dopplerFFTResult = np.fft.fft(windowedChirps, n=numChirps, axis=2)
    dopplerFFTResult = np.fft.fftshift(dopplerFFTResult, axes=2)
    return dopplerFFTResult


def angleFFT(dopplerFFTOutput, numAngleBins):
    num_tx = dopplerFFTOutput.shape[0]
    num_rx = dopplerFFTOutput.shape[1]
    num_doppler = dopplerFFTOutput.shape[2]
    num_range = dopplerFFTOutput.shape[3]

    virtual_array_data = np.zeros((num_tx * num_rx, num_doppler, num_range), dtype=complex)

    for tx_idx in range(num_tx):
         for rx_idx in range(num_rx):
             virtual_ant_idx = tx_idx * num_rx + rx_idx
             virtual_array_data[virtual_ant_idx, :, :] = dopplerFFTOutput[tx_idx, rx_idx, :, :]

    num_virtual_antennas = virtual_array_data.shape[0]
    if numAngleBins > num_virtual_antennas:
         pad_width = numAngleBins - num_virtual_antennas
         padded_data = np.pad(virtual_array_data, ((0, pad_width), (0, 0), (0,0)), mode='constant', constant_values=0)
    else:
         padded_data = virtual_array_data

    window = np.hanning(numAngleBins).reshape(-1, 1, 1)
    windowedAntennas = padded_data * window

    angleFFTResult = np.fft.fft(windowedAntennas, axis=0)
    angleFFTResult = np.fft.fftshift(angleFFTResult, axes=0)
    return angleFFTResult


def detect_peaks_3d(data_cube_db, threshold_db, min_distance=1):
    detected_points_indices = []
    noise_floor = np.percentile(data_cube_db, 75)
    effective_threshold = noise_floor + threshold_db
    potential_peak_indices = np.argwhere(data_cube_db > effective_threshold)
    if potential_peak_indices.shape[0] == 0:
        return []

    valid_peaks = []
    for idx_tuple in potential_peak_indices:
        a_idx, d_idx, r_idx = idx_tuple
        local_max = True
        neighborhood_val = data_cube_db[a_idx, d_idx, r_idx]
        for da in [-1, 0, 1]:
            for dd in [-1, 0, 1]:
                for dr in [-1, 0, 1]:
                    if da == 0 and dd == 0 and dr == 0:
                        continue
                    na, nd, nr = a_idx + da, d_idx + dd, r_idx + dr
                    if 0 <= na < data_cube_db.shape[0] and \
                       0 <= nd < data_cube_db.shape[1] and \
                       0 <= nr < data_cube_db.shape[2]:
                        if data_cube_db[na, nd, nr] > neighborhood_val:
                            local_max = False
                            break
                if not local_max: break
            if not local_max: break

        if local_max:
             is_far_enough = True
             for vp_idx, _ in valid_peaks:
                 dist = np.linalg.norm(np.array(idx_tuple) - np.array(vp_idx))
                 if dist < min_distance:
                     is_far_enough = False
                     break
             if is_far_enough:
                 valid_peaks.append((idx_tuple, neighborhood_val))

    detected_points_indices = [item[0] for item in valid_peaks]
    return detected_points_indices


def calculate_point_properties(peak_indices, range_res, doppler_res, num_angle_bins, num_doppler_bins, data_cube_db):
    points_data = []
    angle_center_idx = num_angle_bins // 2
    doppler_center_idx = num_doppler_bins // 2

    for indices in peak_indices:
        angle_idx, doppler_idx, range_idx = indices
        peak_val_db = data_cube_db[angle_idx, doppler_idx, range_idx]
        range_m = range_idx * range_res
        velocity_mps = (doppler_idx - doppler_center_idx) * doppler_res
        angle_sin = (angle_idx - angle_center_idx) * WAVELENGTH_M / (num_angle_bins * ANTENNA_SPACING_M)
        angle_sin = np.clip(angle_sin, -1.0, 1.0)
        angle_rad = np.arcsin(angle_sin)
        angle_deg = np.degrees(angle_rad)
        x, y, z = spherical_to_cartesian(range_m, angle_deg)
        points_data.append({
            'range_bin': range_idx,
            'doppler_bin': doppler_idx,
            'angle_bin': angle_idx,
            'range_m': range_m,
            'velocity_mps': velocity_mps,
            'angle_deg': angle_deg,
            'x': x,
            'y': y,
            'z': z,
            'peak_val_db': peak_val_db
        })
    return points_data


def spherical_to_cartesian(range_m, angle_deg_azimuth):
    angle_rad = np.radians(angle_deg_azimuth)
    x = range_m * np.sin(angle_rad)
    y = range_m * np.cos(angle_rad)
    z = 0.0
    return x, y, z


def process_frame(frame_idx, frame):
    """Process a single frame and return list of detected points (with frame index added)."""
    try:
        # 1. Range FFT
        range_fft_out = rangeFFT(frame, N_ADC_SAMPLES)

        # 2. Doppler FFT
        doppler_fft_out = dopplerFFT(range_fft_out, N_CHIRPS)

        # 3. Angle FFT
        angle_fft_out = angleFFT(doppler_fft_out, N_ANGLE_BINS)

        # 4. Peak Detection
        data_cube_mag = np.abs(angle_fft_out)**2
        data_cube_db = 10 * np.log10(data_cube_mag + 1e-10)

        detected_indices = detect_peaks_3d(data_cube_db, DETECTION_THRESHOLD_DB, min_distance=MIN_PEAK_DISTANCE)

        # 5. Calculate Properties
        num_doppler_bins = angle_fft_out.shape[1]
        points_in_frame = calculate_point_properties(
            detected_indices,
            RANGE_RESOLUTION_M,
            DOPPLER_RESOLUTION_MPS,
            N_ANGLE_BINS,
            num_doppler_bins,
            data_cube_db
        )

        for p in points_in_frame:
            p['frame'] = frame_idx

        return points_in_frame
    except Exception as e:
        print(f"Error processing frame {frame_idx}: {e}")
        return []

# --- Main Execution Block ---

if __name__ == "__main__":
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.path.abspath(os.getcwd())

    experiments_to_run = [
        ("metal_left_to_right_3m.bin", "metal_left_to_right_3m_points.csv")
    ]

    for bin_filename, output_csv_filename in experiments_to_run:
        BIN_FILE = os.path.join(script_dir, bin_filename)
        CSV_FILE = os.path.join(script_dir, output_csv_filename)

        if not os.path.exists(BIN_FILE):
            print(f"*** WARNING: File not found {BIN_FILE}, skipping. ***")
            continue

        print(f"\n" + "="*50)
        print(f"--- STARTING POINT CLOUD GENERATION FOR: {bin_filename} ---")
        print("="*50)

        frames_data = process_bin_file(BIN_FILE, frame_limit=N_FRAMES_READ_LIMIT)
        if not frames_data:
            print(f"*** No frames processed for {bin_filename}. Skipping. ***")
            continue

        print(f"Processing {len(frames_data)} frames...")

        all_points_collected = []

        # Sequential processing
        for idx in range(len(frames_data)):
            pts = process_frame(idx, frames_data[idx])
            all_points_collected.extend(pts)
            print(f"  Frame {idx}/{len(frames_data)-1}: Detected {len(pts)} points.")

        # --- Save points to CSV ---
        if all_points_collected:
            point_cloud_df = pd.DataFrame(all_points_collected)
            cols_order = ['frame', 'x', 'y', 'z', 'velocity_mps', 'range_m', 'angle_deg', 'peak_val_db',
                          'range_bin', 'doppler_bin', 'angle_bin']
            # Keep only existing columns (safety)
            cols_order = [c for c in cols_order if c in point_cloud_df.columns]
            point_cloud_df = point_cloud_df[cols_order]
            point_cloud_df.to_csv(CSV_FILE, index=False)
            print(f"--- Point cloud data saved to: {CSV_FILE} ---")
        else:
            print(f"--- No points detected for {bin_filename}. No CSV file saved. ---")

    print("\n--- All files processed. ---")
