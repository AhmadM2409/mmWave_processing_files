import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import multiprocessing
from functools import partial
import glob 

USE_3D_PEAKS = False  # set False to go back to 2D CFAR-only detection (BUGGED)


def frameReshape(frame,chirps,tx,rx,adcsamples): #
    frameWithChirp = np.reshape(frame,(chirps,tx,rx,adcsamples))
    return frameWithChirp.transpose(1,2,0,3)

def rangeFFT(reshapedFrame,numADCSamples): #
    windowedBins1D = reshapedFrame*np.hamming(numADCSamples)
    rangeFFTResult=np.fft.fft(windowedBins1D)
    return rangeFFTResult

def range_resolution_compute(num_adc_samples, dig_out_sample_rate=2500, freq_slope_const=60.012):
    """ Calculate the range resolution for the given radar configuration

    Args:
        num_adc_samples (int): The number of given ADC samples in a chirp
        dig_out_sample_rate (int): The ADC sample rate
        freq_slope_const (float): The slope of the freq increase in each chirp

    Returns:
        tuple [float, float]:
            range_resolution (float): The range resolution for this bin
            band_width (float): The bandwidth of the radar chirp config
    """
    light_speed_meter_per_sec = 299792458
    freq_slope_m_hz_per_usec = freq_slope_const
    adc_sample_period_usec = 1000.0 / dig_out_sample_rate * num_adc_samples
    band_width = freq_slope_m_hz_per_usec * adc_sample_period_usec * 1e6
    range_resolution = light_speed_meter_per_sec / (2.0 * band_width)

    return range_resolution, band_width

def process_bin_file(bin_file_name):
    n_chirps=128
    adc_samples=512
    tx_antennas=3
    rx_antennas=4
    bytes_per_sample=4
    frame_size=n_chirps*adc_samples*tx_antennas*rx_antennas*bytes_per_sample 
    complex_frames=[]
    save_images = False
    img_num = 1
    with open(bin_file_name,'rb') as bin_file:
        while True:
            adcData=bin_file.read(frame_size)
            if adcData==b'':
                 break
            adcData=np.frombuffer(adcData,dtype=np.uint16)
            adcDataComplex=np.zeros(adcData.shape[0]//2,dtype=complex)
            adcDataComplex[0::2]=adcData[0::4] + 1j * adcData[2::4]
            adcDataComplex[1::2]=adcData[1::4] + 1j * adcData[3::4]
            try:
                complex_frames.append(frameReshape(adcDataComplex,n_chirps,tx_antennas,rx_antennas,adc_samples))
            except:
                 print("Reached the end")
                 break
    range_resolution,bandwidth=range_resolution_compute(512,10000,62.04)
    return complex_frames


#   NEW STUFF FROM C


def dopplerFFT_minimal(rangeFFTOutput, numChirps, N_DOPPLER_BINS, window_type='hanning', static_clutter_enabled=False):
    """
    Minimal, vectorized Doppler FFT function to add.
    Takes full Range FFT output cube and returns complex Doppler FFT cube.

    Args:
        rangeFFTOutput (np.array): Output from rangeFFT, shape [tx, rx, chirp, range_bin].
        numChirps (int): Number of chirps used (length of axis 2).
        N_DOPPLER_BINS (int): Desired Doppler FFT size (from sensor config).
        window_type (str): 'hanning', 'hamming', 'blackman', or None.
        static_clutter_enabled (bool): If True, remove static clutter.

    Returns:
        np.array: Doppler FFT result, shape [tx, rx, N_DOPPLER_BINS, range_bin].
    """
    # Input shape: [tx, rx, chirp, range_bin]

    # --- Static Clutter Removal (Optional) ---
    if static_clutter_enabled:
        mean_signal = np.mean(rangeFFTOutput, axis=2, keepdims=True)
        rangeFFTOutput = rangeFFTOutput - mean_signal

    # --- Windowing ---
    if window_type == 'hanning':
        window = np.hanning(numChirps)
    elif window_type == 'hamming':
        window = np.hamming(numChirps)
    elif window_type == 'blackman':
        window = np.blackman(numChirps)
    elif window_type is None:
         window = np.ones(numChirps) # No window
    else:
        raise ValueError("Unsupported window_type")

    # Reshape window for broadcasting: [1, 1, numChirps, 1]
    window = window.reshape(1, 1, numChirps, 1)
    windowedChirps = rangeFFTOutput * window

    # --- Doppler FFT ---
    # Perform FFT along the chirp axis (axis 2), pad/truncate to N_DOPPLER_BINS
    dopplerFFTResult = np.fft.fft(windowedChirps, n=N_DOPPLER_BINS, axis=2)

    # --- FFT Shift ---
    dopplerFFTResult = np.fft.fftshift(dopplerFFTResult, axes=2)

    # Output shape: [tx, rx, N_DOPPLER_BINS, range_bin]
    return dopplerFFTResult


def angleFFT_minimal(dopplerFFTOutput, # Input: shape [tx, rx, doppler_bin, range_bin]
                     N_ANGLE_BINS, # Desired FFT size from sensor config
                     num_tx, num_rx, # Antenna counts
                     apply_doppler_comp=True, # Flag to enable/disable
                     apply_rx_comp=False, # Flag to enable/disable
                     rx_comp_coeffs=None): # Array shape [num_tx*num_rx] if apply_rx_comp is True
    """
    Minimal, vectorized Angle FFT function to add. Includes optional compensations.

    Args:
        dopplerFFTOutput (np.array): Output from dopplerFFT_minimal.
        N_ANGLE_BINS (int): Desired Angle FFT size (e.g., 64).
        num_tx (int): Number of Tx antennas.
        num_rx (int): Number of Rx antennas.
        apply_doppler_comp (bool): Apply Doppler phase correction.
        apply_rx_comp (bool): Apply Rx channel phase/gain correction.
        rx_comp_coeffs (np.array): Complex coefficients if apply_rx_comp is True.

    Returns:
        np.array: Angle FFT result, shape [N_ANGLE_BINS, doppler_bin, range_bin].
    """
    # Input shape: [tx, rx, doppler_bin, range_bin]
    N_DOPPLER_BINS = dopplerFFTOutput.shape[2]
    num_range_bins = dopplerFFTOutput.shape[3]
    num_virtual_ant = num_tx * num_rx

    compensated_input = dopplerFFTOutput.copy()

    # --- Doppler Phase Compensation (Optional) ---
    if apply_doppler_comp and num_tx > 1:
        # Simplified formula - Verify/Refine using C code's AoAProcDSP_dopplerCompensation logic if needed
        compensation_phase_unit = 2j * np.pi / (N_DOPPLER_BINS * num_tx)
        doppler_indices_signed = np.fft.fftshift(np.arange(N_DOPPLER_BINS)) - N_DOPPLER_BINS // 2
        tx_indices_comp = np.arange(num_tx)[:, np.newaxis]
        compensation_phases = compensation_phase_unit * doppler_indices_signed
        phase_comp_factors = np.exp(compensation_phases * tx_indices_comp)
        # Reshape for broadcasting [tx, 1, doppler, 1]
        phase_comp_factors = phase_comp_factors[:, np.newaxis, :, np.newaxis]
        compensated_input = compensated_input * phase_comp_factors

    # --- Reshape/Combine Antennas ---
    # Convert [tx, rx, doppler, range] -> [tx*rx, doppler, range]
    # Check C code's virtual antenna order if specific layout matters
    virtual_array_data = compensated_input.transpose(0, 1, 2, 3).reshape(num_virtual_ant, N_DOPPLER_BINS, num_range_bins)

    # --- Rx Channel Compensation (Optional) ---
    if apply_rx_comp and rx_comp_coeffs is not None:
         if len(rx_comp_coeffs) >= num_virtual_ant:
             # Reshape coeffs [tx*rx] -> [tx*rx, 1, 1] for broadcasting
             rx_comp_reshaped = rx_comp_coeffs[:num_virtual_ant].reshape(num_virtual_ant, 1, 1)
             virtual_array_data = virtual_array_data * rx_comp_reshaped
         else:
             print("Warning: Insufficient Rx compensation coefficients provided.")

    # --- Padding / Truncation ---
    if N_ANGLE_BINS > num_virtual_ant:
         pad_width = N_ANGLE_BINS - num_virtual_ant
         # Pad along the virtual antenna axis (axis 0)
         padded_data = np.pad(virtual_array_data, ((0, pad_width), (0, 0), (0,0)), mode='constant', constant_values=0)
    elif N_ANGLE_BINS < num_virtual_ant:
        padded_data = virtual_array_data[:N_ANGLE_BINS, :, :] # Truncate
    else:
         padded_data = virtual_array_data

    # --- Angle FFT (No window applied, matches C)---
    angleFFTResult = np.fft.fft(padded_data, n=N_ANGLE_BINS, axis=0)

    # --- FFT Shift ---
    angleFFTResultShifted = np.fft.fftshift(angleFFTResult, axes=0)

    # Output shape: [N_ANGLE_BINS, doppler_bin, range_bin]
    return angleFFTResultShifted


# --- Helper 1D CFAR ---
def cfar_ca_1d_minimal(data_db, guard_len, noise_len_half, threshold_db, is_cyclic, noise_floor_db):
    """Minimal 1D CFAR-CA working on dB data."""
    num_cells = len(data_db)
    detected_indices = []
    thresholds = np.full_like(data_db, noise_floor_db + threshold_db)
    window_start_offset = guard_len + 1
    window_end_offset = guard_len + noise_len_half

    if num_cells < 2 * (guard_len + noise_len_half) + 1:
        return np.array([], dtype=int) # Too small for window

    for i in range(num_cells):
        noise_sum = 0.0
        noise_count = 0

        # Left window
        start = i - window_end_offset
        end = i - window_start_offset + 1
        indices = np.arange(start, end)
        if is_cyclic: indices = indices % num_cells
        valid_indices = indices[(indices >= 0) & (indices < num_cells)]
        if len(valid_indices) > 0:
            noise_sum += np.sum(data_db[valid_indices])
            noise_count += len(valid_indices)

        # Right window
        start = i + window_start_offset
        end = i + window_end_offset + 1
        indices = np.arange(start, end)
        if is_cyclic: indices = indices % num_cells
        valid_indices = indices[(indices >= 0) & (indices < num_cells)]
        if len(valid_indices) > 0:
            noise_sum += np.sum(data_db[valid_indices])
            noise_count += len(valid_indices)

        if noise_count > 0:
            noise_avg_db = noise_sum / noise_count
            noise_est_db = max(noise_avg_db, noise_floor_db) # Apply floor
            thresholds[i] = noise_est_db + threshold_db
            if data_db[i] > thresholds[i]:
                detected_indices.append(i)

    return np.array(detected_indices, dtype=int)

# --- Main 2D CFAR ---
def cfar_ca_2d_minimal(range_doppler_map_db, # Input [range, doppler] in dB
                       guard_len_range, noise_len_range_half, threshold_db_range,
                       guard_len_doppler, noise_len_doppler_half, threshold_db_doppler,
                       cyclic_doppler=True, noise_floor_db=10.0):
    """Minimal two-pass CFAR-CA detection on dB Range-Doppler map."""
    num_range_bins, num_doppler_bins = range_doppler_map_db.shape
    # Bitmask size: Round up ((num_range * num_doppler) / 32)
    doppler_detections_mask_size = (num_range_bins * num_doppler_bins + 31) // 32
    doppler_detections_mask = np.zeros(doppler_detections_mask_size, dtype=np.uint32)
    final_detections_indices = []

    # --- Pass 1: CFAR along Doppler ---
    for r_idx in range(num_range_bins):
        doppler_line = range_doppler_map_db[r_idx, :]
        detected_doppler_indices = cfar_ca_1d_minimal(
            doppler_line, guard_len_doppler, noise_len_doppler_half,
            threshold_db_doppler, is_cyclic=cyclic_doppler, noise_floor_db=noise_floor_db
        )
        # Set bits in the mask
        for d_idx in detected_doppler_indices:
             bit_index = r_idx * num_doppler_bins + d_idx
             word = bit_index >> 5
             bit = bit_index & 31
             if word < len(doppler_detections_mask):
                doppler_detections_mask[word] |= (np.uint32(1) << bit)

    # --- Pass 2: CFAR along Range ---
    for d_idx in range(num_doppler_bins):
        # Optimization: Check mask if needed before processing range line
        # Requires helper: check_doppler_mask(...)
        # if not check_doppler_mask(d_idx, ...): continue

        range_line = range_doppler_map_db[:, d_idx]
        detected_range_indices = cfar_ca_1d_minimal(
            range_line, guard_len_range, noise_len_range_half,
            threshold_db_range, is_cyclic=False, noise_floor_db=noise_floor_db
        )

        # Check against Doppler bitmask
        for r_idx in detected_range_indices:
            bit_index = r_idx * num_doppler_bins + d_idx
            word = bit_index >> 5
            bit = bit_index & 31
            is_detected_doppler = False
            if word < len(doppler_detections_mask):
                is_detected_doppler = (doppler_detections_mask[word] & (np.uint32(1) << bit)) != 0

            if is_detected_doppler:
                final_detections_indices.append([r_idx, d_idx]) # Store [range, doppler] index

    # --- Peak Grouping (Optional - Skipped in minimal version) ---
    # Simplified grouping could be added here if needed, operating on final_detections_indices

    return np.array(final_detections_indices, dtype=int) # Shape [num_detected, 2]

def detect_peaks_3d(data_cube_db, threshold_db=3.0, min_distance=1):
    """
    3D peak detector on an (angle, doppler, range) cube in dB.
    Returns a list of (angle_idx, doppler_idx, range_idx) tuples.
    """
    # Estimate noise floor and build an effective threshold
    noise_floor = np.percentile(data_cube_db, 75)  # 75th percentile as noise estimate
    effective_threshold = noise_floor + threshold_db

    # Candidates above threshold
    potential_peak_indices = np.argwhere(data_cube_db > effective_threshold)
    valid_peaks = []

    for idx_tuple in potential_peak_indices:
        a_idx, d_idx, r_idx = idx_tuple
        neighborhood_val = data_cube_db[a_idx, d_idx, r_idx]

        # Check 3×3×3 neighbourhood for strict local max
        local_max = True
        for da in (-1, 0, 1):
            if not local_max:
                break
            for dd in (-1, 0, 1):
                if not local_max:
                    break
                for dr in (-1, 0, 1):
                    if da == 0 and dd == 0 and dr == 0:
                        continue
                    na, nd, nr = a_idx + da, d_idx + dd, r_idx + dr
                    if 0 <= na < data_cube_db.shape[0] and \
                       0 <= nd < data_cube_db.shape[1] and \
                       0 <= nr < data_cube_db.shape[2]:
                        if data_cube_db[na, nd, nr] > neighborhood_val:
                            local_max = False
                            break
        if not local_max:
            continue

        # Enforce min_distance between already accepted peaks
        is_far_enough = True
        idx_vec = np.array(idx_tuple, dtype=float)
        for vp_idx, _ in valid_peaks:
            dist = np.linalg.norm(idx_vec - np.array(vp_idx, dtype=float))
            if dist < min_distance:
                is_far_enough = False
                break

        if is_far_enough:
            valid_peaks.append((tuple(idx_tuple), neighborhood_val))

    # Return only index triplets
    return [idx for idx, _ in valid_peaks]


def calculate_point_properties_3d_peaks(
    peak_indices, data_cube_db,
    range_resolution_m, doppler_resolution_mps,
    wavelength_m, antenna_spacing_m,
    range_bias=0.0
):
    """
    Convert 3D peak indices into physical point properties.
    peak_indices: list of (angle_idx, doppler_idx, range_idx)
    data_cube_db: same cube the indices came from (for peak_val_db)
    """
    if len(peak_indices) == 0:
        return []

    num_angle_bins = data_cube_db.shape[0]
    num_doppler_bins = data_cube_db.shape[1]

    angle_center_idx = num_angle_bins // 2
    doppler_center_idx = num_doppler_bins // 2

    points_data = []

    for angle_idx, doppler_idx, range_idx in peak_indices:
        peak_val_db = float(data_cube_db[angle_idx, doppler_idx, range_idx])

        # --- Range ---
        range_m = float(range_idx) * range_resolution_m - range_bias
        if range_m < 0:
            range_m = 0.0

        # --- Velocity ---
        doppler_idx_signed = doppler_idx - doppler_center_idx
        velocity_mps = float(doppler_idx_signed) * doppler_resolution_mps

        # --- Angle (azimuth) ---
        angle_idx_signed = angle_idx - angle_center_idx
        angle_sin_azimuth = float(angle_idx_signed) / float(num_angle_bins) * (wavelength_m / antenna_spacing_m)
        angle_sin_azimuth = np.clip(angle_sin_azimuth, -1.0, 1.0)
        angle_sin_elevation = 0.0  # still placeholder, like your existing code

        # Re-use your existing helper:
        # spherical_to_cartesian_minimal(range_m, angle_sin_azimuth, angle_sin_elevation)
        x, y, z = spherical_to_cartesian_minimal(range_m, angle_sin_azimuth, angle_sin_elevation)
        angle_deg_azimuth = float(np.degrees(np.arcsin(angle_sin_azimuth)))

        points_data.append({
            "range_bin": int(range_idx),
            "doppler_bin": int(doppler_idx),
            "angle_bin": int(angle_idx),
            "range_m": float(range_m),
            "velocity_mps": float(velocity_mps),
            "angle_deg": float(angle_deg_azimuth),
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "peak_val_db": float(peak_val_db),
        })

    return points_data



def spherical_to_cartesian_minimal(range_m, angle_sin_azimuth, angle_sin_elevation=0.0):
    """ Minimal Cartesian coordinate calculation using Pythagorean theorem. """
    x = range_m * angle_sin_azimuth
    z = range_m * angle_sin_elevation

    y_squared = range_m**2 - x**2 - z**2
    if y_squared >= 0:
        y = np.sqrt(y_squared)
    else:
        y = 0.0 # Clamp to 0 if angle components exceed range (e.g., due to noise)
    return x, y, z

def calculate_point_properties_minimal(
    detected_indices_rd, # Input: CFAR result, shape [num_detected, 2] -> (range_idx, doppler_idx)
    angleFFTOutput, # Input: Angle FFT cube, shape [N_ANGLE_BINS, N_DOPPLER_BINS, num_range_bins]
    range_resolution_m, doppler_resolution_mps, # Sensor parameters
    wavelength_m, antenna_spacing_m, # Sensor parameters for angle calc
    range_bias=0.0): # Optional range bias correction
    """
    Minimal function to calculate point properties from 2D CFAR results and Angle FFT cube.

    Args:
        detected_indices_rd (np.array): Output from cfar_ca_2d_minimal.
        angleFFTOutput (np.array): Complex output from angleFFT_minimal.
        ... resolution and sensor parameters ...
        range_bias (float): Range bias correction from sensor config (compRangeBiasAndRxChanPhase).

    Returns:
        list: List of dictionaries, each containing point properties.
    """
    points_data = []
    if detected_indices_rd.shape[0] == 0:
        return points_data

    N_ANGLE_BINS, N_DOPPLER_BINS, num_range_bins = angleFFTOutput.shape
    angle_center_idx = N_ANGLE_BINS // 2
    doppler_center_idx = N_DOPPLER_BINS // 2

    range_indices = detected_indices_rd[:, 0]
    doppler_indices = detected_indices_rd[:, 1]

    # Extract angle profiles for detected Range/Doppler points
    # angleFFTOutput shape [angle, doppler, range]
    angle_profiles_complex = angleFFTOutput[:, doppler_indices, range_indices]
    # Calculate magnitude squared along the angle axis (axis 0)
    angle_profiles_mag_sq = np.abs(angle_profiles_complex)**2

    # Find the peak angle index for each detected point
    angle_peak_indices = np.argmax(angle_profiles_mag_sq, axis=0) # Shape [num_detected]
    # Get the complex value at the peak angle
    complex_val_at_peak = angle_profiles_complex[angle_peak_indices, np.arange(len(angle_peak_indices))]
    # Get the peak magnitude squared
    peak_magnitudes_sq = angle_profiles_mag_sq[angle_peak_indices, np.arange(len(angle_peak_indices))]
    peak_val_db = 10 * np.log10(peak_magnitudes_sq + 1e-10)

    for i in range(len(detected_indices_rd)):
        range_idx = range_indices[i]
        doppler_idx = doppler_indices[i]
        angle_idx = angle_peak_indices[i]

        # --- Range calculation ---
        range_m = range_idx * range_resolution_m
        range_m -= range_bias # Apply bias correction
        if range_m < 0: range_m = 0.0

        # --- Velocity calculation ---
        doppler_idx_signed = doppler_idx - doppler_center_idx
        velocity_mps = float(doppler_idx_signed) * doppler_resolution_mps

        # --- Angle Calculation (Azimuth) ---
        angle_idx_signed = angle_idx - angle_center_idx
        # Formula based on FFT peak index for Uniform Linear Array
        # angle_sin = (angle_idx_signed / N_ANGLE_BINS) * (wavelength / antenna_spacing)
        angle_sin_azimuth = float(angle_idx_signed) / N_ANGLE_BINS * (wavelength_m / antenna_spacing_m)
        angle_sin_azimuth = np.clip(angle_sin_azimuth, -1.0, 1.0) # Ensure valid sine
        angle_deg_azimuth = np.degrees(np.arcsin(angle_sin_azimuth))

        # --- Elevation Calculation (Placeholder - Requires specific antenna geometry/phase logic) ---
        # To match C, would need complex Az/El values and atan2 logic here.
        # Minimal version assumes no elevation array or skips calculation.
        angle_sin_elevation = 0.0
        # --- End Elevation Placeholder ---


        # --- Coordinate Calculation ---
        x, y, z = spherical_to_cartesian_minimal(range_m, angle_sin_azimuth, angle_sin_elevation)

        # --- Store results ---
        # Note: SNR/Noise calculation requires output from CFAR (not implemented in minimal CFAR)
        # Storing peak value in dB instead.
        points_data.append({
            'range_bin': int(range_idx),
            'doppler_bin': int(doppler_idx),
            'angle_bin': int(angle_idx),
            'range_m': float(range_m),
            'velocity_mps': float(velocity_mps),
            'angle_deg': float(angle_deg_azimuth),
            'x': float(x),
            'y': float(y),
            'z': float(z),
            'peak_val_db': float(peak_val_db[i]),
            # 'snr_db': # Requires noise estimate from CFAR
            # 'noise_db': # Requires noise estimate from CFAR
        })

    return points_data



def parseConfigFile(configFileName):
    configParameters = {} # Initialize an empty dictionary to store the configuration parameters
    
    # Read the configuration file
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    
    for i in config:
        
        # Split the line
        splitWords = i.split(" ")
        
        # Get the information about the profile configuration
        if "profileCfg" in splitWords[0]:
            configParameters["startFreq"] = float(splitWords[2])
            configParameters["idleTime"] = float(splitWords[3])
            configParameters["rampEndTime"] = float(splitWords[5])
            configParameters["freqSlopeConst"] = float(splitWords[8])
            configParameters["numAdcSamples"] = int(splitWords[10]) # <--- THIS IS THE FIX
            configParameters["digOutSampleRate"] = float(splitWords[11]) * 1e3 # Convert ksps to sps
            # Store for range_resolution_compute
            configParameters["dig_out_sample_rate"] = float(splitWords[11]) # This is in ksps
            configParameters["freq_slope_const"] = float(splitWords[8])
            
        # Get the information about the frame configuration    
        elif "frameCfg" in splitWords[0]:
            configParameters["chirpStartIdx"] = int(splitWords[1])
            configParameters["chirpEndIdx"] = int(splitWords[2])
            configParameters["numLoops"] = int(splitWords[3])
            
        # Get the information about the channel configuration
        elif "channelCfg" in splitWords[0]:
            rxAntBitmask = int(splitWords[1])
            txAntBitmask = int(splitWords[2])
            
            # Count set bits for Rx antennas
            numRxAnt = 0
            for k in range(4): # 4 possible Rx antennas
                if (rxAntBitmask >> k) & 1:
                    numRxAnt += 1
            configParameters["numRxAnt"] = numRxAnt
            
            # Count set bits for Tx antennas
            numTxAnt = 0
            for k in range(3): # 3 possible Tx antennas
                if (txAntBitmask >> k) & 1:
                    numTxAnt += 1
            configParameters["numTxAnt"] = numTxAnt

        # Get the information about the CFAR configuration (Range)
        elif "cfarCfg -1 0" in i: # Note the "-1 0" for Range
            configParameters["guard_len_range"] = int(splitWords[3])
            configParameters["noise_len_range_half"] = int(splitWords[4]) // 2 # File gives total noise len
            configParameters["threshold_db_range"] = float(splitWords[7])

        # Get the information about the CFAR configuration (Doppler)
        elif "cfarCfg -1 1" in i: # Note the "-1 1" for Doppler
            configParameters["guard_len_doppler"] = int(splitWords[3])
            configParameters["noise_len_range_half"] = int(splitWords[4]) // 2 # File gives total noise len
            configParameters["threshold_db_doppler"] = float(splitWords[7])

        # Get the information about the range bias
        elif "compRangeBiasAndRxChanPhase" in splitWords[0]:
            configParameters["range_bias"] = float(splitWords[1])
            
    # --- Calculate derived parameters ---
    # Total number of chirps in one frame (e.g., (2-0+1) * 128 = 384)
    numChirpsPerFrame = (configParameters["chirpEndIdx"] - configParameters["chirpStartIdx"] + 1) * configParameters["numLoops"]
    configParameters["numChirpsPerFrame"] = numChirpsPerFrame
    
    # Number of chirps per Tx antenna (e.g., 384 total chirps / 3 Tx = 128)
    configParameters["numDopplerBins"] = numChirpsPerFrame // configParameters["numTxAnt"]
    
    light_speed_mps = 299792458
    
    # Calculate Range Resolution using your UNCHANGED function
    range_res_tuple = range_resolution_compute(
        configParameters["numAdcSamples"],
        configParameters["dig_out_sample_rate"], # convert ksps to sps
        configParameters["freq_slope_const"]
    )
    configParameters["rangeResolutionMeters"] = range_res_tuple[0]
    
    # Calculate Doppler Resolution
    chirp_time = (configParameters["idleTime"] + configParameters["rampEndTime"]) * 1e-6 # us to s
    configParameters["dopplerResolutionMps"] = light_speed_mps / (
        2 * configParameters["startFreq"] * 1e9 * chirp_time * configParameters["numChirpsPerFrame"]
    )
    
    # Calculate Wavelength
    configParameters["wavelength_m"] = light_speed_mps / (configParameters["startFreq"] * 1e9)
    configParameters["antenna_spacing_m"] = configParameters["wavelength_m"] / 2.0 # Assuming lambda/2 spacing
    
    return configParameters
   
# ------------------------------------------------------------------


def process_frame(frame_tuple, configParameters):
    """
    Worker function to process a single frame.
    This function will be run in parallel on different CPU cores.
    """
    frame_index, frame = frame_tuple
    
    # Get parameters from config
    numADCSamples = configParameters["numAdcSamples"]
    numTxAnt = configParameters["numTxAnt"]
    numRxAnt = configParameters["numRxAnt"]
    N_DOPPLER_BINS = configParameters["numDopplerBins"]
    numChirps = N_DOPPLER_BINS
    N_ANGLE_BINS = 64
    
    # print(f"Processing frame {frame_index}...") # Now we can track progress
    
    # --- Run the full processing chain ---
        # --- Run the full processing chain ---
    rangeFFT_out = rangeFFT(frame, numADCSamples)

    dopplerFFT_out = dopplerFFT_minimal(
        rangeFFT_out, numChirps, N_DOPPLER_BINS,
        window_type='blackman', static_clutter_enabled=True
    )

    # Range–Doppler map (kept for CFAR / debugging)
    doppler_mag_sq = np.abs(dopplerFFT_out) ** 2
    doppler_sum = np.sum(doppler_mag_sq, axis=(0, 1))
    range_doppler_map_db = 10 * np.log10(doppler_sum + 1e-10)
    range_doppler_map_db_transposed = range_doppler_map_db.T

    # Angle FFT (needed for 3D peaks and for the old CFAR path)
    angleFFT_out = angleFFT_minimal(
        dopplerFFT_out, N_ANGLE_BINS, numTxAnt, numRxAnt,
        apply_doppler_comp=True, apply_rx_comp=False
    )

    frame_points_list = []  # List to hold points for *this frame only*

    if USE_3D_PEAKS:
        # --- 3D peak detection on [angle, doppler, range] cube ---
        angle_mag_sq = np.abs(angleFFT_out) ** 2
        angle_cube_db = 10 * np.log10(angle_mag_sq + 1e-10)

        peak_indices = detect_peaks_3d(
            angle_cube_db,
            threshold_db=3.0,   # relative to 75th percentile noise floor
            min_distance=1
        )

        if len(peak_indices) > 0:
            points_list_this_frame = calculate_point_properties_3d_peaks(
                peak_indices, angle_cube_db,
                configParameters["rangeResolutionMeters"],
                configParameters["dopplerResolutionMps"],
                configParameters["wavelength_m"],
                configParameters["antenna_spacing_m"],
                range_bias=configParameters.get("range_bias", 0.0)
            )

            for point in points_list_this_frame:
                point["frame"] = frame_index
                frame_points_list.append(point)

    else:
        # --- Original 2D CFAR + angle peak path ---
        detected_indices_rd = cfar_ca_2d_minimal(
            range_doppler_map_db_transposed,
            configParameters.get("guard_len_range", 4), # Add .get for safety
            configParameters.get("noise_len_range_half", 8),
            configParameters.get("threshold_db_range", 15.0),
            configParameters.get("guard_len_doppler", 2),
            configParameters.get("noise_len_doppler_half", 4), # Fix key name
            configParameters.get("threshold_db_doppler", 8.0),
            cyclic_doppler=True,
            noise_floor_db=50.0
        )


        if len(detected_indices_rd) > 0:
            points_list_this_frame = calculate_point_properties_minimal(
                detected_indices_rd, angleFFT_out,
                configParameters["rangeResolutionMeters"],
                configParameters["dopplerResolutionMps"],
                configParameters["wavelength_m"],
                configParameters["antenna_spacing_m"],
                range_bias=configParameters.get("range_bias", 0.0)
            )

            for point in points_list_this_frame:
                point["frame"] = frame_index
                frame_points_list.append(point)

    return frame_points_list


def process_single_bin_file(bin_file_name, configParameters, output_csv_file):
    """
    Processes a single .bin file and saves the results to a .csv file.
    """
    
    # 3. --- Read the .bin file ---
    print(f"  Reading all frames from {os.path.basename(bin_file_name)}...")
    try:
        complex_frames = process_bin_file(bin_file_name)
    except Exception as e:
        print(f"    Error: Could not process .bin file. Check file name. {e}")
        print("    This often happens if the .bin file does not match the .cfg file.")
        return # Skip this file
        
    print(f"    Read {len(complex_frames)} total frames.")

    if not complex_frames:
        print("    No frames found, skipping processing.")
        return

    # 4. --- Parallel Processing Setup ---
    
    # Create a list of tuples: [(0, frame_data_0), (1, frame_data_1), ...]
    frames_with_index = list(enumerate(complex_frames))
    
    # 'Freeze' the configParameters argument for our worker function
    # This means the pool only needs to send the frame data
    worker_function = partial(process_frame, configParameters=configParameters)

    print(f"    Starting parallel processing on {multiprocessing.cpu_count()} cores...")
    
    # Create a pool of workers and run the job
    with multiprocessing.Pool() as pool:
        # pool.map runs the worker_function on every item in frames_with_index
        # and returns a list of results (a list of lists of points)
        list_of_lists = pool.map(worker_function, frames_with_index)

    # Flatten the list of lists into a single list of points
    all_points_list = [point for sublist in list_of_lists for point in sublist]

                    
    # 5. --- Save to CSV ---
    if len(all_points_list) > 0:
        print(f"    Processing complete. Total points detected: {len(all_points_list)}")
        
        df = pd.DataFrame(all_points_list)
        
        column_order = [
            'frame', 'x', 'y', 'z', 'velocity_mps', 'range_m', 
            'angle_deg', 'peak_val_db', 'range_bin', 'doppler_bin', 'angle_bin'
        ]
        
        final_columns = [col for col in column_order if col in df.columns]
        df = df[final_columns]
        
        df.to_csv(output_csv_file, index=False)
        print(f"    Successfully saved point cloud data to: {os.path.basename(output_csv_file)}")
        
    else:
        print(f"    Processing complete. No objects were detected in {os.path.basename(bin_file_name)}.")


def run_csv_processing(bin_dir, csv_dir, config_file_path):
    """
    Finds all .bin files in bin_dir, processes them using the config_file,
    and saves the resulting .csv files in csv_dir.
    """
    
    # 1. --- Find all .bin files ---
    # Use glob to find all files ending in .bin in the bin_dir
    bin_files = glob.glob(os.path.join(bin_dir, "*.bin"))
    
    if not bin_files:
        print(f"No .bin files found in directory: {bin_dir}")
        return
        
    print(f"Found {len(bin_files)} .bin files to process.")

    # 2. --- Parse the config file ONCE ---
    print(f"Parsing config file: {os.path.basename(config_file_path)}")
    try:
        configParameters = parseConfigFile(config_file_path)
    except Exception as e:
        print(f"Error: Could not parse config file. Check file name and content. {e}")
        return
        
    # 3. --- Process each .bin file ---
    for bin_file_path in bin_files:
        # Get the base name of the bin file (e.g., "circular_object_center.bin")
        base_name = os.path.basename(bin_file_path)
        # Remove the .bin extension (e.g., "circular_object_center")
        file_name_no_ext = os.path.splitext(base_name)[0]
        # Create the output CSV file name
        output_csv_name = f"{file_name_no_ext}_points.csv"
        # Create the full output path
        output_csv_path = os.path.join(csv_dir, output_csv_name)
        
        print(f"\nProcessing file: {base_name}")
        process_single_bin_file(bin_file_path, configParameters, output_csv_path)


if __name__ == "__main__":
    """
    This block allows the script to be run directly.
    It will use the default directory structure relative to this script.
    """
    # This is crucial for multiprocessing to work correctly on all platforms
    multiprocessing.freeze_support()
    
    print("Running Raw_Csv.py as standalone script...")
    
    # 1. --- Define your default paths ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_bin_dir = os.path.join(script_dir, "bin files")
    default_csv_dir = os.path.join(script_dir, "csv")
    default_config_file = os.path.join(script_dir, '1843RangeDoppler.cfg')
    
    # 2. --- Create CSV directory if it doesn't exist ---
    os.makedirs(default_csv_dir, exist_ok=True)
    
    # 3. --- Run the main processing function ---
    if not os.path.exists(default_config_file):
        print(f"*** ERROR: Config file not found at {default_config_file} ***")
    elif not os.path.exists(default_bin_dir):
        print(f"*** ERROR: Bin directory not found at {default_bin_dir} ***")
    else:
        run_csv_processing(default_bin_dir, default_csv_dir, default_config_file)