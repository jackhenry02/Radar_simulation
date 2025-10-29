import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows
from scipy.fft import fft, fftshift
from numpy.random import normal

# --- Parameters ---
c = 3e8
fc = 77e9
lambda_c = c/fc

# Chirp parameters
bw = 150e6
T_chirp = 10e-6
k = bw / T_chirp
fs = 2*bw

# Frame parameters
n_chirps = 64
T_frame = T_chirp * n_chirps

# Antenna array
N_rx = 16
d = lambda_c/2

# Targets
target_range = np.array([200, 75, 50])
target_vel   = np.array([20, -15, 10])
target_rcs   = np.array([10, 0.8, 0.5])
target_angle = np.deg2rad([30, -20, 10])
n_targets = len(target_range)

# --- Generate Tx waveform ---
t = np.arange(0, T_chirp, 1/fs)
n_samples = len(t)
tx_phase = 2*np.pi*(fc*t + 0.5*k*t**2)
Tx = np.exp(1j*tx_phase)

# --- Simulate received signals ---
Mix_matrix = np.zeros((n_chirps, n_samples, N_rx), dtype=complex)

for i in range(n_chirps):
    Rx_total = np.zeros((N_rx, n_samples), dtype=complex)
    time_at_chirp_start = i*T_chirp
    
    for j in range(n_targets):
        r0 = target_range[j]
        v0 = target_vel[j]
        theta = target_angle[j]
        
        current_range = r0 + v0*time_at_chirp_start
        tau = 2*current_range/c
        t_rx = t - tau
        rx_phase = 2*np.pi*(fc*t_rx + 0.5*k*t_rx**2)
        Rx_j = np.exp(1j*rx_phase)
        
        for m in range(N_rx):
            Rx_total[m,:] += (target_rcs[j]/current_range**4) * Rx_j * np.exp(-1j*2*np.pi*m*d*np.sin(theta)/lambda_c)
    
    # Add noise
    noise_power = np.mean(np.abs(Rx_total)**2)/10**(10/10)  # 10 dB SNR
    Rx_noisy = Rx_total + np.sqrt(noise_power/2)*(normal(size=Rx_total.shape)+1j*normal(size=Rx_total.shape))
    
    # Mix with Tx
    Mix_matrix[i,:,:] = Tx.conj() * Rx_noisy.T

# --- Plot 1: Raw ADC samples (first antenna) ---
plt.figure()
plt.imshow(np.real(Mix_matrix[:,:,0]), aspect='auto', extent=[0,n_samples,1,n_chirps], origin='lower')
plt.xlabel('ADC Sample Index (Fast Time)')
plt.ylabel('Chirp Index (Slow Time)')
plt.title('Raw ADC Samples (Antenna 1)')
plt.colorbar()
plt.show()

# --- Range FFT ---
n_fft_range = 2**int(np.ceil(np.log2(n_samples)))
range_fft = fft(Mix_matrix, n_fft_range, axis=1)
range_axis = (fs*np.arange(n_fft_range)/n_fft_range)*c/(2*k)

# --- Plot 2: Range FFT ---
range_plot = np.abs(range_fft[:,:n_fft_range//2+1,0])
plt.figure()
plt.imshow(20*np.log10(range_plot), aspect='auto', extent=[0,range_axis[n_fft_range//2],1,n_chirps], origin='lower')
plt.xlabel('Range (m)')
plt.ylabel('Chirp Index')
plt.title('Range FFT (Antenna 1)')
plt.colorbar()
plt.show()

# --- Doppler FFT ---
n_fft_vel = 2**int(np.ceil(np.log2(n_chirps)))
rdm = fftshift(fft(range_fft, n_fft_vel, axis=0), axes=0)
fs_doppler = 1/T_chirp
doppler_freq_axis = np.linspace(-fs_doppler/2, fs_doppler/2, n_fft_vel)
vel_axis = doppler_freq_axis * lambda_c / 2

# --- Plot 3: Range-Doppler Map ---
rdm_plot = np.abs(rdm[:,:n_fft_range//2+1,0])
plt.figure()
plt.imshow(10*np.log10(rdm_plot), aspect='auto', extent=[0,range_axis[n_fft_range//2], vel_axis[0], vel_axis[-1]], origin='lower')
plt.xlabel('Range (m)')
plt.ylabel('Velocity (m/s)')
plt.title('Range-Doppler Map (Antenna 1)')
plt.colorbar()
plt.show()

# --- Angle FFT across antennas ---
n_fft_angle = 128
RDA_fft = fftshift(fft(rdm, n_fft_angle, axis=2), axes=2)
angle_axis = np.arcsin(np.linspace(-1,1,n_fft_angle))*180/np.pi

# --- Plot 4: Range-Angle Map (collapsed over Doppler) ---
angle_map_all = np.max(np.abs(RDA_fft[:,:n_fft_range//2+1,:]), axis=0)
angle_map_dB = 20*np.log10(angle_map_all/np.max(angle_map_all))
angle_map_dB[angle_map_dB < -40] = -40
plt.figure()
plt.imshow(angle_map_dB.T, aspect='auto', extent=[0,range_axis[n_fft_range//2], angle_axis[0], angle_axis[-1]], origin='lower')
plt.xlabel('Range (m)')
plt.ylabel('Angle (deg)')
plt.title('Range-Angle Map (collapsed over Doppler)')
plt.colorbar()
plt.show()

# --- Plot 5: 3D Radar Cube ---
cube_data = np.abs(RDA_fft[:,:n_fft_range//2+1,:])
cube_data /= np.max(cube_data)
mask = cube_data > 0.05

vel_grid, range_grid, angle_grid = np.meshgrid(vel_axis, range_axis[:n_fft_range//2+1], angle_axis, indexing='ij')
x = range_grid[mask]
y = vel_grid[mask]
z = angle_grid[mask]
v = cube_data[mask]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x, y, z, c=v, s=15, cmap='jet')
ax.set_xlabel('Range (m)')
ax.set_ylabel('Velocity (m/s)')
ax.set_zlabel('Angle (deg)')
ax.set_title('Radar Cube Point Cloud (Normalized Amplitude)')
fig.colorbar(sc)
plt.show()
