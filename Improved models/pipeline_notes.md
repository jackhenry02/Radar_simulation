
***

# The Bio-Inspired Spiking Radar: Continuous Regression Pipeline

This document outlines the architecture for a Spiking Neural Network (SNN) designed to estimate the precise distance of a target. Unlike classification models that sort targets into discrete "bins" (e.g., 1-2m), this model acts as a regressor, outputting a continuous value (e.g., `5.42m`) by analyzing the timing dynamics between a transmitted signal and its received echo.

---

## 1. The Waveform: Frequency Modulated Continuous Wave (FMCW)

To make the radar robust to noise, we do not simply transmit raw voltage spikes. We use the spikes to trigger a complex waveform known as a **Linear Chirp**.

### The Mechanism: The Sifting Property
We utilize the **Sifting Property** of convolution. A biological spike can be modeled as a Dirac delta function, $\delta(t)$. If we have a chirp waveform template $c(t)$, the actual transmission $x(t)$ is the convolution of the spike train $s(t)$ and the chirp:

$$x(t) = s(t) * c(t) = \int_{-\infty}^{\infty} s(\tau)c(t-\tau) d\tau$$



**Why this is chosen:**
1.  **Bandwidth:** A chirp spreads energy over a wide frequency range, improving range resolution.
2.  **Biology:** This mimics how a bat's click is a complex structured sound, not just a simple impulse.
3.  **Stochasticity:** If the neuron fires multiple times, multiple overlapping chirps are emitted, creating a dense, high-energy signal driven by a random biological process.

---

## 2. Data Generation: The Physical Channel

We simulate the physics of electromagnetic (or acoustic) wave propagation to generate valid training data.

### The Physics of Delay
The distance $d$ is encoded linearly in time. The **Time-of-Flight (ToF)** is calculated as:

$$t_{delay} = \frac{2 \cdot d}{c}$$

*(Where $c$ is the speed of the wave propagation).*



### Signal Processing Chain
1.  **Transmission:** The chirp moves through the medium.
2.  **Attenuation:** The signal strength drops as it travels based on the Inverse Square Law, meaning distant targets return much quieter echoes ($Amplitude \propto \frac{1}{d}$).
3.  **Noise Injection:** Random Gaussian noise is added to simulate thermal noise in the receiver.
4.  **Matched Filtering:** Upon reception, we mathematically correlate the noisy echo with the original chirp template. This compresses the spread-out energy back into a sharp, narrow peak at the exact moment of arrival.
5.  **Re-Encoding:** A second SNN neuron (the "Encoder") watches this analog peak. When the peak crosses its threshold, it fires a spike, converting the echo back into a `recovered_spikes` train.

---

## 3. Model Initialization: The "Physical Basis"

We do **not** initialize the SNN with random weights. Random initialization forces the network to learn the concept of "time" and "decay" from scratch, which is inefficient and prone to failure. Instead, we use a **Heterogeneous Initialization** strategy.

### The Concept: Population Coding
In biological brains, neurons are not identical. They possess a diverse range of **Time Constants** ($\tau$). Some neurons forget inputs instantly (fast leak), while others hold memory for a long time (slow leak).



### The Implementation
We assign the hidden layer neurons a range of decay rates ($\beta$) linearly spaced from "Fast" to "Slow."

$$\beta_i \in [0.5, 0.99]$$

**Why this works:**
* **Fast neurons ($\beta \approx 0.5$):** Capture events that happen quickly (Short range targets).
* **Slow neurons ($\beta \approx 0.99$):** "Remember" the transmitted pulse long enough to compare it with a late-arriving echo (Long range targets).
* This creates a **Temporal Basis Set** that naturally spans the entire range of possible distances before training even begins.

---

## 4. The Output: Continuous Voltage vs. Spike Counting

To achieve regression (getting a number like `3.45`) rather than classification (getting a class like "Bin 3"), we alter the final neuron's behavior.

### The Non-Spiking Leaky Integrator
We use a standard Leaky Integrate-and-Fire (LIF) neuron for the output, but with **two key modifications**:
1.  **Disable Threshold:** The neuron is not allowed to fire.
2.  **Disable Reset:** The membrane potential is never cleared.

### The "Bucket" Analogy
Imagine the output neuron is a bucket with a small leak. Incoming spikes from the hidden layer are cups of water poured into the bucket. The prediction is the **level of the water** at the end of the simulation.



### The Math
The voltage $V$ at time step $t$ is calculated as:

$$V[t] = \beta V[t-1] + W \cdot S_{input}[t]$$

The final predicted distance is simply the voltage at the final time step $T$:

$$Distance_{pred} = V_{mem}[T]$$

This allows for infinite precision (within floating-point limits), whereas counting spikes would limit us to integer precision.

---

## 5. The Training Process

We train the network to minimize the error between its "voltage guess" and the "real distance."

### Loss Function: Mean Squared Error (MSE)
We compare the continuous output voltage to the ground-truth distance label:

$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} (Distance_{true}^{(i)} - V_{output}^{(i)})^2$$

### The Optimization Problem
Standard neural networks use Backpropagation to calculate gradients (slopes). However, spikes are binary step functions (0 or 1). The derivative of a step function is zero almost everywhere, which kills the learning process.

### The Solution: Surrogate Gradients
During the backward pass (learning), `snntorch` replaces the sharp step function with a smooth **Sigmoid** function.



This allows us to calculate a valid gradient and adjust two things:
1.  **Synaptic Weights ($W$):** How much influence one neuron has on another.
2.  **Decay Rates ($\beta$):** We fine-tune the time constants of the hidden neurons to perfectly match the radar delays.