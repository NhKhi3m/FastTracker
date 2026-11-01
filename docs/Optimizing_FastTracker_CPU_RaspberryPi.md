# Optimizing Kalman-Based Trackers (FastTracker) for CPU and Raspberry Pi

Based on the FastTracker paper (Hashempoor & Hwang, arXiv:2508.14370v5) and general edge-deployment engineering practice.

The tracker itself is already cheap: Kalman prediction, IoU matching, a few geometric operations, and no ReID CNN. On a Raspberry Pi, the **detector (YOLOX) is almost always the bottleneck**, typically 90%+ of per-frame time. The biggest wins therefore come from the detection side and the pipeline design, followed by tracker-level micro-optimizations.

## 1. Detector-side optimizations (biggest impact)

The paper's Table VI already shows graceful degradation: YOLOX-Nano (1M params) reaches 68.3 MOTA versus 79.3 for YOLOX-L.

- **Use a smaller detector**: YOLOX-Nano/Tiny, YOLOv8n, or NanoDet. Pick the smallest that holds your accuracy requirements.
- **Lower input resolution**: 640 to 416 or 320 gives a near-quadratic speedup. Small objects suffer, so test against your scene.
- **Quantization**: INT8 (post-training or quantization-aware) typically gives 2-4x on ARM CPUs. FP16 helps on newer ARM cores (Pi 5 / Cortex-A76).
- **Pruning and distillation**: structured channel pruning, or distill a larger model into a nano one.
- **Optimized inference runtimes**: NCNN (very strong on ARM), ONNX Runtime with XNNPACK, TFLite with XNNPACK, OpenVINO (x86), MNN, or TVM-tuned kernels. These beat naive PyTorch by a wide margin.
- **Cheaper NMS**: reduce the pre-NMS candidate count, raise the score threshold early, and do class-wise NMS only if necessary. The paper uses an NMS threshold of 0.75, so filter candidates before NMS runs.
- **Hardware accelerators** (if "CPU-only" is flexible): Coral Edge TPU, Hailo-8L (Raspberry Pi AI Kit), or Pi NPU add-ons offload the detector entirely and leave the CPU for tracking.

## 2. Pipeline-level optimizations

- **Detect every N frames, predict in between**: run the detector on every 2nd or 3rd frame and let the Kalman filter carry tracks across skipped frames. This suits Kalman trackers because prediction is nearly free. Combine with the occlusion damping logic, and consider adapting N dynamically (detect more often when motion is high or tracks are uncertain).
- **Multithreading / pipelining**: run capture, preprocessing, detection, and tracking in separate threads or processes with queues, so stages overlap. Throughput approaches the slowest stage rather than the sum of all stages.
- **Asynchronous inference**: track frame *t* while the detector processes frame *t+1* (costs one frame of latency but raises FPS).
- **Frame skipping / dropping**: drop stale frames from the queue rather than letting latency build up. Always process the latest frame.
- **Efficient video I/O**: use hardware-accelerated decode (V4L2 / GStreamer / libcamera), reduce camera resolution at the source, avoid unnecessary color conversions, and reuse buffers. On a Pi, decode and resize can cost real time.
- **Region-of-interest cropping**: the paper already uses ROIs for road regions. You can also crop the detector input to the ROI and skip irrelevant image areas, which cuts detector cost directly.
- **Pin threads and set the CPU governor**: use `taskset` or thread affinity, set the governor to `performance`, and make sure cooling prevents thermal throttling (very common on a Pi under sustained load).

## 3. Kalman filter optimizations

The standard constant-velocity Kalman filter uses an 8-dimensional state (x, y, a, h and their velocities). This is small, but with 40+ objects per frame (the FastTrack benchmark averages 43.5), Python overhead dominates.

- **Vectorize / batch across all tracks**: store all track states as a single array (N x 8) and covariances as (N x 8 x 8), then predict and update with batched NumPy operations instead of per-track loops. This is the single most effective tracker-side change (ByteTrack's reference code already does this in its multi-predict).
- **Exploit the sparse structure**: the state transition matrix F is mostly identity with a few velocity coupling terms. Write the predict step explicitly (`x[:4] += x[4:]`) rather than doing a full matrix multiply. Covariance updates can use the block structure too.
- **Use a steady-state / constant-gain Kalman filter**: for constant-velocity models, the Kalman gain converges quickly. Precompute a fixed gain and skip covariance propagation entirely. This is an approximation and can slightly reduce accuracy when noise changes (FastTracker's class-aware parameters may make it less exact), so validate it.
- **Float32 instead of float64**: halves memory bandwidth and speeds up SIMD. Most trackers run fine in float32, but watch covariance numerical stability (use the Joseph form or symmetrize P if needed).
- **Avoid matrix inversion**: solve with Cholesky (`scipy.linalg.cho_solve`) instead of `np.linalg.inv`. For a 4x4 innovation covariance, a closed-form or explicit solve is faster still.
- **Avoid allocations in the hot loop**: preallocate arrays and use in-place operations.
- **Compile the hot path**: Numba (`@njit`), Cython, or a C++/Rust implementation with Eigen removes Python interpreter overhead, often a large fraction of tracker time on a slow CPU. Numba may be awkward on some ARM setups, so Cython or a small C++ extension with pybind11 is a dependable alternative.

## 4. Association optimizations

- **Vectorized IoU**: compute the full IoU matrix with broadcasting rather than nested loops (or use a compiled version such as `cython_bbox`).
- **Gating before matching**: skip pairs that are obviously too far apart (center distance or velocity-based gate). This allows a smaller or sparse cost matrix and speeds up assignment.
- **Spatial indexing**: in dense scenes, a grid or KD-tree restricts IoU to nearby candidates, avoiding the O(N x M) full matrix.
- **Faster assignment solver**: use `lap` / `lapjv` (Jonker-Volgenant) instead of older versions of `scipy.optimize.linear_sum_assignment`; `lapjv` is notably faster for larger matrices. For small matrices the difference is minor.
- **Greedy matching for easy cases**: when high-confidence detections have unambiguous IoU matches, accept them greedily and run the Hungarian solver only on the ambiguous remainder.
- **Connected-component decomposition**: split the cost matrix into independent sub-problems (clusters of overlapping tracks and detections) and solve each separately, since assignment cost grows cubically.
- **Early termination in the two-stage scheme**: skip stage 2 when there are no unmatched tracks or no low-confidence detections.

## 5. FastTracker-specific components

- **Occlusion handling (CP score)**: checking each unmatched tracklet against all active tracklets is O(unmatched x active). Compute it as one vectorized IoU matrix, or restrict it to spatial neighbors via the same grid index.
- **Cone/ROI projection**: the region lookup is a point-in-rectangle query. Precompute a low-resolution label raster of region IDs so lookup is O(1) per track, and precompute the cone angles theta per region, since they depend only on the static edges (Eq. 1). Batch the clamp and projection operations.
- **Avoid `arctan` per track per frame**: compute the direction angle only every few frames, or use cross/dot product comparisons against the cone boundary vectors instead of explicit angles.
- **Skip heavy post-processing on-device**: the paper notes GSP and global linking (with a ResNet50-TP ReID encoder) are optional and offline. On a Pi, leave global linking off or run it offline on a server. GSP (Gaussian process smoothing) is also costly; if you need smoothing, use a cheaper alternative such as linear interpolation or an exponential moving average.
- **Track count control**: cap the maximum number of tracks and delete stale tracks promptly. The paper's T\_occ of 30 frames keeps occluded tracks alive, so reduce it if memory or compute is tight.

## 6. Language and system-level

- **Language choice**: a C++ or Rust tracker core removes interpreter overhead and is often 5-20x faster than pure Python for per-track loops. Keep Python only for orchestration if needed.
- **Reduce logging, drawing, and visualization**: drawing boxes and calling `cv2.imshow` can cost as much as the tracker. Disable them in production or render at a lower rate.
- **64-bit OS**: use 64-bit Raspberry Pi OS for better NEON/SIMD library support.
- **Compiler flags**: build native extensions with `-O3 -march=native` (or the appropriate `-mcpu` for the Pi).
- **Zero-copy between stages**: share buffers between threads and avoid copying frames and arrays.
- **Profile first**: use `cProfile`, `py-spy`, or `perf` to confirm where the time really goes before optimizing. Measure per-stage latency (capture, preprocess, inference, NMS, tracking, output).

## Suggested priority order

1. Profile the pipeline to find the true bottleneck.
2. Switch to a nano detector, quantize to INT8, lower the resolution, and use NCNN or ONNX Runtime with XNNPACK.
3. Detect every 2nd-3rd frame and let the Kalman filter bridge the gaps.
4. Pipeline capture, detection, and tracking across threads.
5. Vectorize the Kalman and IoU code, then compile the hot path (Numba, Cython, or C++).
6. Add gating or spatial indexing for dense scenes.
7. Disable optional post-processing (GSP, G-Link) and visualization on-device.

## Caveat

The paper reports results on an RTX 4060 and gives no CPU or Raspberry Pi timings, so the speedups above are general engineering guidance rather than something validated for FastTracker specifically. Accuracy trade-offs (quantization, frame skipping, steady-state Kalman gain) should be re-validated on your own data, ideally on a vehicle-heavy scene similar to the FastTrack benchmark.

# Part 2: Detailed Explanations

This part expands two items from the overview above: asynchronous inference (section 2) and Kalman filter optimizations (section 3).

## A. Asynchronous inference in detail

### The problem

In a normal (synchronous) loop, every stage waits for the previous one:

```
Frame 1: [capture][detect ........][track]
Frame 2:                                   [capture][detect ........][track]
```

Time per frame = capture + detect + track. If detection takes 80 ms and tracking takes 10 ms, you get about 11 FPS, and the CPU cores sit idle while one stage runs.

### The idea

Run the detector in a separate thread (or process) so it works on frame *t+1* while the tracker is still handling frame *t*:

```
Detector thread: [detect F1 ......][detect F2 ......][detect F3 ......]
Tracker thread:                    [track F1]        [track F2]
```

Throughput is now limited by the slowest stage (about 80 ms, roughly 12.5 FPS), not the sum. Stages overlap instead of queueing up. The cost is that each result comes out about one frame later, so latency goes up slightly while FPS goes up.

### How to implement it

- Connect stages with a queue of size 1 (or 2). If the detector is slower than the camera, the queue fills up.
- When the queue is full, drop the old frame and keep the newest one. Without this, frames pile up and latency grows without bound, which defeats the purpose.
- Use threads if the heavy work happens in code that releases the Python GIL (NCNN, ONNX Runtime, OpenCV, and NumPy all do). Use `multiprocessing` if a lot of pure-Python code runs in each stage.
- Give each thread its own cores if you can (for example, 3 cores to the detector and 1 to the tracker) so they don't fight over cache and cause stalls.

### Combining it with "detect every N frames"

This is where Kalman trackers shine. The detector runs in the background as fast as it can, and the tracker runs on every camera frame using Kalman prediction only. When a detection result arrives, it is a few frames old, so you can either:

1. Match it directly against the current predictions (simple, slightly less accurate), or
2. Keep a short history of track states, match against the state from the frame the detection belongs to, then fast-forward the corrected state to the present (more accurate, more code).

### When it helps most

It helps most when the detector takes much longer than the tracker, which is the usual case on a Pi. It helps little if one stage dominates everything: you can't overlap 80 ms with 1 ms and gain much, but you do gain from overlapping capture and decode with inference.

## B. Kalman filter optimizations in detail

First, what the filter computes for each track on each frame, with state x (8 numbers: x, y, aspect, height, plus their velocities) and covariance P (8x8):

```
Predict:  x = F·x              P = F·P·Fᵀ + Q
Update:   S = H·P·Hᵀ + R       K = P·Hᵀ·S⁻¹
          x = x + K·(z − H·x)  P = (I − K·H)·P
```

Each solution below removes waste from these steps.

### a) Batch all tracks together

Instead of a Python loop that runs these formulas once per track, stack the states into arrays and process every track in one operation:

```python
# slow: 40 tracks -> 40 small Python-level calls
for t in tracks: t.predict()

# fast: one call for all tracks
X = np.stack([t.x for t in tracks])      # (N, 8)
P = np.stack([t.P for t in tracks])      # (N, 8, 8)
X[:, :4] += X[:, 4:]                     # all positions move at once
P = F @ P @ F.T + Q                      # batched matmul
```

On a Pi, the interpreter overhead of 40+ tiny calls per frame costs more than the math itself. This is usually the largest tracker-side gain. FastTracker's class-aware noise and occlusion damping fit this layout: keep per-track Q/R scale factors in an array, and use a boolean mask for tracks flagged `occluded` when you multiply their velocities by the damping factor γ.

### b) Exploit the structure of F and H

F is almost an identity matrix: position += velocity, and everything else stays the same. A full 8x8 matrix multiply is about 512 multiplications for something that is really 4 additions:

```python
x[:4] += x[4:]          # instead of x = F @ x
```

H just picks the first 4 states, so `H·P·Hᵀ` is simply the top-left 4x4 block of P (`P[:4,:4]`). No multiplication is needed.

### c) Steady-state (constant-gain) Kalman filter

For a constant-velocity model, P and the gain K settle to nearly fixed values after a few frames. You can precompute K once and skip all covariance math:

```python
x_pred = predict_step(x)                       # cheap
x = x_pred + K_fixed @ (z - x_pred[:4])        # no P, no S, no inverse
```

This removes most of the per-track cost. The catch is that it assumes the noise doesn't change. FastTracker uses different parameters per class (cars vs. pedestrians), and ByteTrack-style filters scale noise with box height. So precompute one K per class (or per height bucket), and test accuracy before committing.

### d) Use float32 instead of float64

This halves memory traffic, and ARM NEON SIMD processes twice as many numbers per instruction. For an 8-state filter it is safe in most cases. If you see covariance blow up or become non-symmetric, symmetrize it (`P = (P + Pᵀ)/2`) or use the Joseph-form update.

### e) Don't invert matrices

The formula `S⁻¹` suggests inverting a 4x4 matrix, but you should never need the inverse itself. Solve the system instead:

```python
# slower and less stable
K = P_Ht @ np.linalg.inv(S)

# better: S is symmetric positive-definite -> Cholesky
K = scipy.linalg.cho_solve(scipy.linalg.cho_factor(S), P_Ht.T).T
```

For a 4x4 matrix, a hand-written closed-form solve inside compiled code is faster still.

### f) Avoid allocations in the hot loop

Creating new arrays every frame (`np.zeros`, `np.stack`, temporary results) costs time and fragments memory. Preallocate arrays for the maximum number of tracks (say 128), reuse them, and use in-place operations (`np.add(a, b, out=a)`, `+=`). Keep a count of active tracks and work on `X[:n]`.

### g) Compile the hot path

After batching, what remains is a few array operations plus bookkeeping. Compiling it removes the interpreter completely:

- **Numba `@njit`**: easiest, since it is mostly just adding a decorator, though ARM installation can be tricky.
- **Cython or C++ with pybind11 and Eigen**: dependable on a Pi, and you control the memory layout.
- **Rust (PyO3)**: similar speed with safer code.

### Which to do first

Measure your tracker time per frame first. If the tracker takes about 2-5 ms and the detector 80 ms, none of the Kalman tweaks will be noticeable, and detector work and async inference matter far more. If you have 100+ tracks, or the tracker is in pure Python and takes 20+ ms, do (a) batching first, then (b), (d), (f), and compile with (g). Treat (c) as a last resort, because it is the only one that changes the filter's behavior.
