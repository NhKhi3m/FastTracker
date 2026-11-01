# FastTracker

#### FastTracker is a real-Time and accurate visual tracking framework.

<div align="center">

<!--
[**Hamidreza Hashempoor**](https://hamidreza-hashempoor.github.io/)
-->

<!-- **TMLCN 2025** -->

</div>



<div align="center">
  <img src="./figs/tracker_radar.jpg" alt="Image main" width="30%" style="margin: 1%;">
</div>

FastTracker is a general-purpose multi-object tracking framework designed for complex traffic scenes. FastTracker supports diverse object types—especially vehicles—and maintains identity through heavy occlusion and complex motion. It combines an occlusion-aware re-identification module with road-structure-aware tracklet refinement, using semantic priors like lanes and crosswalks for better trajectory accuracy. _[Hamidreza Hashempoor](https://hamidreza-hashempoor.github.io/),  Yu Dong Hwang_.

##  Updates

| Date | Update |
|------|---------|
| **2025-12-03** |  Class aware motion prediction source code (import `fasttracker_cls.py` and `kalman_filter_cls.py` instead of `fasttracker.py` and `kalman_filter.py`). |
| **2025-10-31** |  Added C++ deployment version (FastTracker_CPP). |
| **2025-10-21** |  Added ROI and Cone handling functions source codes. |


## Resources
| Huggingface Dataset | Paper |
|:-----------------:|:-------:|
|[![dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/Hamidreza-Hashemp/FastTracker-Benchmark)|[![arXiv](https://img.shields.io/badge/arXiv-2508.14370-blue)](https://arxiv.org/abs/2508.14370)




## Benchmark
FastTrack-Benchmark is a high-density multi-object tracking benchmark tailored for complex urban traffic scenes. It features 800K annotations across 12 diverse scenarios with 9 object classes, offering over 5× higher object density than existing benchmarks—making it ideal for evaluating trackers under extreme occlusion, interaction, and scene variety.
The Benchmark is public and available in our [**Huggingface Dataset**](https://huggingface.co/datasets/Hamidreza-Hashemp/FastTracker-Benchmark)


<div align="center">
  <img src="./figs/fasttrack_benchmark.jpg" alt="Image main" width="50%" style="margin: 1%;">
</div>

## Framework

Occlusion-aware tracking strategy framework that detects occluded tracklets based on center-proximity with nearby objects. Once detected, occluded tracklets are marked inactive, their motion is dampened to prevent drift, and their bounding boxes are slightly enlarged to aid re-identification. 

<div align="center">
<img src="./figs/fasttrack_occ_alg.jpg" alt="Occlusion Algorithm" style="width:70%;"/>
</div>

## Tracking performance
### Results on MOT challenge test set
| Dataset    | MOTA | IDF1 | HOTA | FP    | FN     | IDs |
|------------|------|------|------|-------|--------|-----|
| MOT16      | 79.1 | 81.0 | 66.0 | 8785  | 29028  | 290 |
| MOT17      | 81.8 | 82.0 | 66.4 | 26850 | 75162  | 885 |
| MOT20      | 77.9 | 81.0 | 65.7 | 24590 | 89243  | 684 |
| FastTracker| 64.4 | 79.2 | 61.5 | 29730 | 68541  | 251 |

## Installation on the host machine

Steps: Setup the environment
```shell
cd <home>
conda create --name FastTracker python=3.9
conda activate FastTracker
pip3 install -r requirements.txt  # Ignore the errors
python setup.py develop
pip3 install cython
conda install -c conda-forge pycocotools
pip3 install cython_bbox
```



## Data preparation

Download [MOT16](https://motchallenge.net/), [MOT17](https://motchallenge.net/), [MOT20](https://motchallenge.net/), [FastTracker](https://huggingface.co/datasets/Hamidreza-Hashemp/FastTracker-Benchmark) and put them under `./datasets` in the following structure:
```
datasets
   |——————FastTracker
   |        └——————train
   |        └——————test
   |——————MOT16
   |        └——————train
   |        └——————test
   |——————mot
   |        └——————train
   |        └——————test
   └——————MOT20
            └——————train
            └——————test

```

Then, you need to turn the datasets to COCO format and mix different training data:

```shell
cd <home>
python tools\\convert_mot16_to_coco.py
python tools\\convert_mot17_to_coco.py 
python tools\\convert_mot20_to_coco.py
```
(For FastTracker benchmark use  [`convert_to_coco.py`](https://huggingface.co/datasets/Hamidreza-Hashemp/FastTracker-Benchmark/blob/main/convert_to_coco.py)
to make annotations.

## Pretrained Weights

To use FastTracker's default models for MOT17 and MOT20 benchmarks, download the following pretrained weights from the [ByteTrack repository](https://github.com/ifzhang/ByteTrack):

- `bytetrack_x_mot17.pth.tar`  
- `bytetrack_x_mot20.pth.tar`  

Place both files into the `./pretrained/` directory.

For **MOT16 benchmark**, you can use weights trained for MOT17 `bytetrack_x_mot17.pth.tar`. For the **FastTrack benchmark**, which includes multiple object classes beyond pedestrians, you need to retrain the YOLOX for multi-class detection. 

The FastTrack benchmark uses `frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, 1.0` format which is similar to the standard MOT format (except that class is added and x,y,z coordinates are removed), where each `gt/gt.txt` file already provides frame-level detections with object class annotations. To train a detector:

1. You **do not need to extract frames manually**—the frame-wise detections are already specified.
2. Simply **ignore the track IDs** in `gt.txt`, as detector training requires only bounding boxes and class labels.
3. Convert the annotations to COCO-style format using the provided class and bounding box info.
4. Use the ByteTrack training scripts to retrain YOLOX with the new annotations.

This enables FastTracker to detect and track multiple object types effectively in complex traffic environments.


## Tracking

Run FastTracker:

To evaluate **FastTracker** on the **MOT17 benchmark** and **MOT20 benchmark**, simply run the following command:

```bash
bash run_mot17.sh
bash run_mot20.sh
```
For  **MOT16 benchmark** and **FatTracker benchmark**, you can use `bash run_mot17.sh`, but need to change the weight directory, experiment name and experiment file.

### ⚙️ Understanding `run_mot17.sh` and `run_mot20.sh` Configuration:

The `run_mot17.sh` script begins by defining a set of key variables that control the evaluation pipeline:

```bash
EXP_FILE="exps/example/xx"
CKPT="pretrained/bytetrack_x_motxx.pth.tar"
BATCH=1
DEVICES=1
EXPERIMENT_NAME="xx"
OUTPUTS_ROOT="YOLOX_outputs"
CONFIGS_DIR="./configs"
```

Here is the defenition of each variable:

* `EXP_FILE`:
Path to the YOLOX experiment definition file. This file specifies the model architecture, dataset format, and preprocessing.
Example: `exps/example/mot/yolox_x_mix_det.py`.

* `CKPT`:
Path to the pretrained checkpoint file. For `MOT17` or `MOT20`, you must download the appropriate `.pth.tar` file from the ByteTrack repository
 and place it under the `./pretrained/` directory.
Examples: `pretrained/bytetrack_x_mot17.pth.tar` or `pretrained/bytetrack_x_mot20.pth.tar`

* `BATCH`:
The batch size used during inference. Typically set to `1` for tracking tasks.

* `DEVICES`:
Number of GPUs to use. Set to `1` for single-GPU setups. Can be increased for parallel evaluation.

* `EXPERIMENT_NAME`:
A unique name identifying the current experiment.
The outputs (logs, tracking results, etc.) will be saved under `YOLOX_outputs/<EXPERIMENT_NAME>/`.

* `OUTPUTS_ROOT`:
The root directory where experiment outputs are saved.

* `CONFIGS_DIR`:
Directory containing multiple `.json` configuration files. Each config file defines tracking hyperparameters such as thresholds, buffer sizes, and occlusion handling settings.
The script automatically runs tracking once per config file inside this folder.

### 📄 Details of Tracking Configuration Files (./configs/*.json)
Each JSON file inside the `./configs` directory specifies a different set of hyperparameters for tracking evaluation.

* `track_thresh`:
Minimum detection score for initializing or updating a track. Lowering this allows weaker detections to be considered.

* `track_buffer`:
Maximum number of frames a tracklet is kept alive without receiving a matching detection.

* `match_thresh`:
IOU threshold used for associating detections to existing tracklets.

* `min_box_area`:
Minimum area of bounding boxes to be considered for tracking. Useful for filtering out tiny detections.

* `reset_velocity_offset_occ`:
Velocity smoothing offset applied when a tracklet is marked as occluded. Higher values help reduce drift.

* `reset_pos_offset_occ`:
Position smoothing offset for occluded objects. Controls how much the predicted position can shift.

* `enlarge_bbox_occ`:
Factor to enlarge bounding boxes of occluded objects. Helps improve re-identification during reappearance.

* `dampen_motion_occ`:
Dampening factor for the velocity vector of occluded objects. Reduces aggressive forward prediction.

* `active_occ_to_lost_thresh`:
Number of frames an occluded object can remain unmatched before being marked as lost.

* `init_iou_suppress`:
IOU suppression threshold used to avoid initializing duplicate tracks from overlapping detections.

* `ROIs`:
Defines all Regions of Interest (ROIs) in the scene, each described by a polygon of corner points in pixel coordinates.

* `roi_repair_max_gap`:
Maximum number of frames a track can disappear within an ROI before it’s considered broken and needs re-initialization.

* `dir_window_N`:
Sliding window length (in frames) for estimating object motion direction stability.

* `dir_margin_deg`:
Angular tolerance (in degrees) allowed when comparing direction estimates — smaller values enforce stricter direction consistency.


### 📦 Output Structure 
After running each .sh file (e.g., `run_mot17.sh` or `run_mot20.sh`), the full set of tracking results will be saved in:
```shell
./YOLOX_outputs/<EXPERIMENT_NAME>/runX/track_results/
```

* Each `runX` folder (e.g., `run000`, `run001`, ...) corresponds to a different config file in the `./configs` directory.

* Inside each `track_results/` folder, you will find `.txt` result files for each video sequence.

* These result files are already formatted in the standard MOT Challenge format and can be directly submitted to the MOT Challenge evaluation server.

To reproduce the best performance reported in our paper and in MOT Challenge server, you need to tune the hyperparameters for each video sequence individually 
in  [lines 150-170 mot_evaluator.py](https://github.com/Hamidreza-Hashempoor/FastTracker/blob/main/yolox/evaluators/mot_evaluator.py).
This is done by editing the corresponding JSON config file in `./configs/` with sequence-specific values (e.g., `track_thresh`, `match_thresh`, etc.).

## Obtain MOTA /IDS/ HOTA and other evaluation
To evaluate tracking performance using standard metrics such as MOTA, IDF1, HOTA, FP, FN, and ID switches, we use the [TrackEval repository](https://github.com/JonathonLuiten/TrackEval).

### 🧪 Run Evaluation with run_eval.sh
After running your tracking experiments, simply execute:
```bash
run_eval.sh
```
This script will evaluate the results produced in each:
```bash
./YOLOX_outputs/<EXPERIMENT_NAME>/runX/track_results/
```

* Each runX corresponds to a different tracking configuration file used during tracking.

* The results from TrackEval will be saved under the same `track_results/` directory.

* These include `.txt` summary files containing the full set of metrics

### ⚙️ Understanding run_eval.sh Configuration
The `run_eval.sh` script is used to evaluate the tracking results using TrackEval.
It defines several key variables that control what experiment to evaluate, where the predictions are located, and where the ground-truth lives. In this bash file, as example, we have:
```shell
TRACKEVAL_ROOT="${SCRIPT_DIR}/TrackEval"
BENCHMARK="MOT15"
USE_PARALLEL="False"

TRACK_SCRIPT="tools/track.py"
HOTA_PREP="${TRACKEVAL_ROOT}/hotaPreparation.py"
RUN_MOT="${TRACKEVAL_ROOT}/scripts/run_mot_challenge.py"
path_folder="${TRACKEVAL_ROOT}/data/"
EXP_DIR="${OUTPUTS_ROOT}/${EXPERIMENT_NAME}"

GT_ROOT="${SCRIPT_DIR}/gt/MOT17"
RESULTS_DIR="${SCRIPT_DIR}/YOLOX_outputs/yolox_x_mix_det/run002/track_results"
```
* `TRACKEVAL_ROOT`: Path to the local clone of the TrackEval
 repository.

* `BENCHMARK`: The benchmark to evaluate against. Must match the format expected by TrackEval.

* `USE_PARALLEL`: Set to "`True`" to enable parallel evaluation over multiple sequences.

* `TRACK_SCRIPT`: (Optional) Path to your internal tracking script, if reused during eval.

* `HOTA_PREP`: Path to the script that prepares outputs for HOTA evaluation.

* `RUN_MOT`: Path to TrackEval’s main evaluation script (`run_mot_challenge.py`).

* `path_folder`: Internal path for TrackEval data structures.

* `GT_ROOT`: Path to the directory containing ground truth annotations in `MOT` format. Update this path to point to the correct benchmark dataset.

* `RESULTS_DIR`: Directory containing tracking result `.txt` files (in `MOT` format).

## Post-processing with Gaussian Smoothing Interpolation (GSI)
After running your tracking experiments, you can optionally apply Gaussian Smoothing Interpolation to refine the results. We provide a tool for this:

```bash
tools/GSI.py
```
To run GSI on your saved tracking results:
```bash
python tools/GSI.py \
    --loadpath "./YOLOX_outputs/experiment_name/runx/track_results" \
    --savepath "./YOLOX_outputs/experiment_name/runx/track_results_gsi"
```

* `--loadpath`: Path to the directory containing the original tracking result `.txt` files. These are generated after running `run_mot17.sh` or `run_mot20.sh`.

* `--savepath`: Output directory where the smoothed results will be saved.

* The smoothed results will maintain `MOT` format, and can be used directly for evaluation.

This post-processing step is optional and meant for offline use only — it is not applicable in real-time systems.

## Deployment

* Deployment for C++ application is added. Refer to `deployc/c++`.
## Demo


<div align="center">
  <img src="./figs/fasttrack_occ_enlarge_bb.jpg" alt="Image main" width="50%" style="margin: 1%;">
</div>


Simply run:

```bash
python tools/demo_track.py video -f exps/example/mot/yolox_x_mix_det.py -c pretrained/bytetrack_x_mot17.pth.tar --fp16 --fuse --save_result
```

## Citation
If you use our code or Benchmark, please cite our work.


```
@misc{hashempoor2025fasttrackerrealtimeaccuratevisual,
      title={FastTracker: Real-Time and Accurate Visual Tracking}, 
      author={Hamidreza Hashempoor and Yu Dong Hwang},
      year={2025},
      eprint={2508.14370},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.14370}, 
}
```


## CPU & Embedded Optimizations (Constant-Gain Kalman Filter)

This repository includes steady-state constant-gain optimizations designed for real-time tracking on edge devices (such as Raspberry Pi 5 and ARM Cortex-A76 CPUs):

- **Constant-Gain Kalman Filter (yolox/tracker/constant_gain_kf.py)**:
  - Eliminates  \times 8$ covariance matrix Riccati recursion and online matrix inversion ((1)$ scalar-vector operations).
  - Precomputes steady-state gain schedules across object height buckets to maintain fast initialization snap-to-box dynamics.
  - Achieves tracker update latency under **.10\text{ ms}$** on Raspberry Pi 5.
- **Standalone Module (yolox/tracker/fasttracker_constant_gain.py)**:
  - Standalone, zero-heavy-dependency FastTracker implementation integrating constant-gain KF, occlusion handling, and duplicate suppression.
- **Documentation**:
  - Detailed architectural guide: [docs/Optimizing_FastTracker_CPU_RaspberryPi.md](docs/Optimizing_FastTracker_CPU_RaspberryPi.md)
  - Mathematical derivation and notes: [docs/Steady_State_Constant_Gain_Kalman_Filter_Notes.md](docs/Steady_State_Constant_Gain_Kalman_Filter_Notes.md)

## Acknowledgement
Our work is built upon [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), [ByteTrack](https://github.com/FoundationVision/ByteTrack/tree/main), [TransTrack](https://github.com/PeizeSun/TransTrack) and [TrackEval](https://github.com/JonathonLuiten/TrackEval). Highly appreciated!






