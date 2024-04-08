# Methods

The methods describe the initial collection of the ground truth dataset and then the framework for the applied augmentation methods and subsequent evaluation of various 3D real-time HPE models. 

**Keep dataset characteristics in methods - collection method into appendix?**
# Dataset Collection
The following chapter describes the experimental setup and procedure of the dataset collection, the subsequent data processing and the contents and structure of the dataset. **(Take out)**
## Experimental Setup 
### Capture space
The study was conducted at the Motion Capture and Visualization Laboratory (VizLab) at the Norwegian University of Science and Technology (NTNU). The capture space measured 5.74 x 5.4 meters with a central movement area of 1.6 x 1.6 meters in which the participant performed the movements (Figure **?**). The laboratory does not have any windows and its floor as well as walls are painted blue. 
### Marker-based Motion Capture
The baseline employed marker-based motion capture, considered as the current gold-standard method. Eight ceiling-mounted cameras, consisting of 1x Oqus 300, 2x Oqus 310, and 5x Qualisys Opus 500+ from Qualisys AB (Sweden), were utilized. The data acquisition was facilitated by the Qualisys Track Manager (QTM) software v.**?** (Qualisys AB, Sweden) at a sampling rate of 100 Hz. The position of the motion capture cameras is displayed in (Figure **?**). 

Subsequent to the antorphrometic measurements, reflective markers with a diameter of 14 mm were applied by an investigator with expertise in marker-based motion capture. The markers were placed according to a modified Plug-in Gait model, also known as the Conventional Gait Model (CGM) (**CITE**). This marker set was chosen, due to its widespread use in clinical practice and its modification to focus on the important segments/joints relevant for motor rehabilitation and alignment with current human pose estimation (HPE) models (**CITE**). A detailed description of modified marker placement is illustrated in Figure **?**.
### RGB Cameras 
Video data was recorded using six FLIR cameras **?** (**?**) at a sample rate of 25 Hz and a resolution of 1280x 1024 px. The cameras were mounted on a stative at 1-meter height (approx. desk height) and approximately 3 meters distance from the center of the room (Figure **?**).
### Movement Instructions 
The movement instructions were displayed on a 75" LCD screen (KDL-75W855C, Sony Group Corporation, Japan) in front of the participants at 2.22 meters distance from the center of the room. 
As presentation software Unity v. **?** (Unity Technologies, CA, USA) was used. Upon every displayed movement iteration start and stop an event trigger was sent via LabStreamingLayer (LSL), to facilitate movement segmentation during post-processing.
### Synchronization
The temporal synchronization of both camera systems, namely the motion capture and RGB cameras was achieved through a transistor-transistor-logic (TTL) signal. The Qualisys system operated as master, transmitting a TTL signal upon their shutter opening, to the FLIR cameras, triggering their shutter opening, and thus capturing a frame with a minimal temporal offset of approximately 14 microseconds.
To record the marker positions, camera frames, and movement event marker, the LSL LabRecorder was employed. This tool records and adjusts each data stream's timestamps to a common reference clock. For capturing  the marker positions, a customized version of the Qualisys Lab Streaming Layer App client (**CITE**) was used. Concurrently, a custom Python script saved the corresponding video frames and transmitted the respective frame number via LSL upon receiving the video frame.

> [!todo]
>  FIGURE: Signal flow diagram. Hardware/Software communication of setup.
>  FIGURE: Top-down view of hardware setup with measures in meters 
>  FIGURE: Fotography of Setup 

## Participants
The collection of the dataset was approved by The Data Protection Services of Sikt – Norwegian Agency for Shared Services in Education and Research (Reference number: **?**) and aligned with the declaration of Helsinki. The dataset include **?** healthy volunteers (**?** females) with an average age of **?** (SD: **?**) and an average Body-Mass-Index (BMI) **?** (SD: **?**) **Height and weight instead BMI**. The participants showed neither show any physical or cognitive problems and have had normal or corrected to normal vision.
## Experiment Procedure
Prior to the arrival of the participant, the marker-based system was calibrated. After briefing the participant on the experimental procedure and obtaining their consent, their anthropometric measurements were documented. Then the markers were attached as mentioned and the participant was positioned within the central movement area. A movement sequence  was presented on the LCD screen three times, with the participants encouraged to replicate the movement to familiarize themself with the movements. After a short break, the movement sequence is displayed 10 times and the participant is instructed to perform it synchronously with the displayed movement. 

The movement sequences were chosen together with a subject matter expert in physiotherapy and movement science and represent common movements as part of motor rehabilitation. 
The movement can be categorized as simple movements of isolated segments, as well as complex movements of multiple segments standing and sitting. Complex movements are supposed to be more challenging to HPE models  due to occurring joint occlusion. A detailed list of the movements is displayed in Table (**?**), Table (**?**) and Table (**?**) 

Table (**?**): Movement that can be categorized as simple movements performed standing. The movement number corresponds to the description of the movement in the dataset.

| Body Segment | Movement Sequence       | Movement Number |
| ------------ | ----------------------- | --------------- |
| Shoulder     | Flexion/Extension       | 1               |
|              | Abduction/Adduction     | 2               |
|              | Inward/Outward rotation | 3               |
| Elbow        | Flexion/Extension       | 4               |
| Hip          | Flexion/Extension       | 5               |
|              | Abduction/Adduction     | 6               |
|              | Inward/Outward rotation | 7               |
| Knee         | Flexion/Extension       | 8               |

Table (**?**): Movement that can be categorized as complex movements performed standing. The movement number corresponds to the description of the movement in the dataset.

| Movement Sequence                                     | Movement number |
| ----------------------------------------------------- | --------------- |
| Forward lunge (includes dorsiflexion) n               | 9               |
| Deep squat                                            | 10              |
| Arm/Leg crossover                                     | 11              |
| Trunk rotation (pendulum movement or arms cross-over) | 12              |
| Step sideways                                         | 13              |

Table (**?**): Movement that can be categorized as complex movements performed sitting. The movement number corresponds to the description of the movement in the dataset.

| Movement Sequence                            | Movement number |
| -------------------------------------------- | --------------- |
| Standing up/Sitting down                     | 14              |
| Step sideways                                | 15              |
| Heel raise (dorsiflexion of the ankle joint) | 16              |
| Knee extension                               | 17              |

# Data Processing
The following chapter describes the post-processing of the various data streams recorded.
### Marker-based Motion Capture
#### Qualisys Track Manager
First, the motion capture data recorded with QTM was visually inspected in QTM. Missing or false marker labels were corrected, and gaps in the marker trajectories were manually filled, by an investigator with expertise in marker-based motion capture.
#### Vicon Nexus
For further processing and modeling, Vicon Nexus v. 2.15.0.145908h x64 (Vicon Motion Systems Ltd, UK) was used. First, the subject marker positions were scaled to a static skeleton using a T-Pose sequence. Based on the calibrated skeleton the static PlugIn Gait model was calculated to calculate subject-specific offset.
Next, a second round of automatic and manual gap-filling was applied, using kinematic and pattern-based algorithms. Then the data was smoothed using a Butterworth low-pass filter (4<sup>th</sup> order, zero-lag, cut-off: 6 Hz) and the dynamic PlugIn Gait model was applied to calculate kinematic (and kinetic) outputs.
#### MATLAB
Using MATLAB v. R2023a (The MathWorks, Inc., Massachusetts, USA) costume scripts were developed the further process the  data. A first script verified the resulting files from the previous processing steps. A second script then aligns the output of models with the temporal synchronized motion capture data recorded with the LSL LabRecorder based on the smallest Euclidean distance between both data streams. Thus, the processed modeled motion capture data is now temporally aligned with the RGB camera data and movement event markers. The last script segments the movement data and video data according to the respective movement sequences. The resulting file structure and data types are displayed in Figure (**?**). The model kinematics and kinetics are displayed in Table (**?**) and Table (**?**). 

Table **?**: Data structure of .csv files in 'SegmentedData' and their respective units.

| Column          | Description                             | Unit                                   |
| --------------- | --------------------------------------- | -------------------------------------- |
| Time            | Time in from sequence start             | $[hours:minutes:seconds.milliseconds]$ |
| CameraFrame     | Corresponding camera frame              | $[1]$                                  |
| MarkerPostion_X | X-position of the markers               | $[mm]$                                 |
| MarkerPostion_Y | Y-position of the markers               | $[mm]$                                 |
| MarkerPostion_Z | Z-position of the markers               | $[mm]$                                 |
| JointCenter_X   | X-position of the modeled joint centers | $[mm]$                                 |
| JointCenter_Y   | Y-position of the modeled joint centers | $[mm]$                                 |
| JointCenter_Z   | Z-position of the modeled joint centers | $[mm]$                                 |


> [!todo]
> FIGURE: What the file structure of the dataset is
> TABLE: Data structure (parameters) of extended .csv file (include?! Not really used) 
> 	Include: Joint angles, Joint forces (probably incorrect), Joint centers,  Joint moments (probably incorrect), Joint Power (probably incorrect)
> 
### Cameras
Their extrinsic (position and orientation) and intrinsic parameters (focal length, pixel size, and image origin) were calculated using **?** (**CITE**) and **?** (**CITE**). The final projection matrix was then calculated using a custom algorithm.
# Assessment Pipeline
The clinical evaluation framework for real-time 3D HPE models is based on Python (v. (**?**)). This framework augments the video input according to common errors observed in clinical settings. It evaluates the spatial, tempo-spatial accuracy and inference speed of both mono- and multi-ocular HPE models across different clinically relevant movements and body segments. Thus, discerning the most suitable algorithm for a certain application in motor rehabilitation. (**DONE for various combinations of errors**)
The framework consists of various steps:
**1. Augmentation:** This step augments the input frames and camera calibration parameters to simulate eventual setup errors.
**2. Inference:** The chosen HPE model predicts the joint centers based on the augmented input frames.
**3. Skeleton Morphing:** A pre-trained model aligns predicted joint centers with ground truth by accounting for potential differences in the definitions of joint center positions.
**4. Evaluation:** The predicted joint center position is compared to the ground truth using various spatial and tempo-spatial metrics, as well as the model's inference speed, for different movement types and body segments.
## Augmentation
To explore the algorithm's validity in clinical applications, its robustness has to be assessed through data augmentation. Different combinations of augmentation methods were used the best simulate positive errors encountered in clinical settings. For the augmentation of the image frames (defocus, underexposure, motion blur, occlusion) the python library imaging and pixlib (different backgrounds) were used (**CITE**, **CITE**). For the augmentation of the camera parameters (**CITE**) and for the temporal desynchronization of camera inputs custom script was developed. Examples of the augmentations are shown in Figure (**?**).

| Augmentation             | Library  | Function and/or Parameters                                                               | Description                                                                                                                       |
| ------------------------ | -------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Defocus                  | imgaging | `DefocusBlur(severity=1) `                                                               | Poor camera focus, due to setup error                                                                                             |
| Underexposure            | imgaging | `GammaContrast(0.5)`                                                                     | Dimm environment, due to environmental insufficiencies                                                                            |
| Motion Blur              | imgaging | `MotionBlur(severity=1) `                                                                | Mild motion blur, due to insufficient camera hardware/settings or too fast movements                                              |
| Occlusion                | imgaging | `Cutout(size=(0.1, 0.2), fill_mode="gaussian")`                                          | Randomly positioned occlusion sized at 10-20% of frame size, due to setup error                                                   |
| Background               | pixlib   | `change_frame_bg() `                                                                     | Different noisy backgrounds, such as home, clinical or outdoor settings                                                           |
| Camera parameters        | -        | `np.random.normal(size=rvec.shape, loc=np.mean(rvec) * 0.02, scale=np.std(rvec) * 0.02)` | Noisy internal and external camera parameters, adding normal distributed noise (mean: 2 px, std: 1 px) using a multi-ocular setup |
| Camera desynchronization | -        | `cap.set(cv2.CAP_PROP_POS_FRAMES, np.random.integers(low=0, high=2, size=1))`            | Insufficient temporal synchronization (random offset of 0-2 frames) of camera input, when using a multi-ocular setup              |

> [!todo]
> Figure: Showing influence of different augmentation methods graphically
### Camera Position/Number
Besides the named error sources, the setup's usability, accuracy and inference time of the algorithm are dependent on the number and placement of cameras. More cameras result in a more tedious setup and calibration process, as well as require more computational resources. On the other hand, the accuracy of the predicted joint centers is increased, especially for 3D models (**CITE**).

> [!todo]
> TABLE/Figure: Showing different camera placements

**Just talk about the most obvious and mention extrem cases.**

2. Predifined camera settings:
	1. 1 camera (all 6 positions)
	2. 2 cameras (45 Grad front, front-back, 90 Grad front)
	3. 3 cameras (only front, one side, side/front/back) **3 and more cameras as an own category**
## Inference 

### Models
Table (**?**): Mono-ocular real-time 3D HPE algorithms used.

| Model name | Description | Reference |
| ---------- | ----------- | --------- |
| MediaPipe  |             |           |
| VNect      |             |           |
| CanonPose           |             |           |


Table (**?**): Multi-ocular real-time HPE algorithms used.

| Model name | Description | Reference |
| ---------- | ----------- | --------- |
| OpenPose   |             |           |
| CDRNet     |             |           |
|            |             |           |

### Post-processing of predicted keypoints
As suggested by [@karashchuk_anipose_2021] the predicted key points are post-processed to further improve their viability. Thus a median filter is applied to the predicted outputs.
## Skelton Morphing
As different models are trained on different datasets, their definition of the keypoint location varies. To compensate for the offset between the respective model's prediction and our ground truth dataset we train a network predicting this, according to [@wandt_canonpose_2020]. The morphing models are trained on participant **?, ?**  (1 male, 1 female) keypoints', which data is excluded during the subsequent model evaluation.
## Evaluation
A subset of relevant spatial and tempo-spatial metrics was chosen to facilitate the clinical evaluation of different real-time 3D HPE algorithms for different movement types and body segments in different settings. The Python library wandb was used to run different augmentation and subsequently log relevant metrics [@biewald_experiment_2020].
### Spatial Accuracy
#### MPJPE
[[MPJPE]] (Mean per joint position error) calculates the Euclidean distance between the predicted keypoints and ground truth to evaluate the spatial accuracy of the model (Formular **?**). We include it for general comparison between datasets and algorithms, due to its widespread use in the field of HPE (**CITE**). Lower is better.
$$MPJPE = \frac{1}{N_F}\frac{1}{N_{J}\sum\limits_{f,j}} \left\lVert p_{f,j}-\hat{p}_{f,j} \right\rVert _2$$
THE MPJPE is calculated for all frames $N_F$ and all joints $N_J$ of the predicted keypoints $p$ compared to their ground truth $\hat{p}$ of the same frame.
#### PA-MPJPE
PA-MPJPE (Procrustes-MPJPE) performs a rigid transformation on the predicted 3D HPE to align it with the ground truth and then calculates the [[MPJPE]] (Formular **?**). Therefore different rotations and translations of the 3D HPE in space are not taken into consideration, which are not relevant for the prediction of therapy-relevant parameters such as ROM, and velocity/acceleration changes [@goodall_procrustes_1991]. Lower is better.
#### Angular Error
The Range of Movement (ROM) is a relevant metric in rehabilitation, providing an assessment of movement ability through joint ranges. Therefore, the mean absolute angular error for each joint over multiple frames for a movement is calculated (Formular **?**). Notably, the comparison is limited to **flexion/extension angles**, as most HPE models cannot predict rotational angles.

>[!todo]
> Formular?!
#### Correct Pose Score
The correct pose score (CPS) evaluates the prediction not joint by joint, but rather on the correctness of the whole pose. Therefore a threshold is set, which defines a pose as correct if the Euclidean distance between each joint's prediction and ground truth for a single frame is below it. Then the percentage of correct poses is calculated for a given dataset. The threshold is set to 180 mm according to [@wandt_canonpose_2020] introducing this metric.

>[!todo]
> Formular?!
### Spatio-Temporal Accuracy
As movements are 4D dimensional (3D space over time) temporal parameters should be taken into consideration. In clinical assessment not only joint position and its trajectories play a role, but also kinematics like movement velocity and acceleration (**CITE**). 
#### Velocity Error
The movement velocity is also an important metric in the assessment of rehabilitation exercises (**CITE**). Although they directly derive from the positional measurement, separate demonstrations enable easier assessment of the model's usability in certain applications. Their mean is calculated for a given dataset.

>[!todo]
> Formular?!
#### Similarity
Further, it might aid clinicians in understanding the similarity of the movement trajectories between the current gold standard and HPE model measurements by assessing the spatio-temporal similarity.
Using Pearson correlation for movement segments and specific joints the spatio-temporal can be calculated for the angular position of the joints. Their mean is calculated for a given dataset
$$r = \frac{\sum_\limits{f,j} (p_{f,j} - \bar{p})(\hat{p}_{f,j} - \bar{\hat{p}})}{\sqrt{\sum_\limits{f,j}(p_f,j - \bar{p})^2} \sqrt{\sum_\limits{f,j}(\hat{p}_{f,j} - \bar{\hat{p}})^2}}$$
The correlation is calculated for each frame $f$ and joint $j$ between the predicted keypoints and their ground truth $\hat{p}$. The mean for a movement sequence is then calculated.
## Inference Speed
The inference speed gives an insight into the real-time applicability of a model. The calculated inference speed is based on the mean time of the model's inference and post-processing steps for a single frame.
