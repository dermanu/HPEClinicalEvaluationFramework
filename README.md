# Framework for the Evaluation of Real-Time 3D Human Pose Estimation Algorithms for Motor Rehabilitation

This repository provides the implementation and resources for the publication:

**"Framework for the Evaluation of Real-Time 3D Human Pose Estimation Algorithms for Motor Rehabilitation"**  
*Emanuel Alexander Lorenz*  
Norwegian University of Science and Technology (NTNU), 2025  
DOI: ???

The study introduces a comprehensive framework for evaluating the applicability of real-time 3D human pose estimation (HPE) models in clinical rehabilitation settings. The framework facilitates the benchmarking of HPE models for spatial accuracy, robustness against setup errors, and clinical relevance using a marker-based motion capture system as the gold-standard baseline.

---

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Repository Structure](#repository-structure)
5. [Datasets](#datasets)
6. [Citation](#citation)
7. [License](#license)

---

## Features

- **Spatial and Spatiotemporal Accuracy Assessment:** Evaluate the precision of HPE models using metrics like MPJPE, MPJAE,MPJVAE and PCC.
- **Setup Error Simulation:** Simulate clinical conditions like occlusions, blur, underexposure, camera miscalibration, and background noise.
- **Camera Placements:** Ability to evaluate different camera angles and combinations.
- **Multi-Camera Support:** Test mono-ocular and multi-ocular setups with synchronized inputs.
- **Morphing Model:** Align keypoints across datasets to harmonize anatomical definitions using a pre-trained morphing model
- **Benchmarking Tools:** Includes augmentation scripts and statistical analysis pipelines.
- **Modularity:** Possibility to evaluate other models/datasets, setup errors and metrics, with relative little adjustments.

---

## Installation

1. Clone the repository and install dependencies:
```bash
# Clone the repository
git clone https://github.com/username/HPEClinicalEvaluationFramework.git
cd HPEClinicalEvaluationFramework

# Install Python dependencies (there might be the need to install additional dependencies)
pip install -r requirements.txt
```

2. Download [`xception_pascalvoc.pb`](https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.1/xception_pascalvoc.pb) and `haarcascade_frontalface_alt.xml` and place into `/utils`. Those are used for 
   changing the background and occluding faces for anonymisation.

3. Login into your W&B account, following the instructions here: [https://docs.wandb.ai/quickstart/].

4. Add the model you want to evaluate to the folder `/models`. Use the existing models as template.

5. Make changes in named scripts to allow for the testing of other HPE models (or ground-truth datasets). Those adjustments, will be replaced by a global yaml file including all parameters in the future:
   1. `skeletonMorphing/loadMorphDataset.py`:  Adjust path to your dataset, used model and evtl. difference in dataset structure.
   2. `skeletonMorphing/readDatasetMorph.py` and `utils/readDatasetEval.py`: Add output structure of specific HPE models keypoints (marked).
   3. `skeletonMorphing/trainSkeletonMorphing.py`: Change data paths.
   4. `utils/readDataEval`: Adjust order of HPE models keypoint output order.
   5. `evaluation_pipeline`: Add other HPE models if needed.

### System Requirements
- Python 3.8+
- CUDA-enabled GPU (optional for acceleration)
- Optional: High-Performance-Cluster (HPC) to parallelize the evaluation.

---

## Usage
1. **Prepare Dataset:**

   Prepare the dataset by running the modified `loadMorphDataset.py`. This will generate a serialized PyTorch state dictionary (.pth) for
   each participant, including the ground-truth and model-specific keypoints.
   ```bash
   python skeletonMorphing/loadMorphDataset.py
   ```

1. **Training the Morphing Model:**

   Train the morphing model using W&B hyperparameter sweep. The related config with the hyperparameters is `skeletonMorphing/config.yaml`.
   ```bash
   python skeletonMorphing/trainSkeletonMorphing.py
   ```
   
   If you don't want to do a hyperparameter sweep use this function instead. The related config with the hyperparameters is `skeletonMorphing/configFinal.yaml`.
   ```bash
   python skeletonMorphing/trainSkeletonMorphingFinal.py
   ```

2. **Run Evaluation Framework:**

   Evaluate an HPE model using the previously trained morphing model:
   ```bash
   python evaluation_pipeline.py --model_type mono
   python evaluation_pipeline.py --model_type multi
   ```
   To change the parameters of the evaluation adjust `config_mono.yaml` and `config_multi.yaml`.
   To run the evaluation on a HPC adjust `run_job_Mono.slurm`, `run_job_Mono.slurm`, and `submit_jobs.sh`.


3. **Analyze Results:**

   1. Statistical analysis of the morphing models performance:
      ```bash
      python statistics/morphing_statistics.py <path/to/all_ground_truths.npy> <path/to/all_hpe_truths.npy> <path/to/all_predictions.npy>
      ```

   2. Statistical analysis of the HPE models evaluation:
      ```bash
      python statistics/recalculate_metrics.py <path/to/evaluation/results> /statistics/results
      ```
	
   3. Visualisation of the results:
         ```bash
         python statistics/plot_metrics.py --data_type mono
		 python statistics/plot_metrics.py --data_type multi
         ```

---

## Repository Structure

```plaintext
HPEClinicalEvaluationFramework/
├── models/                         # HPE models and templates to include new HPE models
├── results/                        # Evaluation outputs
├── skeletonMorphing/               # Source code for training the morphing model
│   ├── loadMorphDataset.py         # Scripts for preparing and loading the training/evaluation dataset
│   ├── modelSkeletonMorphing.py    # Model of the morphing model
│   └── trainSkeletonMorphing.py    # Training/Evaluation/Testing of the morphing model
├── statistics/                     # Statistical analysis
│   ├── morphing_statistics.py      # Statistical analysis of the morphing models performance
│   ├── plot_metrics.py             # Plotting the results of the HPE model evaluation
│   └── recalculate_metrics.py      # Statistical analysis of the HPE model evaluation
├── utils/                          # Scripts for calculating various metrics, and augmenting the input frames
├── requirements.txt                # Python dependencies
├── evaluation_pipeline.py          # Main evaluation pipeline testing various augmentations using W&B
└── README.md                       # Project description
```

---

## Datasets

This repository uses the **VizLab Dataset**, collected at NTNU's Motion Capture and Visualization Laboratory. The dataset includes:

- **Participants:** 23 healthy adults (9 females)
- **Movements:** Common rehabilitation exercises (e.g., lunges, squats, rotations)
- **Cameras:** Six synchronized FLIR Blackfly S cameras at fixed positions

The dataset is currently not publicly available, due to ethical considerations.

---

## Citation

If you use this framework in your research, please cite:

```
@article{lorenz2025framework,
  title={Framework for the Evaluation of Real-Time 3D Human Pose Estimation Algorithms for Motor Rehabilitation},
  author={Emanuel Alexander Lorenz},
  journal={NTNU Technical Reports},
  year={2025}
}
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
