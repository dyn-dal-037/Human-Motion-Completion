
# Unified Hand Motion Completion under Occlusion

This project investigates robust hand motion modeling under partial and noisy observations—a critical challenge in vision-based human–robot interaction (HRI). Using a lightweight GRU-based sequence model, we reconstruct physically plausible full hand motion from incomplete joint trajectories.

## Overview

The focus of this work is **problem formulation, controlled evaluation, and failure analysis**, rather than large-scale deployment. We address common vision-based tracking failures such as self-occlusion, sensor noise, and missing joints due to lighting or viewpoint constraints.

---

## Problem Formulation

Let  denote a sequence of hand joint positions over time, where  joints.

Given an occluded observation  and a binary visibility mask  indicating missing joints, the task is to predict:



The goal is to reconstruct missing trajectories while strictly preserving temporal smoothness.

---

## Methodology

### Dataset & Occlusion Modeling

Hand joint trajectories were extracted from RGB videos using **MediaPipe Hands**, focusing on repeated grasp–release interactions. We simulate real-world degradation through:

* **Random Joint Dropout:** Simulating individual sensor failure.
* **Temporal Frame Dropout:** Simulating camera stutter or total occlusion.
* **Gaussian Noise:** Simulating low-light sensor interference.

### Model Architecture

A lightweight **GRU-based** (Gated Recurrent Unit) architecture was chosen to ensure interpretability and prevent overfitting on small datasets.

| Layer | Configuration | Output Shape |
| --- | --- | --- |
| **Input** | Flattened Joint Coordinates |  |
| **GRU 1** | 128 Units |  |
| **GRU 2** | 128 Units |  |
| **Dense** | Linear Projection |  |

### Training Objective

We utilize a **Masked Mean Squared Error (MSE)** loss. Error is computed *only* on the missing joints to force the model to learn temporal relationships rather than simply "copying" visible inputs.

---

## Evaluation

### Quantitative Results

* **Primary Metric:** Reconstruction MSE (on missing joints only).
* **Benchmark:** Achieved **MSE ≈ 0.015** under simulated occlusion.

### Qualitative Analysis

Trajectory plots demonstrate that the model successfully preserves the overall temporal structure of a grasp. However, degradation is observed during high-velocity finger movements.

> **Note:** Due to local hardware constraints (8 GB RAM), inference and evaluation were conducted via Google Colab.

---

## Limitations & Future Work

* **Data Scale:** Currently limited to 3 short sequences.
* **Generalization:** Focused on a single interaction type (grasp-release).
* **Physics:** No explicit physical constraints (e.g., bone length) are currently enforced.

**Future Path:** We aim to scale to diverse datasets, implement **Attention-based Transformers**, and integrate these modules into closed-loop robotic hand control systems.

---

## Project Structure

```text
hand_motion_completion/
├── data/
│   ├── joints_clean/          # Raw MediaPipe extractions
│   └── joints_occluded/       # Synthetically degraded data
├── models/
│   └── hand_completion_gru.h5 # Trained weights
├── scripts/
│   ├── extract_hand_pose.py   # MediaPipe preprocessing
│   ├── apply_occlusion.py     # Degradation scripts
│   └── train.py               # Training pipeline
├── day3_evaluation.ipynb      # Visualization & failure analysis
└── README.md


```
