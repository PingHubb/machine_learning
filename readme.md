# Advanced Gesture Recognition with Self-Supervised Learning and Denoising

This repository contains a comprehensive PyTorch pipeline for training advanced gesture recognition models on custom 8x7 capacitive sensor data. The project explores two distinct and innovative research tracks for processing and classifying spatio-temporal gesture sequences.

## Table of Contents
- [Project Overview](#project-overview)
- [Research Tracks](#research-tracks)
- [Required Folder Structure](#required-folder-structure)
- [Workflow and Usage](#workflow-and-usage)
  - [Track 1: Noise-Robust Classification](#track-1-noise-robust-classification)
  - [Track 2: Denoising Autoencoder](#track-2-denoising-autoencoder)
- [Model Architectures](#model-architectures)
- [Dependencies](#dependencies)

## Project Overview

The core of this project is to develop a highly accurate gesture classifier for data captured from an 8x7 sensor array. Instead of a simple supervised approach, this pipeline leverages modern deep learning paradigms to build more robust and generalizable models. It is structured to support two parallel research investigations.

## Research Tracks

This pipeline is organized into two primary research tracks, each with a distinct goal:

### 1. Noise-Robust Classification (Options 0, 1, 2, 3)
This track investigates the hypothesis that a model can be made more robust by first pre-training it on a large amount of general, potentially noisy, unlabeled data before fine-tuning it on a smaller, clean, labeled dataset.
- **Goal:** To create a state-of-the-art gesture classifier.
- **Method:** Uses a **Self-Supervised Learning** (SSL) paradigm with a **Masked Autoencoder** task. The model first learns the fundamental "structure" of gestures and then learns to associate that structure with specific labels.

### 2. Denoising Autoencoder (Options 4, 5, 6)
This track trains a dedicated sequence-to-sequence model to explicitly remove noise from a gesture signal.
- **Goal:** To create a model that can take a noisy gesture file as input and produce a clean version of it as output.
- **Method:** Uses a full **Encoder-Decoder Transformer** architecture trained to reconstruct clean data from artificially noised versions.

## Required Folder Structure

Before running the script, ensure your project directory is set up as follows:

## Workflow and Usage

The script is menu-driven. Run it from your terminal and select the desired option.

```bash
python your_script_name.py

## Workflow and Usage

The script is menu-driven. Run it from your terminal and select the desired option.

Track 1: Noise-Robust Classification
This track must be executed in the following order: 0 → 1 → 2 → 3.
Option 0: PREPARE Pre-training Data
What it does: This is a one-time setup step. It finds all .txt files in training_data/ and testing_data/, copies them to the pre_train_data/ folder, and renames them anonymously (e.g., data_1.txt, data_2.txt). This creates a clean, unified dataset for self-supervised learning.
When to run: Run this once at the very beginning of your experiment.
Option 1: Pre-train Classification Backbone
What it does: This is the core self-supervised learning stage. It uses the data in pre_train_data/ (without labels) to train the GestureBackbone. The model learns the fundamental structure of gestures by trying to predict randomly masked (hidden) time steps in the data. It also performs a hyperparameter search to find the best model architecture, saving the winning model (backbone.pth) and its configuration.
When to run: Run this after completing Option 0. This is the most computationally intensive step.
Option 2: Fine-tune Classifier
What it does: This stage takes the "smart" backbone pre-trained in Option 1 and teaches it the specific classification task. It loads the backbone.pth and fine-tunes it on your clean, labeled data from the training_data/ folder. This process is very fast and efficient. The final, ready-to-use classifier is saved as classifier_finetuned.pth.
When to run: Run this after completing Option 1.
Option 3: PREDICT gesture label
What it does: Loads the final classifier_finetuned.pth model and uses it to predict the gesture labels for any new .txt files you have placed in the predict/ folder.
When to run: Run this anytime after completing Option 2.
Track 2: Denoising Autoencoder
This track is independent of Track 1 and must be executed in the order: 4 → 5 → 6.
Option 4: PREPARE data for Denoising
What it does: This is a one-time setup step for this track. It loads all clean data, calculates a scaler, adds artificial Gaussian noise to create a "noisy" version of each file, and saves the (noisy, clean) pairs as .npy files in a dedicated pre-processing folder.
When to run: Run this once to begin the denoising experiment.
Option 5: TRAIN Denoising Autoencoder
What it does: Trains the full Encoder-Decoder DenoisingGestureTransformer. The model learns to take a noisy sequence as input and reconstruct the original clean sequence. The trained denoising model is saved.
When to run: Run this after completing Option 4.
Option 6: DENOISE a new gesture sequence
What it does: Loads the trained denoising model and uses it to clean any new gesture files placed in the predict/ folder. It outputs a new file with the prefix DENOISED_.
When to run: Run this anytime after completing Option 5.
Model Architectures
GestureBackbone (Encoder): A Convolutional Transformer (Conv-Transformer) that uses 2D convolutions for spatial feature extraction per time-step, followed by a Transformer Encoder to model temporal relationships. This is the core of both research tracks.
FineTunedGestureModel (Classifier): Wraps the pre-trained GestureBackbone and adds a simple linear layer on top for final classification.
DenoisingGestureTransformer (Autoencoder): A full Encoder-Decoder model. It uses the GestureBackbone as its encoder and a standard nn.TransformerDecoder to generate the denoised output sequence.