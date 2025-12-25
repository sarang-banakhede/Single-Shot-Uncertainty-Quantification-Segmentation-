# Probabilistic Segmentation Framework: Beta-Binomial Uncertainty Estimation

> **Status:** *Active Research / Ongoing Experimentation*

## Project Overview

This repository hosts a research framework designed to explore **uncertainty quantification in medical image segmentation**. Moving beyond standard deterministic segmentation, this project implements a probabilistic approach that models the distribution of segmentation masks rather than just point estimates.

By predicting distribution parameters directly from the input image, the model provides both a precise segmentation map and a pixel-wise uncertainty map, offering critical interpretability for medical diagnosis.

---

## Key Advantages of this Framework

Unlike existing uncertainty estimation methods, this framework offers distinct computational and analytical benefits:

### 1. Single-Shot Inference (Efficiency)
Standard Bayesian methods like **Monte Carlo (MC) Dropout** or **Deep Ensembles** require multiple forward passes (often 10–50) during inference to estimate uncertainty variance.
* **Our Approach:** Because the network predicts the distribution parameters ($\alpha, \beta$) directly, we generate both the segmentation and the uncertainty map in a **single forward pass**. This makes the model significantly faster and suitable for real-time clinical applications.

### 2. Uncertainty-Aware Training Dynamics
Since uncertainty is a differentiable output of the network, we can track uncertainty metrics (such as Average Variance or Evidence) **over epochs** alongside standard performance metrics. This allows us to observe how the model's confidence evolves during training—often revealing overfitting before it becomes apparent in the validation loss.

---

## Theoretical Foundation: The "GAMLSS" Inspiration

The core innovation of this framework draws inspiration from **Generalized Additive Models for Location, Scale, and Shape (GAMLSS)**.

In traditional regression or segmentation tasks, deep learning models often predict the conditional mean $\mu = E[Y|X]$. However, GAMLSS frameworks allow for the prediction of **all** parameters of a target distribution (not just the mean) as functions of the input.

### How this is applied here:

1.  **Assumption:** We assume that binary segmentation labels at each pixel follow a **Beta-Binomial distribution**.
2.  **Dual Prediction:** Instead of predicting a single probability score (logit), the neural network is designed to predict two parameters, **$\alpha$ (alpha)** and **$\beta$ (beta)**, for every pixel.
3.  **Output:**
    * **Segmentation Map ($\mu$):** Derived analytically as $\frac{\alpha}{\alpha + \beta}$.
    * **Uncertainty ($\sigma^2$):** Derived from the variance of the distribution $\frac{\alpha \beta}{(\alpha + \beta)^2 (\alpha + \beta + 1)}$.

This approach allows the model to learn "how confident" it should be. High $\alpha$ and $\beta$ values indicate high confidence (low variance), while low values indicate uncertainty (high variance), often occurring at object boundaries or in noisy regions.

### Methodology: The Combined Loss Function

To successfully train a neural network to learn both accurate segmentation **and** valid probability distributions, we employ a hybrid loss function that balances structural accuracy with probabilistic fit.

The total loss is defined as:
$$L_{total} = L_{Dice}(\mu, Y) + \lambda L_{NLL}(\alpha, \beta, Y)$$

**How it works:**
1.  **Dice Loss ($L_{Dice}$):**
    * **Role:** Ensures the derived mean mask $\mu$ overlaps accurately with the ground truth $Y$.
    * **Why:** Handles class imbalance (common in medical imaging) and ensures the global shape of the segmentation is correct.

2.  **Negative Log-Likelihood ($L_{NLL}$):**
    * **Role:** Maximizes the likelihood of the ground truth $Y$ given the predicted Beta distribution parameters $\alpha$ and $\beta$.
    * **Mechanism:**
        * If the pixel is Foreground ($Y=1$): The loss forces $\alpha$ to be high and $\beta$ to be low.
        * If the pixel is Background ($Y=0$): The loss forces $\beta$ to be high and $\alpha$ to be low.
        * **Crucially**, if the model cannot minimize the error easily (e.g., at a boundary), the NLL allows it to increase uncertainty (lower $\alpha$ and $\beta$) to reduce the penalty. This allows the model to "admit" it doesn't know, rather than making a confident wrong prediction.

---

## Current Experiments & Architectures

The project currently focuses on **Binary Thyroid Nodule Segmentation** using Ultrasound images. To test the hypothesis, I have implemented two distinct architectural paradigms to evaluate parameter disentanglement and feature reuse.

### 1. Dual Independent Stream Framework
* **Structure:** Two completely separate U-Net-like networks. One network explicitly learns $\alpha$, while the other learns $\beta$.
* **Goal:** To allow maximum flexibility where the feature space for "foreground evidence" ($\alpha$) and "background evidence" ($\beta$) can be entirely distinct.

### 2. Shared Encoder Framework 
* **Structure:** A single backbone encoder extracts high-level semantic features, which branch into two lightweight decoder heads for $\alpha$ and $\beta$.
* **Goal:** To test computational efficiency and regularization. Can a shared representation sufficiently capture the distinct evidence required for distributional parameters?

### Backbones Tested
* **ConvUNet:** Standard CNN-based U-Net.
* **TransUNet:** Hybrid Transformer-CNN architecture to capture long-range dependencies.

---

## Preliminary Results

![Inference Example 1](Single Shot Uncertainty Quantification & Segmentation /Outputs/Experiment_Dual_UNet/Inference_Results/0033_dual_unet_result.png)

![Inference Example 2](Single Shot Uncertainty Quantification & Segmentation /Outputs/Experiment_Shared_UNet/Inference_Results/0033_shared_unet_result.png)

![Experiment_Dual_UNet](Single Shot Uncertainty Quantification & Segmentation /Outputs/Experiment_Dual_UNet/Experiment_Summary_HighRes.png)

![Experiment_Shared_UNet](Single Shot Uncertainty Quantification & Segmentation /Outputs/Experiment_Shared_UNet/Experiment_Summary_HighRes.png)

![Experiment_Dual_TransUNet](Single Shot Uncertainty Quantification & Segmentation /Outputs/Experiment_Dual_TransUNet/Experiment_Summary_HighRes.png)

![Experiment_Shared_TransUNet](Single Shot Uncertainty Quantification & Segmentation /Outputs/Experiment_Shared_TransUNet/Experiment_Summary_HighRes.png)

## Roadmap & Upcoming Updates

This project is evolving to prove **architecture agnosticism** and **robustness**. The following updates are currently in the research pipeline:

### 1. Architecture Agnosticism: Swin-UNet Integration
I am extending the experiments to include **Swin-UNet** (Hierarchical Vision Transformer). By implementing Swin-UNet within both the *Dual* and *Shared* frameworks, I aim to demonstrate that this probabilistic paradigm works independently of the underlying feature extractor (CNN vs. Hybrid vs. Pure Transformer).

### 2. Uncertainty Benchmarking
To validate the quality of the Beta-Binomial uncertainty, I will benchmark the results against established frequentist and Bayesian methods:
* **Monte Carlo (MC) Dropout:** Running multiple forward passes at inference time.
* **Deep Ensembles:** Training multiple independent models to estimate epistemic uncertainty.
* **Goal:** To prove that the Beta-Binomial approach yields comparable or superior uncertainty calibration with significantly lower inference cost (single forward pass).

### 3. Domain Generalization & Robustness
While currently tested on Thyroid Ultrasound, the framework is being prepared for extension to diverse medical modalities to test robustness:
* **Dermatology:** Skin Lesion Segmentation (ISIC datasets).
* **Microscopy:** Cell nuclei segmentation.
* **Volumetric Data:** 3D MRI segmentation (Brain/Prostate).

---

> *This repository is a representation of ongoing research. The code structure reflects an experimental environment optimized for rapid iteration and hypothesis testing.*