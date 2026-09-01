# Fruit Quality Detection

A Mini Project — APJ Abdul Kalam Technological University
Vimal Jyothi Engineering College, Chemperi (March 2025)

By: Adityan P, Anna Scaria, Aman SV, Agastya Harichandran
Guide: Ms. Silna KV

## Overview

Uses a Convolutional Neural Network (CNN) with transfer learning to classify
uploaded fruit images as **Good (fresh)** or **Bad (spoiled/rotten)** based on
color, texture, and surface defects.

## Tech Stack

- Python (TensorFlow / Keras, OpenCV, NumPy)
- Flask (backend + web interface)
- HTML/CSS (frontend)

## How it works

1. User uploads a fruit image through the web interface.
2. Image is preprocessed (resized to 100x100, normalized).
3. The trained CNN model (`model/fruit_quality_model.h5`) predicts a class.
4. The class is mapped to Good/Bad and shown to the user with a confidence score.

## Setup

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

## Notes

- `class_labels` in `app.py` are taken from the report and may need to be
  updated to match the actual labels used during training.
- The model file is large (~110MB); this repo uses Git LFS to track `.h5` files.
