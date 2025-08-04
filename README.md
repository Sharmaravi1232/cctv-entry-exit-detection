# cctv-entry-exit-detection
Detect and count people entering and exiting through a virtual line in CCTV footage using YOLOv8 and DeepSORT. Real-time tracking with Gradio UI.
# 🦀 CCTV Entry/Exit Detection

A computer vision application to detect and count people crossing a virtual line in CCTV footage using YOLOv8 and DeepSORT. Built with Gradio for interactive use.

[![Hugging Face Spaces](https://img.shields.io/badge/Live%20App-HuggingFace-blue?logo=huggingface)](https://huggingface.co/spaces/Ravishankarsharma/cctv-entry-exit-detection)

 # Overview

This application uses YOLOv8 for object detection and DeepSORT for real-time tracking to count how many people enter or exit a defined area in CCTV footage. The app is deployed with Gradio on Hugging Face Spaces.

 📦 Metadata

`yaml
title: Cctv Entry Exit Detection
emoji: 🦀
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.39.0
app_file: app.py
pinned: false
short_description: Detect and count people crossing a virtual line in CCTV
 Key Features
🚶 Detects and tracks people using YOLOv8 + DeepSORT

📈 Counts people entering and exiting via a virtual line

🎥 Accepts video input in .mp4, .avi, or .mov formats

🖼️ Displays processed frame with visual annotations

☁️ Deployable via Hugging Face Spaces using Gradio
File Structure
# app.py – Main Gradio application logic

# requirements.txt – Required dependencies

# yolov8n.pt – YOLOv8 model weights (optional if downloaded automatically)

# Tech Stack
# Python 3.9+

# OpenCV (opencv-python-headless)

# Ultralytics YOLOv8

# DeepSORT Realtime

# Gradio 5.39.0




