# Enforcing Uniform Policies: A Computer Vision Project

## Introduction

This project aims to enforce uniform policies in colleges using computer vision techniques. It detects Uniform violations in real-time video feeds, logs data, and sends alerts for non-compliance.

## Features

- Uniform detection using YOLOv7.
- Facial recognition using dlib, face-recognition.
- Flask web application for real-time video feed streaming.
- Data logging in pandas DataFrame.
- Email alerts for violations.
- A maintained database.

## Installation

1. Clone the repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Download YOLOv7 weights and place them in the appropriate directory.
4. Set up Gmail API for email alerts.
5. Set up MongoDB and configure database connection.

## Usage

1. Run the Flask application with `python app.py`.
2. Access the application through the provided URL.
3. Connect an IP camera for live video feed.
4. Monitor for Uniform violations and receive email alerts.
5. View logged data in the MongoDB database.

## Dataset

- Dataset includes images for Uniform classification.
- Faces of individuals for facial recognition.

## Contributors

- Aparnesh Shukla
- Divyanshu Kumar
- Team GI Infoventures

## Acknowledgements

- Special thanks to GI Infoventures, Anubhav Patric, Hamza Sir, Sugandha Mam for their contribution.
