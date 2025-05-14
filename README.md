# ![logo-semarec (1)](https://github.com/user-attachments/assets/819b8d60-b396-45ac-bdfc-197fe6356722) SemaREC
Multimedia Semaphore Pose Detection

SemaREC is a Web Application written in Python developed by Team Superconductor that utilizes YOLOv11 & YOLOv12 for Image/Video classification, as well as real-time inference.

# Pre-requisites
This repository will assume that Python is configured with the correct PATH settings, as well as pip.

# How to Deploy your local copy?
1) Clone this repository
   `git clone https://github.com/AHG-BSCS/semaphore-machine-learning/`
2) Navigate to the repository folder, this may vary depending where you have stored your file.
3) Download the required dependencies
   `pip install -r requirements.txt`
4) Navigate to `app` directory
   `cd app`
5) Run the `app.py` file using
   `python app.py`

You may browse the Web App using http://localhost:8000

# Note
Depending on the specifications of your hardware, it may take some time to infer images/videos, real-time classification might get a "laggy" experience for older hardware.

# License
This repository is licensed under MIT.

# Acknowledgements
- Python
- numpy
- OpenCV
- Flask
- Werkzeug
- Ultralytics
- Roboflow

# Special Thanks 
For providing extra Semaphore Datasets
- Skripsi Cyber University 
- skripsian
- And to you for checking out this project!
