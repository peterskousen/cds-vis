# A1: Building a simple image search algorithm
## Overview


## Table of Contents

- [Repo Structure](#repo-structure)
- [Data Source and Prerequisites](#data-source-and-prerequisites)
- [Reproducing the Analysis](#reproducing-the-analysis)
- [Key Points from the Outputs](#key-points-from-the-outputs)
- [Discussion of Limitations and Possible Improvements](#discussion-of-limitations-and-possible-improvements)

## Repo structure

```bash
a1
├── README.md
├── in
│   └── flowers
├── out
│   └── similar_images.csv
├── requirements.txt
├── setup.sh
└── src
    └── img_search.py
```

## Data Source and Prerequisites:

The dataset used for this project is a collection of 1,360 images of flowers common to the UK distributed across 17 different classes. More information about the dataset, its authors, as well as a download link to a compressed file of the images can be found [here](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/).

The main script was written and executed using ```Python v.1.89.1```.
For the processing and analysis of the data, the following packages were used:

```
opencv-python-headless==4.9.0.80
pandas==2.2.2
```

## Reproducing the Analysis:

Before running the script, the unzipped folder containing the dataset should be placed in the *in* folder as shown in the repo tree above. The folder contains a couple .txt files which can safely be deleted prior to the analysis.

To reproduce the analysis, firstly change directory to *a1* and run *setup.sh* from the terminal:

```bash
cd local_path_to_a1
```

```bash
bash setup.sh
```

*setup.sh* perform the following actions:

1. Sets up a virtual environment called *.venv* in the root directory using ```venv```:
   ```sh
   python -m venv .venv
   ```
2. Activates the environment:

   ```sh
   source .venv/bin/activate
   ```
3. Fetches and installs required dependencies:

   ```
   pip install -r requirements.txt
   ```

Then, run *img_search.py* with an argument for the image file to base the image search on like so:

   ```
   python img_search.py filename.png
   ```

## Key Points from the Outputs:

| Filename        | Distance |
|-----------------|----------|
| target          | 0.0      |
| image_1303.jpg  | 2.47     |
| image_0248.jpg  | 2.47     |
| image_0247.jpg  | 2.47     |
| image_0791.jpg  | 2.56     |