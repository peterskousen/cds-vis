**Face Detection in Newspaper Images**
=====================================

This repository contains a Python script that detects faces in newspaper images using the MTCNN algorithm and Facenet PyTorch library. The script processes images, detects faces, and generates a CSV file and a plot showing the percentage of pages that include faces in relation to decade for three different historical Swiss newspapers.

**Data Source:**

The dataset used in this project is a collection of image files of pages from three different Swiss newspapers, ranging from the years 1804-2017. Information about the authors and a download link for a compressed zip file can be found [here](https://zenodo.org/records/3706863). <br>

**Reproducing the Analysis:**
------------------------------------

The entire extracted *Newspaper* folder containing *GDL*, *IMP*, and *JDG* should be placed in the `in` directory.

To reproduce the analysis, set directory to *a4* and run ```bash run.sh``` from the terminal. *run.sh* performs the following actions:
1. ```python -m venv .venv``` Sets up a virtual environment in the root directory
2. ```source .venv/bin/activate``` Activates the environment
3. ```pip install -r requirements.txt``` Fetches and installs required dependencies
4. ```python src/script.py``` Runs the main Python script


**Key Points from the Outputs:**
-----------------------------------------

* The script generates a CSV file `data.csv` in the `out` directory, containing the frequency of faces in relation to decade for each newspaper.
* The script generates a plot `plot.png` in the `out` directory, showing the frequency of faces in relation to decade for all three newspapers.

**Discussion of Limitations and Possible Steps to Improvement:**
-----------------------------------------------------------
