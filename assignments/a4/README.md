**Face Detection in Newspapers**
=====================================

This repository contains a Python script that detects faces in images of pages from three different historical newspapers using a Multi-Task Cascaded Convolutional Network (MTNN) algorithm along with the Facenet PyTorch library. For each newspaper, the script processes the images with ```Pillow```, detects faces using ```MTCNN.detect()```, and generates a CSV file and a plot showing the percentage of pages that include faces in relation to decade for each of the three newspapers using ```Pandas``` and ```matplotlib```.

**Repo structure**
---------------------

```bash
└── a4
    ├── README.md
    ├── in
    │   └── newspapers
    │       ├── GDL
    │       ├── IMP
    │       └── JDG
    ├── out
    │   ├── data.csv
    │   └── plot.png
    ├── requirements.txt
    ├── run.sh
    └── src
        └── script.py
```
**Data Source and Prerequisites:**
-----------------------

The dataset used in this project is a collection of image files of pages from three different Swiss newspapers, ranging from the years 1804-2017.<br>
Information about the authors and a download link for a compressed zip file of all image data can be found [here](https://zenodo.org/records/3706863). <br>

The main script was written and executed using ```Python v.1.89.1```. 
For the processing and analysis of the data, the following packages were used:
```
pillow==10.2.0
tqdm==4.66.4
matplotlib==3.9.0
facenet-pytorch==2.6.0
pandas==2.2.2
```

When executed on a 32-core Intel Xeon-based CPU, the program took approx. 90 minutes to complete.

**Reproducing the Analysis:**
------------------------------------
Before running the program, the entire uncompressed *newspapers* folder containing *GDL*, *IMP*, and *JDG* should be placed in the *in* directory as showcased in the repo structure above.

To reproduce the analysis, change directory to *a4* and run *run.sh* from the from the terminal:
```bash
cd a4
bash run.sh
``` 
 *run.sh* performs the following actions:
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
4. Runs the main Python script:
    ```
    python src/script.py
    ``` 


**Key Points from the Outputs:**
-----------------------------------------
Appreciating the resulting dataframe and line plot for the three newspapers analysed, it appears that there has been a significant and steady increase in the frequency of images containing faces since the early 19th century. Especially around the start of the 20th, a upgoing trend which continues to accelerate during the entire 20th century can be observed. <br>
This trend is likely due to the advent of photography (incidently around the 1820s) and the rapid technological advantancements and increasing accessibility of cameras ushering in a shift from mainly text-based news to more multimodal, visually appealing publications.

**Discussion of Limitations and Possible Steps to Improvement:**
-----------------------------------------------------------

Possible pitfalls of this project may include limitations in the MTCNN algorithm used and the quality of the dataset, e.g. resolution of the scans, the contrast of colors, or the state of the source material itself, which may be scratched, torn, faded, poorly printed or otherwise damaged and unsuitable for the MTCNN. As the images can potentially differ widely from the data on which the model was trained, these are factors that may affect the perfomance of the pretrained neural network negatively.

While the analysis indicates a clear trend across time, it is worth noting that the dataset is limited to only three newspapers, all of which were published in the same country. As such, it does not necessarily reflect the trends of other societies and cultures. This is noteworthy, as the result may also be influenced by factors like changes in editorial policies, technological advancements, or shifts in societal values, which are not explicitly accounted for in the analysis.

In conclusion, the dataset suggests a significant increase in the presence of faces in historical newspapers, with varying patterns across the three newspapers. The results provide insights into the evolution of newspaper content and the growing importance of visual storytelling over time. Depending on the goal of the research, further work into this area would likely benefit from larger, more diverse datasets which include newspapers from different countries and from local or international publishers.