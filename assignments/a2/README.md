# A2: Classification benchmarks with Logistic Regression and Neural Networks
## Overview
For this project, the goal is to train two image classifiers based on logistic regression and multi-layer perceptron (MLP) architectures, respectively. This is done by firstly preprocessing the image data, including greyscale conversion, pixel value normalization and reshaping of data, i.e., flattening of 2D array into 1D, using `OpenCV`. Then, utilizing  `scikit-learn`'s built in functions `LogisticRegression()` and `MLPClassifier()`, along with `classification_report`, a classification report for evaluating model performance is generated. Simoultaneously, two `.joblib` files are created for each classifier for easier future reemployment.

## Table of Contents

- [Repo Structure](#repo-structure)
- [Data Source and Prerequisites](#data-source-and-prerequisites)
- [Reproducing the Analysis](#reproducing-the-analysis)
- [Key Points from the Outputs](#key-points-from-the-outputs)
- [Discussion of Limitations and Possible Improvements](#discussion-of-limitations-and-possible-improvements)

## Repo structure

```bash
a2
├── ReadMe.md
├── in
├── out
│   ├── lrc_report.txt
│   ├── mlp_report.txt
│   ├── lr_classifier.joblib
│   ├── mlp_classifier.joblib
│   └── plot.png
├── requirements.txt
├── run_lrc.sh
├── run_mlp.sh
└── src
    ├── lrc.py
    └── mlp.py
```

## Data Source and Prerequisites:

The classifiers were trained on the `Cifar10` dataset which comprises 60,000 32x32 px full color images. The image data is split into 10 classes, each containing 6,000 images. Further information about the dataset and its creators along with a download link for a Python-friendly version can be found [here](https://www.cs.toronto.edu/~kriz/cifar.html).

 Note, however, that when running the scripts, `Cifar10` is fetched and loaded via `TensorFlow`. Thus, no prior manual steps are needed to obtain the images for the scripts to work.

The main script was written and executed using ```Python v.1.89.1```.
For the processing and analysis of the data, the following packages were used:

```
matplotlib==3.8.3
numpy==1.26.4
opencv-python-headless==4.9.0.80
scikit-learn==1.4.1.post1
tensorflow==2.16.1
```

## Reproducing the Analysis:

To reproduce the analysis, change directory to *a2* and run either *run_lrc.sh* or *run_mlp* from the from the terminal:

```bash
cd a2
```

```bash
bash run_mlp.sh
```

OR

```bash
bash run_lrc.sh
```

*run_lrc.sh* & *run_mlp.sh* perform the following actions:

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
   python src/lrc.py 
   ```

   OR

   ```
   python src/mlp.py
   ```

## Key Points from the Outputs:
<table>
  <tr>
    <td style="width: 50%; font-size: 12px;">

### LRC Report

| Class       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Airplane    | 0.32      | 0.37   | 0.34     | 1000    |
| Automobile  | 0.27      | 0.28   | 0.27     | 1000    |
| Bird        | 0.21      | 0.16   | 0.18     | 1000    |
| Cat         | 0.17      | 0.13   | 0.15     | 1000    |
| Deer        | 0.19      | 0.16   | 0.18     | 1000    |
| Dog         | 0.26      | 0.25   | 0.26     | 1000    |
| Frog        | 0.20      | 0.18   | 0.19     | 1000    |
| Horse       | 0.23      | 0.24   | 0.23     | 1000    |
| Ship        | 0.31      | 0.36   | 0.33     | 1000    |
| Truck       | 0.31      | 0.41   | 0.35     | 1000    |
|**Accuracy** |           |        | **0.25** | 10000   |
|**Macro Avg**| **0.25**  |**0.25**| **0.25** | 10000   |
|**Weighted Avg**|**0.25**|**0.25**| **0.25** | 10000   |

</td>
<td style="width: 50%; font-size: 12px;">

### MLP Report

| Class       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Airplane    | 0.38      | 0.50   | 0.43     | 1000    |
| Automobile  | 0.41      | 0.36   | 0.38     | 1000    |
| Bird        | 0.31      | 0.30   | 0.31     | 1000    |
| Cat         | 0.27      | 0.14   | 0.19     | 1000    |
| Deer        | 0.32      | 0.14   | 0.19     | 1000    |
| Dog         | 0.31      | 0.38   | 0.34     | 1000    |
| Frog        | 0.35      | 0.37   | 0.36     | 1000    |
| Horse       | 0.34      | 0.49   | 0.40     | 1000    |
| Ship        | 0.40      | 0.51   | 0.45     | 1000    |
| Truck       | 0.42      | 0.39   | 0.40     | 1000    |
|**Accuracy** |           |        | **0.36** | 10000   |
|**Macro Avg**| **0.35**  |**0.36**| **0.35** | 10000   |
|**Weighted Avg**|**0.35**|**0.36**| **0.35** | 10000   |

</td>
</tr>
</table>


Upon assessment of the classifications reports, it is evident that while both models score relatively low across all measures, the general trend is that the neural network appears to perform slightly better across the board when i comes to classifying the images contained within `Cifar10`; F1-scores are all higher for the MLP classifier with the highest score being 0.41 (ship) and the lowest being 0.19 (cat/deer). In contrast, the highest F1-score for the logistic regression is 0.35 (truck) while the lowest score reaches 0.15 (cat).

Both models appear to be best at classifying machinery such as trucks, ships, airplanes, and automobiles, while having more trouble identifying the images of animals present in the dataset. This is possibly due to a larger variety in unique features in animals, e.g. the scenes in which they appear, relative bodily proportions, or lighting, among others.
