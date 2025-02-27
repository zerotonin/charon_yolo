# Charon YOLO Detector

A streamlined YOLOv11-based object detection pipeline designed to handle image datasets in multiple treatments (subfolders). The system:

1. Recursively scans a parent directory for images.
2. Groups images by treatment (subfolder).
3. Runs YOLO-based detection to classify objects into **dead**, **alive**, or **ginko**.
4. Outputs:
    - A **multi-sheet Excel** or **multiple CSV** files:
        - **Summary** sheet/file listing each treatment and total counts per object class.
        - **One sheet/file per treatment** with columns `[imagename, class, confidence, xmin, ymin, xmax, ymax]`.
    - (Optional) **annotated images** with bounding boxes in a mirrored directory structure named `annotated/`.

---

## 1. Conda Environment Setup

This project comes with a pre-defined Conda environment file: **charon_detct_env.yaml**.

Below is an example of what it might look like:

```
name: charon_detect_env 
channels: 
 - conda-forge 
 - defaults 
dependencies:
 - python=3.10 
 - pip
 - openpyxl 
 - pandas 
 - pip: 
 - ultralytics 
 - opencv-python
```
### Installation Steps

1. Install Miniconda or Anaconda (https://docs.conda.io/en/latest/miniconda.html) if not already installed.
    
2. Navigate to the project folder containing `charon_detct_env.yaml`.
    
3. Run:
    
    ```conda env create -f charon_detct_env.yaml```
    
    This will create and install the environment named `charon_detect_env`.
    
4. Activate the environment:
    
    ```conda activate charon_detect_env```
    

Now you have all dependencies (including `ultralytics` for YOLO, `opencv-python`, `pandas`, etc.) ready.

---

## 2. Scripts Overview

### 2.1 charon_detector.py

A **Python module** containing the core YOLO detection logic. It:

- Recursively finds images in a specified directory.
- Loads YOLO weights from `weights/best.pt`.
- Uses a dictionary mapping to unify class labels:
    - 0: dead
    - 1: alive
    - 2: ginko
- Detects objects, aggregates results by treatment, and outputs:
    - **Excel** (multi-sheet) or
    - **Multiple CSV files**
- Optionally saves **annotated images** in an `annotated/` subfolder.

Docstrings are included in a NumPy-friendly format so that automated doc generators (like Sphinx or pdoc) can parse them.

### 2.2 charon_detect.py

A **CLI wrapper** that imports and calls `run_detection(...)` from `charon_detector.py`.

Usage:

``` bash 
python charon_detect.py /path/to/parent_dir [--get_resultimages] [--output xlsx|csv]
```
CLI Arguments:

- `directory`: (positional) The path to the parent directory containing treatment subfolders.
- `--get_resultimages`: When set, annotated images (with bounding boxes) are saved in a mirrored `annotated/` directory.
- `--output`: Select between `xlsx` (default, multi-sheet Excel) or `csv` (multiple CSV files in a `detection_results/` folder).

---

## 3. Directory Structure (Input)

This tool expects a **parent** directory with one or more **subfolders** (treatments), each containing image files:
```
    parent_dir/ 
    ├── treatment_hypoxia 
    │ ├── image_001.png
    │ ├── image_002.png
    │ └── image_003.png
    ├── treatment_normoxia 
    │ ├── image_001.png 
    │ ├── image_002.png 
    │ └── image_003.png 
    └── treatment_pharmacon 
    ├── image_001.png 
    ├── image_002.png 
    └── image_003.png
```

parent_dir can have multiple treatment subfolders, each with multiple images. Files can be .png, .jpg, .jpeg, .tif, etc.

---

## 4. Directory & File Structure (Output)

1. When `--output xlsx` is used:
    
    - A single `detection_results.xlsx` file is created in `parent_dir/` with:
        - Sheet 1: `Summary` (columns: `Treatment, alive, dead, ginko`).
        - Additional sheets, one for each treatment subfolder, with columns `[imagename, class, confidence, xmin, ymin, xmax, ymax]`.
2. When `--output csv` is used:
    
    - A folder named `detection_results` is created under `parent_dir/`, containing:
        - `Summary.csv` for treatment-level object counts.
        - One CSV per treatment (e.g., `treatment_hypoxia.csv`, `treatment_normoxia.csv`, etc.) with `[imagename, class, confidence, xmin, ymin, xmax, ymax]`.
3. Annotated Images:
    
    - If `--get_resultimages` is specified, an `annotated/` subfolder is created in `parent_dir/`.
    - Each treatment subfolder (e.g., `treatment_hypoxia/`) is mirrored under `annotated/` (e.g., `annotated/treatment_hypoxia/...`).
    - Each image is saved with bounding boxes and class labels drawn.

---

## 5. Class Label Mismatch Note

We have two sets of weights (`best.pt` and `last.pt`) where the internal labeling might be:  
0: alive  
1: dead  
2: ginko

But our training data was actually:  
0: dead  
1: alive  
2: ginko

To fix this mismatch at run-time, we override the class logic in the script so that:

- 0 is forced to be labeled "dead"
- 1 is labeled "alive"
- 2 is labeled "ginko"

In the future, we will retrain the system without this mix-up, ensuring the weights directly match the data. For now, the script’s dictionary approach works around the mismatch.

---

## 6. Example Usage

1. Activate environment:
    ```
    conda activate charon_detect_env
    ```
2. Run detection on `parent_dir` using Excel output and annotated images:
    ```
    python charon_detect.py /path/to/parent_dir --get_resultimages --output xlsx
    ```
3. Check `detection_results.xlsx` in `parent_dir/` and annotated images in `parent_dir/annotated/`.