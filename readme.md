
---

# Audio Pattern Detection Project

This repository contains two Python applications for detecting repeated words or phrases in audio files. The project provides two different approaches: one based on classic signal processing and the other using machine learning techniques.

### Versions

* **V1 (app.py):**
  Implements a traditional signal processing method using Normalized Cross-Correlation. This version is fast and effective for finding exact matches.

* **V2 (app\_v2.py):**
  Uses MFCCs (Mel-Frequency Cepstral Coefficients) combined with Dynamic Time Warping (DTW). This approach is more robust and performs better when handling variations in speech, such as differences in pitch, speed, or speaker.

---

## Prerequisites

* Python 3 installed on your system.
  Verify installation with:

  ```bash
  python --version
  ```

---

## Setup and Installation

Follow the steps below to prepare and run the project:

### Step 1: Download the Code

Clone or download all project files and place them in a single directory.

### Step 2: Create a Virtual Environment

Create an isolated environment for the project:

```bash
python -m venv venv
```

This will create a folder named `venv` inside your project directory.

### Step 3: Activate the Virtual Environment

Activate the virtual environment before installing dependencies:

* **Windows:**

  ```bash
  .\venv\Scripts\activate
  ```
* **Mac/Linux:**

  ```bash
  source venv/bin/activate
  ```

Once activated, `(venv)` will appear at the beginning of your terminal prompt.

### Step 4: Install Dependencies

While the virtual environment is active, install all required libraries:

```bash
pip install -r requirements.txt
```

This will ensure all dependencies are properly installed.

---

## Running the Applications

Ensure your virtual environment is still active. Then, use the following commands:

* **To run V1 (Cross-Correlation):**

  ```bash
  streamlit run app.py
  ```

* **To run V2 (MFCC + DTW):**

  ```bash
  streamlit run app_v2.py
  ```

A new browser tab will automatically open with the application interface.

---

## Usage Instructions

1. **Upload Files:**
   In the sidebar, upload the main audio file (the *Target*) and the shorter clip to be searched (the *Template*).
   *Note:* For large audio files, `.mp3` format is recommended due to smaller file sizes and better browser compatibility.

2. **Adjust Parameters:**
   Use the sidebar sliders to configure detection settings.

3. **Run Analysis:**
   Click the **Analyze Audio** button to process the input.

4. **View Results:**
   The app will display a plot with detected matches highlighted, along with a list of timestamps where the phrase or word occurs.

---

