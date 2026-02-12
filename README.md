# Sign Language MNIST — Translation Engine

A translation engine that takes a static image of a hand sign (A–Z) and predicts the letter. Built with a classical ML baseline (SVM/Random Forest) and a neural network (PyTorch/TensorFlow), with a local dashboard for inference.

## Setup

### 1. Download the data

You need to download the Sign Language MNIST dataset and place it in the `data` folder.

**Using Kaggle CLI:**

```bash
kaggle datasets download -d datamunge/sign-language-mnist
```

Then unzip and move the contents into the project’s `data` directory (e.g. create `data/sign-language-mnist/` and put the CSV files there, or unzip directly into `data/`).

**Note:** You must have the [Kaggle API](https://github.com/Kaggle/kaggle-api) set up (API key in `~/.kaggle/kaggle.json`) to use the command above.

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

### 3. Activate the virtual environment

**macOS / Linux:**

```bash
source .venv/bin/activate
```

**Windows (Command Prompt):**

```cmd
.venv\Scripts\activate.bat
```

4. Install dependencies

With the virtual environment activated:

```bash
pip install -r requirements.txt
```

## Project structure

- `baseline_model/` — Classical ML model (SVM or Random Forest) and training notebook
- `data/` — Sign Language MNIST data (download and place here)
- `dashboard.py` — Local dashboard for uploading hand sign images and viewing predictions

## Usage

After setup, you can train the baseline model (see `baseline_model/model.ipynb`) and run the dashboard (e.g. `streamlit run dashboard.py` if using Streamlit).
