# Naive Bayes vs BERT on Emotion Classification

**Description:** This project compares the performance of a Naive Bayes model and a BERT model on emotion classification from text. 
- Using a emotion classification dataset from huggingface (https://huggingface.co/datasets/dair-ai/emotion), the models are trained and evaluated on the dataset.
- a comprehensive grid search for Naive Bayes is performed to find the best hyperparameters which resulted in an accuracy of 0.8385.
- the pre-trained BERT model is used in an out-of-the-box or fine-tuned (just head or full model) approach with the best one achieving an accuracy of 0.9320.

`report.pdf` contains a description of all our experiments and results.

## Installation
Before running the project, you need to set up the required environment. Follow these steps:

**1. Clone the Repository:**
```
git clone https://github.com/jantiegges/emotion-classification-from-text.git
cd emotion-classification-from-text
```
**2. Create a Virtual Environment (Optional but Recommended):**
```
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```
**3. Install Dependencies:**
```
pip install -r requirements.txt
```

## Usage
To use this project, follow these steps:

**1. Run Jupyter Notebooks:**
* Launch Jupyter Notebook in the project directory:
```
jupyter notebook
```
* Open the relevant Jupyter notebooks, such as:
  - `experiments.ipynb` - contains all of the experiments
  - `data_analysis.ipynb` - contains data analysis
  
**2. Explore the Code:**
* Review the codebase:
  - `models/` - contains the naive bayes and BERT models
  - `utils/` - contains data preprocessing functions
  - `main.py` - contains functions for running grid search and experiments
 
**3. Customize and Experiment:**
* Feel free to customize parameters and experiment with the code.
* Note any additional instructions provided within the notebooks.