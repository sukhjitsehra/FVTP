# Vehicle Trajectory Prediction Using LSTM Fusion Models (FTVP)
This project implements and evaluates two LSTM-based fusion strategies (Early and Late Fusion) for predicting vehicle trajectories using spatio-temporal data from the NGSIM US-101 highway dataset.

## 🚗 Project Overview
### Pipeline:

1. Data Ingestion & Preprocessing:
* Raw NGSIM vehicle trajectory CSV data.
* Cleaning and converting to spatio-temporal sequences.

2. Dataset Creation:
* Generating training and test sets with surrounding vehicle context.
* Spatial indexing to identify neighbor vehicles.

3. Model Training and Evaluation:
* Early Fusion: LSTMFusion.
* Late Fusion: LSTMLateFusion.
* Metrics: MSE, MAE, R², Accuracy.

4. Results Visualization:
* Comparative analysis of model performance.


## Project Structure

```
FTVP/
├── fusili/
│   ├── fusionmodels/
        ├── base_model.py          # base model from fusilli
│   │   └── lstm_fusion.py         # Fusion LSTM models
│   ├── tabularfusion/
│   │   └── train_lstm.py          # Main training script
│   └── data.py                    # Data loading & preprocessing logic
├── data/
│   └── ngsim_subset.csv           # NGSIM Dataset subset used for training
├── images/
│   ├── architecture.png           # Model architecture diagram
│   ├── results.png                # Experimental results and visualizations
│   └── pipeline.png               # Data pipeline illustration
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```



## 📂 File Structure:
| File/Folder             | Description                                    |
| ----------------------- | ---------------------------------------------- |
| `fusilli/`              | Main codebase extended from Fusilli repository |
| `data.py`               | Data preprocessing & dataset creation          |
| `train_lstm.py`         | Model training & evaluation script             |
| `lstm_fusion.py`        | Early and Late fusion LSTM model architectures |
| `data/ngsim_subset.csv` | Subset of the NGSIM dataset for experiments    |
| `images/`               | Diagrams and visualization results             |

## ⚙️ Installation

1. Clone Repository
```
git clone https://github.com/your-username/FTVP.git
cd FTVP
```
2. Set Up Environment
```
python -m venv venv
source venv/bin/activate    # Linux/MacOS
.\venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

## 📊 Running Experiments
1. Data Preparation and Model Training

Run the main training script:
python fusilli/train_lstm.py

This script will:

* Load and preprocess the dataset.

* Train LSTM fusion models.

* Evaluate performance using provided metrics.

## 🖥️ Evaluation Metrics:

* Mean Squared Error (MSE): Measures prediction accuracy by squaring the error.

* Mean Absolute Error (MAE): Average absolute prediction error.

* Coefficient of Determination (R²): Indicates variance explained by the model.

* Accuracy: Threshold-based positional accuracy.

## 📈 Results and Visualizations

After training, results and visualizations are saved in the images/ folder:

* architecture.png: Model structure visualization.

* pipeline.png: Data pipeline steps.

* results.png: Performance comparison graphs.

📦 Dependencies:

* Python 3.8+

* PyTorch

* NumPy

* Pandas

* Matplotlib

* scikit-learn

Full dependencies listed in requirements.txt

## 📝 Contributions

Contributions are welcome! Please open an issue or submit a pull request.

## 📜 License

This project is licensed under the MIT License.

## Dataset
NGSIM US-101 Highway dataset subset (ngsim_subset.csv): Used for model training and evaluation. Contains vehicle trajectories, speeds, acceleration, and positional data.