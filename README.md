# Electricity Price Forecasting Using Neural Networks

This project investigates forecasting the Hourly Ontario Electricity Price (HOEP) using machine learning models, with a focus on neural networks. The goal is to model the complex, non-linear dynamics of HOEP using temporal, dispatch, and lag-based features, and compare performance against baseline linear models.

## Progress Summary

The notebook below documents the initial stage of the project. It includes motivation, feature engineering, a theoretical understanding of forward propagation, early model architectures, and preliminary RMSE results.

- [`HOEP_Modeling_Summary.ipynb`](./HOEP_Modeling_Summary.ipynb)

This is a **work in progress**, and further experiments, features, and model improvements are ongoing.

## Repository Structure

- `main_model_training.ipynb`: Implementation of data preprocessing, feature engineering, model training, and evaluation.
- `HOEP_Modeling_Summary.ipynb`: Progress summary of modeling pipeline and initial results.
- `data/`: Contains raw or processed data (not version-controlled if private).
- `models/`: Directory for saved models (if applicable).
- `README.md`: Project overview and documentation.

## Preliminary Results (RMSE)

| Model                               | RMSE (CAD/MWh) |
|------------------------------------|----------------|
| LR: Demand only                    | 26.17          |
| LR: + Datetime, Dispatch           | 46.62          |
| NN (32→16→1): + Dispatch           | 27.95          |
| NN (64→32→1): + Dispatch           | 31.79          |
| LR: + Prev_HOEP                    | 44.65          |
| NN (32→16→1): + Prev_HOEP          | 25.53          |
| NN (64→32→1): + Prev_HOEP          | 26.57          |
| NN (32→16→1): Multi-lag + MA       | 21.95          |
| NN (64→32→1): Multi-lag + MA       | 14.46          |

## Environment

- Python 3.10+
- TensorFlow 2.x
- NumPy, Pandas, scikit-learn, Matplotlib

## Planned Work

- Add external signals (weather, gas prices, outages)
- Integrate more historical and seasonal data
- Experiment with sequential models (LSTM, Temporal CNNs)
- Evaluate model generalization across seasons and years
