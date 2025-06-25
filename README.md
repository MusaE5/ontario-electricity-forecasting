#  Electricity Price Forecasting Using Neural Networks: Motivation, Theory, and Next Steps

---

## 1. Motivation

The goal of this project is to forecast the **Hourly Ontario Electricity Price (HOEP)** using machine learning models, particularly neural networks.  
HOEP is highly volatile and influenced by demand, dispatch forecasts, time of day, and more. Traditional linear regression models are limited in their ability to capture the non-linear patterns of electricity pricing.

This project explores whether **deeper neural architectures** combined with temporal and contextual features can more effectively model these dynamics.

---

## 2. Data and Features

The dataset merges two public IESO sources: HOEP pricing and zonal demand data.  
Features engineered so far include:

- **Basic temporal context:** hour, day of week, month  
- **Forecasted dispatch signals:** Hour 1 Predispatch, OR 10 Min Sync, OR 30 Min  
- **Lag features:** previous HOEP and demand values (1, 2, 3 hours ago)  
- **Rolling features:** 3-hour moving averages of HOEP and demand  

These features reflect the temporal dependencies present in electricity markets.

---

## 3. Models and Training

### 3.1 Linear Regression
- Used as a baseline.
- Tested with demand only, then with added temporal and lag features.
- Could not effectively model non-linear dynamics, leading to limited performance.

### 3.2 Neural Networks
- Built using **TensorFlow/Keras**.
- Two main architectures:
  - Shallow: `32 → 16 → 1`
  - Deeper: `64 → 32 → 1`
- All layers used ReLU activations; output layer used no activation for regression.
- Loss function: **Mean Squared Error (MSE)**  
- Training settings:
  - Epochs: 50  
  - Batch size: 32  
  - Validation split: 10%

---

## 4. Theory: Forward Propagation

### Current Mathematical Understanding

**Step 1: Weighted sum**  
For layer \\( l \\):

\\[
z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}
\\]

**Step 2: Activation function (ReLU)**  
\\[
a^{[l]} = \text{ReLU}(z^{[l]}) = \max(0, z^{[l]})
\\]

**Step 3: Output layer (Linear)**  
\\[
\hat{y} = W^{[L]} a^{[L-1]} + b^{[L]}
\\]

**Step 4: Loss Function (MSE)**  
\\[
\text{MSE} = \frac{1}{m} \sum_{i=1}^{m} \left( y^{(i)} - \hat{y}^{(i)} \right)^2
\\]

Forward propagation computes predictions from input features.  
I do not yet fully understand backpropagation, but I know that forward propagation passes data through layers to produce a prediction.

---

## 5. Results Summary

| Model                                 | RMSE (CAD/MWh) |
|--------------------------------------|----------------|
| LR: Demand only                      | 26.17          |
| LR: + Datetime, Dispatch             | 46.62          |
| NN (32→16→1): + Dispatch             | 27.95          |
| NN (64→32→1): + Dispatch             | 31.79          |
| LR: + Prev_HOEP                      | 44.65          |
| NN (32→16→1): + Prev_HOEP            | 25.53          |
| NN (64→32→1): + Prev_HOEP            | 26.57          |
| NN (32→16→1): Multi-lag + MA         | 21.95          |
| NN (64→32→1): Multi-lag + MA         | 14.46          |

---

## 6. Observations

- Neural networks **outperform** linear regression when given lag and rolling context.
- Larger NNs perform better when fed richer temporal features.
- RMSE fluctuates slightly due to stochastic elements in training, even with a seed.

---

## 7. Future Work

- Add more seasonal and multi-year data for better generalization.
- Integrate **external factors** like weather, gas prices, or outage events.
- Explore **LSTM** or **Temporal CNNs** for better sequence modeling.
- Test generalization across seasons (e.g., training on summer, testing on winter).

---
