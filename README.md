# Electricity Price Forecasting Using Neural Networks: Motivation, Theory, and Next Steps

## 1. Motivation

The goal of this project is to forecast the Hourly Ontario Electricity Price (HOEP) using machine learning models, particularly neural networks. HOEP is highly volatile and affected by various factors like demand, dispatch forecasts, time of day, and more. Traditional methods like linear regression provide limited performance due to the non-linear nature of electricity pricing. This project aims to explore whether deeper architectures with temporal and contextual features can better capture the complex patterns in HOEP.

## 2. Data and Features

The dataset combines two public IESO sources: HOEP pricing data and zonal demand data. Features engineered so far include:  
Basic temporal context: hour, day of week, month  
Forecasted dispatch signals: Hour 1 Predispatch, OR 10 Min Sync, OR 30 Min  
Lag features: Previous HOEP values (1, 2, 3 hours ago), Previous Demand values (1, 2, 3 hours ago)  
Rolling features: 3-hour moving averages of HOEP and Demand  
These features are motivated by the temporal dependencies in electricity markets.

## 3. Models and Training

### 3.1 Linear Regression

Linear regression was used as a baseline. It was tested with just demand, then gradually enhanced with additional temporal and lag features. However, its inability to capture non-linear interactions led to limited performance improvements.

### 3.2 Neural Networks

Neural networks were trained using the TensorFlow/Keras library. Two main architectures were explored:  
Shallow NN: 32 -> 16 -> 1  
Deeper NN: 64 -> 32 -> 1  
All models used ReLU activation and MSE as the loss function.  
Training details:  
Epochs: 50  
Batch size: 32  
Validation split: 10%

## 4. Theory: Forward Propagation

### Current Mathematical Understanding: Forward Propagation

As of now, I understand the forward propagation step of a neural network as follows:

**Neurons compute weighted sums:**  
Each neuron in a layer takes inputs from the previous layer, multiplies them by a set of weights, adds a bias, and passes the result through an activation function.  
Mathematically, for layer \( l \):

\[
z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}
\]

**Activation functions are applied:**  
The output of the weighted sum is passed through a non-linear activation function to introduce non-linearity. In our neural network, we use ReLU:

\[
a^{[l]} = \text{ReLU}(z^{[l]}) = \max(0, z^{[l]})
\]

**Final output layer is linear:**  
The last layer in a regression model (like predicting HOEP) does not use any activation function. This allows the model to predict real-valued outputs:

\[
\hat{y} = W^{[L]} a^{[L-1]} + b^{[L]}
\]

**Loss function is MSE:**  
The model is trained to minimize the mean squared error (MSE) between predictions \( \hat{y} \) and actual targets \( y \):

\[
\text{MSE} = \frac{1}{m} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2
\]

I initially used MSE because it is standard for regression tasks and penalizes larger errors more heavily, aligning well with the goal of minimizing large fluctuations in electricity price forecasts.

**Flow of values:**  
Data flows from input through the layers using the above operations, producing a prediction. I currently do not yet understand how weights are updated (that happens in backpropagation), but I know forward propagation is the step that calculates predictions from input features.

## 5. Results Summary

| Model                               | RMSE (CAD/MWh) |
|------------------------------------|----------------|
| LR: Demand only                    | 26.17          |
| LR: + Datetime, Dispatch           | 46.62          |
| NN (32->16->1): + Dispatch         | 27.95          |
| NN (64->32->1): + Dispatch         | 31.79          |
| LR: + Prev_HOEP                    | 44.65          |
| NN (32->16->1): + Prev_HOEP        | 25.53          |
| NN (64->32->1): + Prev_HOEP        | 26.57          |
| NN (32->16->1): Multi-lag + MA     | 21.95          |
| NN (64->32->1): Multi-lag + MA     | 14.46          |

## 6. Observations

Neural networks outperform linear models as more lag and context are added.  
Larger networks show stronger performance, especially when paired with more temporal features.  
RMSE fluctuates slightly even with a random seed due to stochastic training behavior.

## 7. Future Work

- Add more seasonal and yearly historical data to improve generalization  
- Incorporate additional external signals (weather, gas prices, outages)  
- Experiment with LSTM or Temporal CNNs for sequential modeling  
- Evaluate generalization across years and regions (e.g., does a model trained on summer generalize to winter?)
