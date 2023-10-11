# Laptop-Price-Prediction
Predict laptop prices using machine learning and deep learning. This project employs Python, Pandas, Scikit-Learn, and Keras to create models for accurate price predictions. Whether you're a data science enthusiast or a laptop shopper, explore the code and insights into laptop pricing trends.


# Laptop Price Prediction using Machine Learning and Deep Learning

This repository contains code for predicting laptop prices using both traditional machine learning (Linear Regression) and deep learning (Neural Networks) approaches. The dataset used for this project is `amazon_laptop_prices_v01.csv`.

## Overview

- The project starts with data preprocessing and feature engineering using Python libraries like Pandas and Scikit-Learn.
- Categorical features in the dataset are label encoded to convert them into numerical values.
- Missing values in the 'rating' column are filled with the mean rating.
- The data is split into training and testing sets using `train_test_split`, and feature scaling is performed using `MinMaxScaler`.
- Linear Regression is applied as the first machine learning model to predict laptop prices.
- A neural network model is built using Keras with TensorFlow backend to predict laptop prices. The architecture consists of multiple dense layers.
- Early stopping is used during training to prevent overfitting and restore the best weights.
- The training and validation results are visualized using scatter plots and a 45-degree line to assess model performance.

## Dependencies

- Python 3.x
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Keras (with TensorFlow backend)

## Running the Code

1. Make sure you have all the required dependencies installed in your Python environment.

2. Clone this repository to your local machine or download the code files.

3. Place your dataset as `amazon_laptop_prices_v01.csv` in the same directory as the code files.

4. Execute the Jupyter Notebook or Python script to run the code. You can follow the step-by-step instructions provided in the code.

## Results

- The Linear Regression model's performance can be assessed using the R-squared score.
- The neural network model's performance can be observed through scatter plots and the validation loss graph.

## Issues and Enhancements

- If you encounter any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

  
### Training and Validation Results Visualization
This section of the project is dedicated to visually representing the training and validation results using images. The code leverages the Matplotlib library to create informative visualizations of the model's performance.

## Results Visualization
The primary output of this code is a pair of scatter plots that display the model's predictions in comparison to the actual values. Here's how to interpret these plots:

# Training Results:

![image](https://github.com/LukaNdr/Laptop-Price-Prediction/assets/147658141/e8f9437d-6dce-4d0f-a16b-f6f05778177a)


In this scatter plot, each point represents a data point from the training set. The x-axis corresponds to the actual laptop prices, while the y-axis displays the prices predicted by the model during training. The red line represents the ideal scenario where predictions perfectly match actual prices.

# Validation Results:

![image](https://github.com/LukaNdr/Laptop-Price-Prediction/assets/147658141/81df97a4-8bb0-419f-8e59-da559e55a793)


Similar to the training results, this scatter plot focuses on the validation set. It illustrates the relationship between actual laptop prices and the model's predictions. Again, the red line represents the perfect prediction scenario.
