# Linear-Regression-Model-for-TV-Advertising-Sales-Prediction
This repository contains a Python script that builds and evaluates a linear regression model to predict sales based on TV advertising budget data. The dataset used in this project is a CSV file named tv_dataset.csv.

Table of Contents

Introduction

Dataset

Installation

Usage

Model Evaluation

Prediction

## Introduction
This project demonstrates the process of building a simple linear regression model using Python. The objective is to predict sales based on the amount spent on TV advertising. The steps include data acquisition, preprocessing, model building, evaluation, and making predictions on new data.

## Dataset
The dataset (tv_dataset.csv) contains two columns:

TV: Advertising budget spent on TV (in thousands of dollars)

Sales: Sales generated (in thousands of units)

## Installation
To run this script, you need to have Python installed along with the following libraries:

numpy

pandas

matplotlib

seaborn

scikit-learn

## Usage
Clone the repository:

git clone https://github.com/your-username/tv-advertising-sales-prediction.git

Navigate to the project directory:

cd tv-advertising-sales-prediction

Ensure that you have the dataset tv_dataset.csv in the project directory.

Run the script:

python tv_sales_prediction.py

## Model Evaluation
The script performs the following tasks:

Displays the number of records and features in the dataset.

Shows feature names and information about the dataset.

Provides numerical description of the dataset.

Checks for missing values.

Visualizes the relationship between TV advertising budget and sales using scatter plots.

Splits the dataset into training and testing sets (80:20 ratio).

Builds a linear regression model.

Predicts sales for the test set and compares actual vs. predicted values.

Visualizes the trained model with a fitted line.

Evaluates the model using Mean Squared Error (MSE).

Visualizes prediction errors.

## Prediction

The script also demonstrates how to make a prediction for a new unseen record. For example, it predicts sales for a TV advertising budget of $150,000.
