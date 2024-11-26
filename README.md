
# USED CAR PRICE PREDICTION
## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Project Overview](#project-overview)
3. [Objectives](#objectives)
4. [Design and Analysis](#design-and-analysis)
5. [Functional Requirements](#functional-requirements)
6. [Methodology](#methodology)
7. [Tools Used](#tools-used)
8. [Implementation: Dataset and Preprocessing](#implementation-dataset-and-preprocessing)
9. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
10. [Removing Outliers](#removing-outliers)
11. [Model Selection](#model-selection)
12. [Model Training](#model-training)
13. [User interface](#UserInterface )

## 1. Problem Statement

Most people waste time manually inquiring about the expected car prices within their social circles or using websites that provide price predictions. However, these predictions often lack accuracy due to missing data or incomplete feature specifications. The goal of this project is to provide a quick and accurate used car price prediction tool. Users can enter the car details, and in return, they will receive an estimated price. This application is designed to be user-friendly, requiring little technical knowledge.

## 2. Project Overview

This project uses machine learning techniques to predict the price of a used car based on various attributes such as car make, model, year of manufacture, mileage, condition, and more. The backend of the project is built using **Flask** for creating the web application, while **Jupyter Notebook** is used to develop and train the machine learning models. We will also use **Pickle** to serialize and load the trained models for deployment. The frontend utilizes **HTML** and **CSS** to structure and style the web page, while **JSON** is used for data exchange between the client and server.

## 3. Objectives

- **Data Preprocessing**: Clean and prepare data to be suitable for modeling.
- **Exploratory Data Analysis (EDA)**: Understand the distribution and relationships of data points.
- **Modeling**: Implement various machine learning regression models such as Lasso, Linear Regression, Decision Tree, and Random Forest to predict used car prices.
- **Web Deployment**: Integrate the trained model into a web-based application using Flask and Pickle for serialization.
  
## 4. Design and Analysis

The design phase focuses on fulfilling critical project requirements such as functional, data, usability, and aesthetic aspects. By meeting these requirements, we ensure that the end-users find the application accessible and comfortable. The project design outlines how the features and functionalities will align to meet the goal of providing a reliable used car price prediction tool.

### Key Design Components:
- **User Interface (UI)**: Clean, minimalistic, and easy-to-use interface for inputting car details.
- **Model**: A robust machine learning model that can predict car prices based on various features.
- **Scalability**: The model will handle multiple requests from users and provide quick responses.

## 5. Functional Requirements

The application allows users to:
- Enter car details (e.g., car price in lakhs, transmission type, fuel type, year of manufacture).
- Receive a predicted car price based on the entered attributes.
- Make repeated predictions without interference.
- The web application ensures that the user can input and modify car attributes and see predictions instantly.

## 6. Methodology

The methodology is based on several key steps to ensure that the used car price prediction model is effective and scalable:

1. **Data Collection**: Gather car-related data from online sources like Kaggle, focusing on attributes such as car make, model, year, mileage, price, etc.
2. **Data Preprocessing**: Clean the dataset by handling missing values, encoding categorical variables, and scaling numerical features.
3. **Exploratory Data Analysis (EDA)**: Visualize and explore relationships in the dataset to understand patterns and outliers.
4. **Feature Engineering**: Create new features or transform existing ones to improve the model’s predictive power.
5. **Model Selection**: Choose and implement appropriate machine learning models.
6. **Model Training and Evaluation**: Train the models on the training data, evaluate them on testing data, and choose the best model based on performance metrics.
7. **Deployment**: Deploy the best-performing model into a web-based application using Flask.

## 7. Tools Used

- **Anaconda**: Python distribution for scientific computing, used for managing environments and libraries.
- **Jupyter Notebook**: Interactive coding environment for developing and testing machine learning models.
- **Flask**: Web framework used for creating the backend of the application.
- **Pickle**: Serialization library to save and load machine learning models.
- **HTML & CSS**: For frontend development to create the structure and styling of the web application.
- **JSON**: For data exchange between server and client.

## 8. Implementation: Dataset and Preprocessing

The dataset used for this project is a collection of used car data available from Kaggle. The dataset includes multiple features such as:
- Make, model, year of manufacture
- Fuel type, transmission type
- Kilometers driven, owner details
- Price and other specifications

### Preprocessing Steps:
- Removed irrelevant columns (e.g., seller).
- Checked for and handled missing values.
- Converted categorical data into numerical format using encoding techniques.
- Normalized numerical features to make them comparable.
- Added new features like `selling_price_inr` for better price prediction.

## 9. Exploratory Data Analysis (EDA)

EDA helps uncover insights and relationships between various attributes. Here’s what was performed during this phase:
- **Visualizations**: Used plots like pair plots and histograms to understand distributions.
- **Comparisons**: Analyzed manual vs automatic cars, and how various attributes (e.g., transmission type) affected the price.
- **Outlier Detection**: Identified and visualized outliers in key columns like selling price and kilometers driven.

## 10. Removing Outliers

Outliers in the dataset, especially in columns like `Selling Price` and `Kilometers Driven`, were removed using the Interquartile Range (IQR) method. This ensured that extreme data points did not distort the model training.


## 11. Model Selection and Model Training

### Machine Learning Models:
- **Linear Regression**: A basic regression model to understand the relationship between the car features and selling price.
- **Decision Tree Regressor**: A tree-based model to predict the price based on feature splits.
- **Random Forest Regressor**: An ensemble of decision trees that improves prediction accuracy by combining multiple trees.

### Model Training:
- The dataset was split into training and testing sets.
- Trained the models using the training set (`x_train`, `y_train`).
- Evaluated the models using the testing set (`x_test`, `y_test`), measuring accuracy with metrics like Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
  
### Hyperparameter Tuning:
Used techniques like grid search to tune the model parameters and optimize performance.

---

This README provides a comprehensive overview of the used car price prediction project. It includes the project's problem statement, methodology, data preprocessing steps, model selection, and training process, as well as the tools used and functional requirements of the web application.

# # User Interface (UI)
The User Interface (UI) of this application is designed to be simple, clean, and user-friendly. The interface is built using HTML, CSS, and the Bootstrap framework to ensure it is responsive and visually appealing across all devices.
Landing Page: The landing page provides an introduction to the project and includes a button that directs users to the prediction form.


Prediction Form: Users can input car details through an intuitive form. The fields include:

Make (e.g., Toyota, BMW)
Model
Year of Manufacture
Mileage
Fuel Type (e.g., Petrol, Diesel)
Transmission Type (e.g., Automatic, Manual)
The form uses dropdowns, text input fields, and sliders to make data entry straightforward.


Price Prediction Output: After submitting the form, the user sees a predicted price displayed on the same page. The result is shown in a highlighted box with a message, providing clarity to the user about the estimated price of the car.


Responsiveness: The UI is fully responsive, meaning it adjusts seamlessly across different screen sizes (desktop, tablet, and mobile devices), ensuring a smooth user experience.
![Screenshot 2024-05-28 171853](https://github.com/user-attachments/assets/68753fae-70eb-45d8-be6f-48c250b5ccec)

