
# # USED CAR PRICE PREDICTION

#PROBLEM STATEMENT 
Most people waste their time inquiring about the expected car price in and around their friend’s 
circle and their associates manually. Even some websites can predict the price but it is not very 
accurate due to the unavailability of feature data and Specifying them as the NULL value in the 
dataset or just dropping the feature column. With this project, in no time people can access the 
website and insert their requirements, and can get a predicted price of the car. Indeed, people 
who have very little technical knowledge can be able to reach the site. 

1. Table of Contents
2. Project Overview
3. Installation
4. Technologies Used
5. Features
6. Usage
7. Model Selection & Training

# Project Overview
This project is structured into two main components:

Backend (Machine Learning Model): A regression-based model that predicts used car prices. The model uses popular machine learning algorithms like Linear Regression, Decision Tree, Random Forest, and Lasso Regression to predict the price based on input features.

Frontend (Web Interface): A Flask-based web application that allows users to input car details and receive a price prediction. The interface is built to be simple and easy to use, guiding users through the process of inputting necessary car attributes.

# Installation
Prerequisites

Python 3.x
Anaconda (for dependency management)
# Steps to Install
 1. Clone the Repository:
   git clone https://github.com/your-username/used-car-price-prediction.git cd used-car-price-prediction

 2. Set Up a Virtual Environment: Create and activate a virtual environment to manage dependencies:
   conda create --name car-price-prediction python=3.8
   conda activate car-price-prediction
 3. Install Required Libraries: Install the necessary Python packages by running:
   pip install -r requirements.txt
 4. Run the Application: Start the Flask web application:
   python app.py
Now, you can access the application through your browser at http://127.0.0.1:5000/.

# Technologies Used
This project utilizes several key technologies:

Backend: Python, Flask
Machine Learning: Scikit-learn (including Lasso, Linear Regression, Decision Tree, Random Forest Regressors)
Frontend: HTML, CSS, Bootstrap for styling the web interface
Data Analysis: Pandas for data manipulation, NumPy for numerical computations, Matplotlib and Seaborn for visualizations
Dataset: The model is trained using a dataset of used car listings, cars_details.csv, sourced from Kaggle.
Features
Car Price Prediction: Users can input specific details of a used car, including make, model, year, mileage, fuel type, and more, and the app will predict the car’s price.

Multiple Machine Learning Models: The system uses multiple regression models to predict prices, giving a reliable output by comparing results from different models.

Simple Web Interface: The user interface is built using Flask, HTML, and Bootstrap, making it easy for users to interact with the app and get quick results.

Real-time Results: As soon as the user submits the car details, the predicted price is displayed instantly.


Input Car Details: In the input form, fill in the required car attributes, such as:
Make
Model
Year of manufacture
Mileage
Fuel type (Petrol, Diesel, etc.)
Transmission type (Manual or Automatic)
Get the Prediction: After submitting the form, the predicted car price will appear on the screen.

Model Selection & Training
This project uses several machine learning algorithms to predict the price of used cars:

Linear Regression: A simple linear approach that predicts the price based on the linear relationship between car features.

Decision Tree Regression: A non-linear regression method that splits data into subsets based on decision rules, making it useful for complex relationships between features.

Random Forest Regression: An ensemble method that uses multiple decision trees to improve accuracy and reduce overfitting. It is particularly robust in real-world data.

Lasso Regression: A linear regression technique that uses L1 regularization to handle high-dimensional data and avoid overfitting,
and other models also.

The model is trained on a dataset of used cars with various features. It is evaluated using standard metrics like Mean Squared Error (MSE) and R-squared to measure prediction accuracy.
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

