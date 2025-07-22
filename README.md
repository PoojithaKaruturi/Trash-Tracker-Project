# Trash-Tracker-Project
TrashTracker: Waste Classification and Quantification

Project Overview

TrashTracker is an AI-powered waste management application designed to improve efficiency in waste sorting and recycling processes. This project leverages machine learning and computer vision to classify and quantify various types of waste in mixed waste scenarios, helping to streamline waste separation in dumpyards.

Features

Waste Classification: Identifies different types of waste such as metal, glass, cardboard, etc. Quantification: Calculates the percentage of each waste type in a mixed waste image. User-Friendly Interface: A web application built using Django with a Bootstrap front-end for easy interaction.

Technologies Used

Backend: Django Frontend: Bootstrap, HTML, CSS Machine Learning: TensorFlow, CNNs for image classification Dataset: Custom synthetic dataset derived from TrashNet

Installation

To run this project locally, follow these steps:

Clone the Repository

git clone git clone https://github.com/your-username/TrashTracker.git

Navigate to the Project Directory

cd TrashTracker
Install Dependencies Create a virtual environment and install the required Python packages:

python -m venv venv
source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
Set Up the Database Run migrations to set up the database:

python manage.py migrate
Run the Development Server Start the Django development server:

python manage.py runserver
Now, you can open your browser and go to http://127.0.0.1:8000/ to access the application.

