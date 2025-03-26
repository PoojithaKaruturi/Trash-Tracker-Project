

TrashTracker: Waste Classification and Quantification

Project Overview

TrashTracker is an AI-powered waste management application designed to improve efficiency in waste sorting and recycling processes. This project leverages machine learning and computer vision to classify and quantify various types of waste in mixed waste scenarios, helping to streamline waste separation in dumpyards.

Features

Waste Classification: Identifies different types of waste such as metal, glass, cardboard, etc.
Quantification: Calculates the percentage of each waste type in a mixed waste image.
User-Friendly Interface: A web application built using Django with a Bootstrap front-end for easy interaction.

Technologies Used

Backend: Django
Frontend: Bootstrap, HTML, CSS
Machine Learning: TensorFlow, CNNs for image classification
Dataset: Custom synthetic dataset derived from TrashNet

 Installation

To run this project locally, follow these steps:

1. Clone the Repository
   ```bash
   git clone https://github.com/your-username/TrashTracker.git
   ```

2. Navigate to the Project Directory
   ```bash
   cd TrashTracker
   ```

3. Install Dependencies
   Create a virtual environment and install the required Python packages:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

4. Set Up the Database
   Run migrations to set up the database:
   ```bash
   python manage.py migrate
   ```

5. Run the Development Server
   Start the Django development server:
   ```bash
   python manage.py runserver
   ```

Now, you can open your browser and go to `http://127.0.0.1:8000/` to access the application.

---

