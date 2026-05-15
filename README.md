# PlaceAI - Student Placement Predictor

A premium, AI-powered SaaS dashboard that predicts student placement probability using Machine Learning. This application features a modern Glassmorphism UI, real-time probability visualization, and personalized career roadmaps.

## Live Demo

[View the deployed app](https://student-placement-predictor-1yne.onrender.com/)

## Features

- **Smart Resume Analysis**: Upload your PDF resume to automatically populate the form. Uses intelligent text extraction to identify skills, CGPA, internships, and projects.
- **AI-Powered Predictions**: Uses a Random Forest Classifier trained on student records.
- **Modern UI/UX**: Dark-themed dashboard with sleek animations and a two-column interactive layout.
- **Real-time Probability**: Animated radial progress chart showing exact placement chances.
- **Feature Importance**: Dynamic horizontal bar charts showing which factors influenced your result most.
- **Smart Feedback**: Automatically identifies weak points (CGPA, Skills, etc.) and provides actionable tips.
- **Roadmap Integration**: Personalized resource links for every identified area of improvement.

## Tech Stack

- **Backend**: Python, Flask
- **Libraries**: pdfplumber (Resume Parsing), Scikit-learn, Pandas, NumPy
- **Frontend**: HTML5, CSS3 (Glassmorphism, CSS Variables, Keyframe Animations)
- **Model**: Random Forest (200 Estimators)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/jayshreek2511-cloud/Student-Placement-predictor.git
   cd placement-predictor
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (optional - model.pkl is included):
   ```bash
   python train.py
   ```

## How to Run

1. Start the Flask server:
   ```bash
   python app.py
   ```

2. Open your browser and navigate to:
   ```text
   http://127.0.0.1:5000
   ```

## Project Structure

- `app.py`: The main Flask application (Backend + Frontend Templates).
- `train.py`: Machine learning training script using RandomForest.
- `test_model.py`: Script to test model predictions with sample data.
- `model.pkl`: Serialized model and encoders.
- `train.csv`: The dataset used for training.
- `importance.png`: Feature importance visualization generated during training.
- `requirements.txt`: Python dependencies.

## Model Factors

The AI evaluates your profile based on:

- **Academic Metrics**: CGPA, Branch, Degree, Backlogs.
- **Technical Skills**: Coding Skills, Projects, Certifications.
- **Professional Skills**: Internships, Aptitude Score, Communication, Soft Skills.

---

Built for students to bridge the gap between education and employment.
