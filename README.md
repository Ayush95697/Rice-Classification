GrainPalette: Rice Type Classification

📌 Project Overview

Rice is a staple crop, and identifying different types of rice is crucial for efficient farming and production. Farmers often struggle to distinguish between rice varieties, which affects their water, manure, and cultivation strategies.

🔹 What does this project do?An AI-powered rice classification model that allows users to upload an image of a rice grain and receive predictions on its type. The model can identify up to five different types of rice using Convolutional Neural Networks (CNNs) with Transfer Learning.

🔹 Who can benefit?✅ Farmers✅ Agricultural scientists✅ Home farmers & gardeners✅ Agribusiness professionals

📊 Dataset

The model is trained on the Rice Image Dataset sourced from Kaggle.

Dataset Details

Description

📁 Source

Kaggle

🏷️ Classes

5 Rice Types

🖼️ Data Type

Image

🛠️ Technologies Used

Technology

Purpose

🐍 Python

General programming

🔬 TensorFlow & Keras

Deep Learning

📊 Scikit-learn

Evaluation Metrics

🖼️ OpenCV & PIL

Image Processing

📈 Matplotlib

Data Visualization

🚀 Installation & Setup

1️⃣ Clone the repository

git clone https://github.com/your-username/GrainPalette_Rice_Type_Classification.git
cd GrainPalette_Rice_Type_Classification

2️⃣ Install dependencies

pip install -r requirements.txt

3️⃣ Download the dataset (if not included)

import kagglehub
path = kagglehub.dataset_download("muratkokludataset/rice-image-dataset")
print("Dataset downloaded at:", path)

4️⃣ Run the Jupyter Notebook

jupyter notebook GrainPallete_Rice_Type_Classification.ipynb

📈 Model Training & Evaluation

✔️ Uses TensorFlow Hub for pre-trained CNN models✔️ Includes data augmentation and early stopping for better generalization✔️ Evaluates performance using Confusion Matrix & Classification Report

🎯 Usage

1️⃣ Upload an image of a rice grain2️⃣ The model will classify the rice type based on its visual features3️⃣ Output includes a Predicted Label and Confidence Score

📊 Results & Accuracy

🏆 Achieves high accuracy in classifying different rice types using transfer learning techniques.✅ Confusion Matrix & Metrics included in the notebook.✅ Further improvements can be made using Hyperparameter Tuning and Larger Datasets.

🔮 Future Enhancements

🔹 Deploying the model as a web app for easy access🔹 Expanding the dataset for better generalization🔹 Integrating Explainable AI (XAI) features to improve trust in predictions

👥 Contributors

Kunal Goel<br>
Ayush Mishra<br>
Aashish<br>
Abjeet Singh<br>

📜 License

This project is open-source and available under the MIT License.

