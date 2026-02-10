
# 📸 Facial Recognition Application

## 📌 Project Overview
This is a **Facial Recognition Application** built using:
- **Kivy** for the Graphical User Interface (UI)
- **OpenCV** for computer vision to capture and compare real-time facial data with a stored database
- **Siamese Neural Network (SNN)** for accurate face verification

The system performs real-time face detection, feature extraction, and verification against stored faces with high accuracy and minimal false positives.

---

## ✨ Features
- **Real-Time Face Capture** – Detects and captures faces from a live camera feed.
- **Face Matching** – Compares captured face embeddings with stored database entries.
- **Deep Learning (Siamese Network)** – Improves verification accuracy and reduces false positives.
- **Interactive UI** – Built with Kivy for a clean and user-friendly experience.

---

## 🛠️ Tech Stack
- **Python**
- **Kivy** (UI)
- **OpenCV** (Computer Vision)
- **TensorFlow / Keras** (Deep Learning)

---

## 🧠 Model Training Details

The Siamese Neural Network (SNN) was trained using a combination of **custom face images** and a **public benchmark dataset** for robust negative sampling.

### 📊 Training Data Composition

- **Anchor Images:** ~3,000  
- **Positive Images:** ~3,000  
- **Negative Pairs:** ~13,000  

### 🗂️ Negative Dataset Source

- **LFW (Labeled Faces in the Wild)** dataset was used to generate **negative pairs**.
- LFW provides faces captured under unconstrained conditions, which helps the model:
  - Learn strong dissimilarity representations
  - Generalize better to unseen identities
  - Reduce false positives in real-world scenarios
  
---

## 📥 Model Download
The trained **Siamese Neural Network model** is not included in the repository due to large file size.  
Download it from the link below and place it in the `model/` directory:

🔗 **[Download Siamese Model](https://drive.google.com/file/d/1w7290K6fZXck_gaOuSFgIOzfDNYYRV85/view?usp=sharing)**

---

## 📁 Dataset / Application Data Setup (Required)

Before running the application, you must **manually create an `application_data` folder** inside the main project directory (`FaceRecogApp`).

Inside the `application_data` folder:

- Create an **`input_image`** folder  
  - This folder stores the face image captured from the live camera during verification.

- Create a **`verification_images`** folder  
  - Inside this folder, create **one subfolder per user** (for example: `user1`, `user2`, etc.).
  - Each user folder should contain **25–50 face images** of that person.

### 📸 Dataset Guidelines

- Use **25–50 images per user** for better verification performance.
- Include:
  - Good lighting conditions
  - Slight variations in face angle
  - Minor changes in facial expressions
- Greater variation improves robustness and reduces false positives.

> ⚠️ The `application_data` folder is intentionally excluded from Git version control to prevent uploading personal or large data.

---

## 📚 Reference
I have taken reference from the **Siamese Model PDF** uploaded in this repository while designing and implementing the model.

---

## 🚀 How to Run
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/facial-recognition-app.git
   cd Face-Recognition/FaceRecogApp
   
2. **Create and Activate Virtual Environment (Recommended)**
   Create a virtual environment to isolate project dependencies:
   ```bash
   python -m venv faceidenv

3.  **Activate the virtual environment (Windows):**
    ```bash
    faceidenv\Scripts\activate
   Once activated, the virtual environment name should appear in your terminal.

4. **Install Dependencies**
   Install all required Python packages using the requirements.txt file:
     ```bash
     pip install -r requirements.txt

6. **Run the Application**
   After completing all setup steps, start the application using:
     ```bash
     python faceid.py

---

## 👤 Author
Atharva Dharmadhikari
