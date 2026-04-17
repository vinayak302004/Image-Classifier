# 🚀 Image Classifier (FastAPI + PyTorch)

This project is a simple **Image Classification Web App** built using:

* PyTorch (ResNet18)
* FastAPI (Backend)
* HTML, CSS, JavaScript (Frontend)

It classifies images into 10 classes from the CIFAR-10 dataset.

---

## 📸 Features

* Upload image from frontend
* Real-time prediction
* Confidence score display
* Image preview
* FastAPI backend API

---

## 🧠 Model

* Architecture: ResNet18
* Dataset: CIFAR-10
* Trained on subset (2000 images)
* Transfer Learning used

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/vinayak302004/Image-Classifier.git
cd Image-Classifier
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Run the Project

### Step 1: Train Model

```bash
python train_and_predict.py
```

### ✅ Step 1: Start Backend

```bash
python -m uvicorn app:app --reload
```

---

### ⚠️ If `uvicorn` is not recognized

```bash
pip install uvicorn
```

OR run:

```bash
python -m uvicorn app:app --reload
```

---

### ⚠️ Required Dependency for File Upload

```bash
pip install python-multipart
```

---

### ✅ Step 2: Open Frontend

Open `index.html` in your browser
(or use Live Server in VS Code)

---

### ✅ Step 3: Use App

* Upload image
* Click **Predict**
* See prediction + confidence

---

## 🌐 API Endpoint

* **POST** `/predict`
* Input: Image file
* Output:

```json
{
  "prediction": "automobile",
  "confidence": 0.87
}
```

---

## ⚠️ Notes

* Model trained on CIFAR-10 → may not perform well on real-world images
* Keep `model.pth` in the same folder as `app.py`
* Backend must be running before using frontend

---

## 🔥 Future Improvements

* Top-3 predictions
* Better UI (React)
* Deployment (Render / AWS)
* Custom dataset training

---

## 👨‍💻 Author

Vinayak Sanjay Dhulubulu
