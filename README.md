# 🧠 Text Classification using LSTM (Deep Learning NLP Project)

This project demonstrates how to build a **Long Short-Term Memory (LSTM)** based deep learning model for **text classification** using **TensorFlow and Keras**.  
It involves **text preprocessing**, **tokenization**, **sequence padding**, **LSTM model training**, and **evaluation**.

---

## 🚀 Project Overview

This notebook covers the following major steps:

1. **Library Setup** – Install and import required dependencies.  
2. **Data Loading** – Load a dataset containing text samples and corresponding labels.  
3. **Text Preprocessing** – Clean text data using regular expressions and remove stopwords.  
4. **Tokenization & Padding** – Convert text into integer sequences suitable for neural networks.  
5. **Model Creation** – Build a Sequential LSTM-based neural network.  
6. **Training & Validation** – Train the model and monitor loss/accuracy.  
7. **Evaluation** – Generate classification metrics like accuracy, confusion matrix, and classification report.  
8. **Visualization** – Plot training history and confusion matrix heatmap.

---

## ⚙️ Requirements

Install the following dependencies before running the notebook:

```bash
pip install tensorflow scikit-learn matplotlib seaborn nltk numpy pandas kagglehub
```

---

## 🧩 Dataset

- The dataset is likely loaded from **KaggleHub** or a local CSV file.  
- It contains **text data** (e.g., reviews, comments, or tweets) and **labels** (e.g., sentiment or emotion categories).  
- Modify the path inside the notebook if you are using your own dataset.

Example structure:

| Text | Label |
|------|--------|
| "I love this movie!" | Positive |
| "The product was terrible" | Negative |

---

## 🧱 Model Architecture

The model uses the following architecture:

```python
Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
    LSTM(128, return_sequences=False),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])
```

- **Embedding Layer**: Converts words into vector representations.  
- **LSTM Layer**: Learns long-term dependencies in sequential text data.  
- **Dropout Layers**: Prevent overfitting.  
- **Dense Layers**: Map learned features to output categories.

---

## 📊 Evaluation Metrics

The notebook computes:
- **Accuracy Score**
- **Confusion Matrix**
- **Classification Report (Precision, Recall, F1-score)**
- **Training & Validation Loss Curves**

  ```bash
  Classification Report:
               precision    recall  f1-score   support

        Fake       0.98      0.96      0.97      3536
        Real       0.97      0.98      0.97      4285

    accuracy                           0.97      7821
   macro avg       0.97      0.97      0.97      7821
  weighted avg       0.97      0.97      0.97      7821
  ```

---

## 📈 Visualization

You’ll find:
- **Loss and Accuracy Graphs** during training.
<img width="708" height="393" alt="image" src="https://github.com/user-attachments/assets/03f47a33-4d25-46d0-8344-d91c8bf67680" />
<img width="700" height="393" alt="image" src="https://github.com/user-attachments/assets/6af6a449-9cb3-4a18-b4f3-39d565b46c71" />

- **Confusion Matrix Heatmap** for better interpretability.
<img width="548" height="455" alt="image" src="https://github.com/user-attachments/assets/9bcce7cf-4d70-4b4e-a07f-56c38053258b" />


---

## 🪄 How to Use

1. Clone this repository:

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

2. Open the notebook:

```bash
jupyter notebook Fake News Detection using LSTM.ipynb
```

3. Run all cells sequentially to:
   - Install dependencies
   - Load dataset
   - Train model
   - View evaluation metrics and plots

---

## 🔧 Customization

- Replace the dataset with your own text file or CSV.  
- Adjust `LSTM` units, learning rate, or embedding size for experimentation.  
- Add `GRU` or `Bidirectional LSTM` layers for better performance.  
- Tune hyperparameters using `GridSearch` or `KerasTuner`.

---

## 🧠 Future Enhancements

- Add **Word2Vec** or **GloVe** embeddings.  
- Implement **attention mechanisms** for better context capture.  
- Create a **Streamlit/Gradio** app for real-time text classification.  
- Extend to **multilingual** text datasets.

---

## 🙏 Acknowledgements

- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [NLTK](https://www.nltk.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
