# 🌸 Syntecxhub Flower Classification

A machine learning project for classifying flowers using the classic **Iris dataset**.This project features a robust CLI for instant predictions and model evaluation.  
This repository demonstrates end-to-end data analysis, model training, evaluation, and prediction.

---
The model analyzes four key dimensions:

Sepal Length (cm)

Sepal Width (cm)

Petal Length (cm)

Petal Width (cm)

---
## 📂 Project Structure

- **iris.csv** → Dataset containing flower measurements (sepal length/width, petal length/width).
- **irisclassification.ipynb** → Jupyter Notebook with data exploration, visualization, and model training.
- **iris_classification_profiling_report.html** → Automated profiling report of the dataset.
- **iris_pairplot.png** → Pairplot visualization of feature relationships.
- **Decision Tree Confusion Matrix.png** → Confusion matrix for Decision Tree classifier.
- **Logistic Regression Confusion Matrix.png** → Confusion matrix for Logistic Regression classifier.
- **iris_model.pkl** → Saved trained model for reuse.
- **Cli_predict_iris_csv.py** → Command-line script to make predictions from a CSV file.
- **README.md** → Project documentation.

---

## 🚀 Features

- Exploratory Data Analysis (EDA) with visualizations.
- Multiple classification models (Decision Tree, Logistic Regression).
- Performance evaluation using confusion matrices.
- Model persistence with `pickle`.
- CLI-based prediction tool for batch classification.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Chazgrey/Syntecxhub_Flower_Classification.git
cd Syntecxhub_Flower_Classification
```
---
## 📊 Usage
### Run Jupyter Notebook
```bash
jupyter notebook irisclassification.ipynb
```

### Generate Predictions via CLI
```bash
python Cli_predict_iris_csv.py iris.csv
```
This will output predicted flower classes for the dataset.
---
## 📈 Results
- Decision Tree and Logistic Regression models were trained and evaluated.
- Confusion matrices show classification accuracy and misclassifications.
- The saved model (iris_model.pkl) can be loaded for future predictions.
---
## 🛠️ Technologies Used
- Python
- Scikit-learn
- Pandas
- Matplotlib / Seaborn
- Jupyter Notebook
---
## 📌 Future Improvements
- Add more classifiers (Random Forest, SVM).
- Implement cross-validation for robust performance metrics.
- Build a simple web app interface for predictions.
---
## 👤 Author
Developed by [Chazgrey](https://github.com/Chazgrey)

---
