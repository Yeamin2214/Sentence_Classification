# Sentence_Contradiction_Classification
Develop a machine learning model to classify pairs of sentences into one of three categories: "Contradiction," "Entailment," or "Neutral" based on their semantic relationships.

## Project Overview

This project focuses on finding the best machine learning or deep learning model for **sentence contradiction classification**. The goal is to accurately classify the relationship between a **premise** and a **hypothesis** into three categories:  
- **Entailment** (the hypothesis follows logically from the premise).  
- **Neutral** (the hypothesis is unrelated to the premise).  
- **Contradiction** (the hypothesis contradicts the premise).  

To achieve this, different models were implemented and evaluated based on their **accuracy, precision, recall, and F1-score**.

---

## Dataset Description

The dataset used for this project consists of **12,120 samples** and includes multiple languages. It contains the following columns:

- **`id`**: A unique identifier for each sample.  
- **`premise`**: The first sentence in the pair, providing a statement or context.  
- **`hypothesis`**: The second sentence that needs to be classified as **entailment, neutral, or contradiction** with respect to the premise.  
- **`lang_abv`**: The language abbreviation (e.g., "en" for English, "fr" for French).  
- **`language`**: The full name of the language.  
- **`label`**: The classification label:
  - **0**: Contradiction  
  - **1**: Neutral  
  - **2**: Entailment  

The dataset is **multilingual**, containing sentences in **12 different languages**, making preprocessing and model training more challenging.

---

## Model Implementation Details

To classify **contradiction, entailment, and neutral relationships**, multiple machine learning and deep learning models were implemented to compare their performance.

### **Traditional Machine Learning Models**
- **Random Forest (RF)**
- **Extreme Gradient Boosting (XGBoost)**

### **Deep Learning Models**
- **Bidirectional Long Short-Term Memory (BiLSTM)**
- **Artificial Neural Network (ANN)**

### **Transformer-Based Models**
- **Multilingual BERT (mBERT)**
- **Cross-lingual Language Model - RoBERTa based (XLM-R )**

Each model was trained and evaluated on the dataset to determine the most effective approach.

---

## Steps to Run the Code

This project includes **four Jupyter notebooks** that need to be executed sequentially.

1. **EDA, Preprocessing, and Baseline Models Notebook**  
   - Load the dataset and explore it using **exploratory data analysis (EDA)**.
   - Apply **preprocessing techniques** such as tokenization, lemmatization, and stopword removal.
   - Train and evaluate **baseline models** (Random Forest, XGBoost).

2. **Transformer & Deep Learning Model Notebooks**  
   - **mBERT Notebook** – Runs the **Multilingual BERT** model.
   - **BiLSTM & ANN Notebook** – Implements **BiLSTM and Artificial Neural Network (ANN)**.
   - **XLM-R Notebook** – Runs the **XLM-RoBERTa** model.

### **Execution Instructions**
- All dependencies are included in the code.
- Simply open the notebooks and **execute the cells** in their respective environments.
- The evaluation metrics will be displayed in each notebook.

---

## Model Evaluation Results

A detailed **performance report** comparing different models is provided in a separate **document file** uploaded with this project. The report includes:
- Accuracy, Precision, Recall, and F1-score for each model.
- Confusion matrices to visualize misclassifications.
- AUC-ROC curves to evaluate model performance.

---

## Model Tuning and Optimization

To improve model performance, the following tuning techniques were applied:

- **Optimizers:**  
  - Adam (Adaptive Moment Estimation) was used for deep learning models for better convergence.  

- **Learning Rate Adjustment:**  
  - Different learning rates were tested to find the best balance between training speed and accuracy.  

- **Hyperparameter Tuning:**  
  - Random Search were used to find the best parameters for traditional ML models.  
  - In neural networks, different **batch sizes** and **epochs** were tested to optimize training.  

These optimizations helped in improving the overall model accuracy while reducing overfitting.

---

## Additional Observations and Notes

### **Challenges Faced**
- **Multilingual Complexity**: The dataset contains **12 different languages**, making **text preprocessing difficult**. Handling different tokenization and language-specific rules added complexity.
- **Hardware Limitations**: Due to the **unavailability of a GPU**, training transformer-based models and deep learning architectures took a significant amount of time.

### **Future Improvements**
- The model can achieve **better accuracy** by **adding more training data**.
- **Fine-tuning hyperparameters** of transformer models can improve classification performance further.

---

## Contributors

- **Yeamin2214**

For any queries, please raise an issue in the repository.

---

