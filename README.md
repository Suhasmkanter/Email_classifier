
# 📩 Email Spam Detection Project

## 📝 Project Overview
This project focuses on building a **machine learning model** to classify emails as spam or not spam using the UCI Spam Dataset.  
We applied preprocessing, feature scaling, model training with Logistic Regression, Random Forest, and cross-validation, along with performance evaluation using confusion matrix and ROC curve.



## 📚 Dataset Information
- **Source:** UCI Machine Learning Repository  
- **Features:** The dataset consists of 57 features extracted from emails such as word frequencies, character frequencies, and capital letter statistics.
- **Target:**  
  - `1` = Spam Email  
  - `0` = Non-Spam (Ham) Email  



## 🔥 Project Workflow
1. Loading the Data
2. Data Cleaning
   - Removed missing values
   - Removed duplicate rows
3. Exploratory Data Analysis (EDA)
   - Checked class distribution (spam vs non-spam)
   - Generated word clouds
   - Plotted feature distributions
4. Feature Engineering
   - StandardScaler used to scale continuous variables
5. Model Building
   - Logistic Regression
   - Random Forest Classifier
6. Model Evaluation
   - Confusion Matrix
   - Accuracy Score
   - ROC-AUC Curve
7. Hyperparameter Tuning
   - GridSearchCV used for tuning:
     - `max_iter` for Logistic Regression
     - `n_estimators`, `max_depth` for Random Forest
8. Cross Validation
   - 5-fold Cross-Validation used to check model stability
9. Final Conclusion

---

## ⚙️ Models Used
| Model | Accuracy |
| Logistic Regression | ~91% |
| Random Forest Classifier | ~94% |

- ROC-AUC scores were also strong, indicating good model performance.



## 📈 Evaluation Metrics
- Confusion Matrix
- Accuracy
- Precision, Recall, F1-Score (optional, could be added)
- ROC Curve Analysis



## 📊 Key Visualizations
- Word Clouds for Spam and Non-Spam Emails
- Distribution plots for important features
- ROC Curves showing model discrimination power

---

## 📦 Requirements
- Python 3.x
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - wordcloud

Install libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn wordcloud
```

---

## 🚀 How to Run
```bash
# Clone the repository
git clone https://github.com/yourusername/email-spam-detection.git

# Navigate to project folder
cd email-spam-detection

# Open the Jupyter Notebook
jupyter notebook Email_Spam_Detection.ipynb
```



## ✨ Future Improvements
- Try other models like XGBoost, SVM
- Do advanced text preprocessing (like TF-IDF)
- Build a simple web-app using Flask or Streamlit

---

## 🙏 Acknowledgements
- UCI Machine Learning Repository for providing the dataset.
- Scikit-learn library for easy model building.



# 🛡️ Conclusion
This project shows a complete machine learning pipeline — from loading raw data, cleaning, exploring, training models, evaluating them, and optimizing performance.  
It highlights the importance of **EDA**, **Feature Scaling**, **Model Validation**, and **Ensemble Methods** for real-world machine learning tasks.




