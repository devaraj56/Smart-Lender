\# 🏦 Smart Lender – Loan Approval Prediction System



\## 📌 Project Overview



Smart Lender is a Machine Learning-powered web application that predicts whether a loan application is likely to be approved based on applicant details. The system helps financial institutions make faster and more consistent lending decisions by using predictive analytics.



The application is built using Python, Flask, and Scikit-learn, with a Random Forest Classifier as the final prediction model.



\---



\## 🚀 Features



\- Predicts loan approval status instantly

\- User-friendly web interface built with Flask

\- Data preprocessing and feature engineering

\- Missing value handling

\- Label Encoding of categorical features

\- Data balancing using SMOTE

\- Feature scaling using StandardScaler

\- Multiple Machine Learning model comparison

\- Best model saved using Pickle

\- Responsive and modern web interface



\---



\## 🛠 Technologies Used



\### Programming Language

\- Python 3.x



\### Libraries

\- Flask

\- NumPy

\- Pandas

\- Matplotlib

\- Scikit-learn

\- Imbalanced-learn (SMOTE)

\- XGBoost

\- Pickle



\### Development Tools

\- Visual Studio Code

\- Git

\- GitHub



\---



\## 📂 Project Structure



```

SmartLender/

│

├── app.py

├── train\_model.py

├── requirements.txt

├── README.md

├── .gitignore

│

├── dataset/

│   └── loan\_prediction.csv

│

├── model/

│   ├── rdf.pkl

│   └── scaler.pkl

│

├── templates/

│   ├── home.html

│   ├── predict.html

│   └── submit.html

│

└── static/

&#x20;   ├── css/

&#x20;   ├── images/

&#x20;   └── js/

```



\---



\## 📊 Dataset



The project uses the Loan Prediction Dataset containing applicant information such as:



\- Gender

\- Married

\- Dependents

\- Education

\- Self Employed

\- Applicant Income

\- Coapplicant Income

\- Loan Amount

\- Loan Amount Term

\- Credit History

\- Property Area



Target Variable:



\- Loan Status (Approved / Rejected)



\---



\## ⚙ Data Preprocessing



The following preprocessing steps were performed:



\- Handling missing values

\- Removing unnecessary columns

\- Label Encoding categorical features

\- Feature Scaling using StandardScaler

\- Balancing dataset using SMOTE

\- Train-Test Split



\---



\## 🤖 Machine Learning Models Used



The following algorithms were trained and evaluated:



\- Decision Tree Classifier

\- Random Forest Classifier ✅ (Selected Model)

\- K-Nearest Neighbors (KNN)

\- Gradient Boosting Classifier



\### Model Performance



| Model | Accuracy |

|--------|----------|

| Decision Tree | 76.92% |

| Random Forest | \*\*79.88%\*\* |

| KNN | 69.23% |

| Gradient Boosting | 78.11% |



Random Forest achieved the highest accuracy and was selected as the final prediction model.



\---



\## 💾 Saved Models



The trained models are stored in the \*\*model/\*\* directory.



\- rdf.pkl (Random Forest Model)

\- scaler.pkl (StandardScaler)



\---



\## 🌐 Web Application



The Flask application allows users to:



\- Enter loan applicant details

\- Submit the form

\- Predict loan approval instantly

\- Display Approval or Rejection result



\---



\## ▶ How to Run



\### Clone the Repository



```bash

git clone https://github.com/devaraj56/Smart-Lender.git

```



\### Navigate to the Project



```bash

cd Smart-Lender

```



\### Install Dependencies



```bash

pip install -r requirements.txt

```



\### Train the Model (Optional)



```bash

python train\_model.py

```



\### Run the Flask Application



```bash

python app.py

```



Open your browser and visit:



```

http://127.0.0.1:5000

```



\---



\## 📸 Screenshots



Add screenshots here after uploading.



Example:



\- Home Page

\- Prediction Form

\- Prediction Result

\- Model Accuracy Output



\---



\## 📈 Future Enhancements



\- User Authentication

\- Database Integration

\- Loan History Tracking

\- Cloud Deployment

\- Interactive Dashboard

\- Email Notification

\- PDF Report Generation



\---



\## 🎯 Learning Outcomes



Through this project, I learned:



\- Machine Learning workflow

\- Data preprocessing techniques

\- Feature engineering

\- Model evaluation

\- Flask web development

\- Model deployment using Pickle

\- Git \& GitHub version control



\---



\## 👨‍💻 Author



\*\*Kovuri Udaya Siva Kumar\*\*



B.Tech – Computer Science and Engineering (Artificial Intelligence \& Machine Learning)



Kallam Haranadha Reddy Institute of Technology



GitHub:

https://github.com/devaraj56



\---



\## 📜 License



This project is developed for educational and academic purposes.



