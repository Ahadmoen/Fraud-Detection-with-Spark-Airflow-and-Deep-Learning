
# 💳 Credit Card Fraud Detection with ETL, Machine Learning & Deep Learning

An end-to-end machine learning project that detects fraudulent credit card transactions using advanced data engineering and modeling techniques. It integrates PySpark for ETL, Airflow for scheduling, and models like Random Forest, Gradient Boosting, and Deep Learning (Keras Functional API). SMOTE is used to handle the class imbalance.

---

## 📁 Dataset

- **Source**: Kaggle Credit Card Fraud Detection Dataset
- **Rows**: 284,807
- **Features**: V1 to V28 (PCA), Time, Amount
- **Target**: Class → 0 (Legit), 1 (Fraud)

---

## 📊 Workflow Overview

1. **ETL with PySpark & Airflow**
2. **Exploratory Data Analysis**
3. **Preprocessing (Scaling, SMOTE)**
4. **Modeling (RF, GBM, Deep Learning)**
5. **Evaluation (Accuracy, AUC, F1)**
6. **Deployment (User Prediction)**

---

## ⚙️ ETL (Airflow DAG)

```python
def ETL(filepath, output_path):
    spark = SparkSession.builder.appName("CreditCardFraudDetection").getOrCreate()
    df = spark.read.csv(filepath, header=True)
    df.createOrReplaceTempView("creditcard_table")
    clean_df = spark.sql("SELECT * FROM creditcard_table WHERE Class IS NOT NULL")
    clean_df.write.mode("overwrite").csv(output_path, header=True)
    spark.stop()
    return clean_df.toPandas()

dag = DAG(
    'creditcard_etl_dag',
    default_args={'owner': 'airflow', 'start_date': datetime(2025, 4, 26), 'retries': 1},
    schedule_interval='@weekly',
    catchup=False,
)

etl_task = PythonOperator(
    task_id='run_creditcard_etl',
    python_callable=lambda: ETL("creditcard.csv", "updated.csv"),
    dag=dag,
)
```

---

## 📊 Data Summary

- **Fraud Percentage**: `0.1727%`
- **Skewness of Amount**: `1.57 (positively skewed)`
- **Outliers**: Visualized, not removed

---

## 📈 SMOTE for Class Balancing

```python
from imblearn.over_sampling import SMOTE
X_train, y_train = smote.fit_resample(X_train, y_train)
```

---

## 🧠 Models

### Random Forest

```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

### Gradient Boosting

```python
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)
```

### Deep Learning (Keras)

```python
inputs = Input(shape=(X_train.shape[1],))
x = Dense(128, activation='relu')(inputs)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(32, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)
model = Model(inputs, outputs)
model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['AUC'])
```

---

## 📊 Evaluation

| Model             | AUC     | Accuracy |
|------------------|---------|----------|
| Random Forest     | 0.9849  | 99.97%   |
| Gradient Boosting | 0.9844  | -        |
| Deep Learning     | 0.9734  | 99.77%   |

---

## 🧪 Prediction Interface

```python
input_df = pd.DataFrame([user_inputs])
scaled_input = scaler.transform(input_df)
prediction = model.predict(scaled_input)
```

---

## 📎 Future Enhancements

- Streamlit/Flask Web App
- Real-time streaming with Kafka
- SHAP Interpretability
- Hyperparameter tuning

