<!-- 🌌 HEADER -->
<p align="center">
  <img src="assets/branding/logo/mlforge-logo-primary.svg" alt="MLForge Logo" width="380"/>
</p>
<p align="center">
  <strong>Enterprise Automated Machine Learning & Experimentation Platform</strong>
</p>

---

# MLForge  
### Advanced AutoML Data Cleaning, Clustering & Model Export Platform

An **end-to-end Machine Learning Platform (MLForge)** that allows you to **upload, clean, preprocess, analyze, cluster, train, and export models** — all in one powerful interface.

🔗 **Live Project Link:** [Click Here to View App](https://mlforge.streamlit.app/)

---

# 🚀 Description

MLForge is a **complete AutoML pipeline system** designed for:

- Data Cleaning  
- Smart Preprocessing  
- Automated Model Training  
- Clustering & Visualization  
- Multi-format Model Export  

It eliminates manual effort and provides **AI-assisted decision-making at every step**.

---

# 🎯 Key Features

- 📂 Multi-format file upload (CSV, Excel, JSON, XML, YAML, SQLite)
- 🔍 Advanced Data Profiling
- 🧹 Intelligent Missing Value Handling
- ♻️ Smart Duplicate Removal
- ⚙️ Automated Preprocessing (Encoding + Scaling)
- 🤖 AutoML Model Training (Classification & Regression)
- 📊 Clustering (KMeans, DBSCAN, Agglomerative)
- 📈 Interactive Visualization Dashboard
- 📑 AI-generated insights & reports
- 💾 Export data + models (Pickle, Joblib, ONNX, TensorFlow, Torch)

# 🏆 Comparison with Industry Tools

| Feature / Capability                  | **MLForge (Your Platform)** | Google AutoML | H2O.ai | DataRobot | KNIME / RapidMiner |
|-------------------------------------|----------------------------------|---------------|--------|-----------|--------------------|
| Data Upload                         | ✅ Multi-format                  | ⚠️ Limited    | ✅     | ✅        | ✅                 |
| Data Cleaning                       | ✅ Manual + AI                   | ❌            | ⚠️     | ⚠️        | ⚠️                |
| Preprocessing Control               | ✅ Full Control                  | ❌            | ⚠️     | ⚠️        | ✅                 |
| AI Suggestions                      | ✅ Strong (USP)                  | ❌            | ⚠️     | ⚠️        | ❌                |
| Model Selection                     | ✅                              | ✅            | ✅     | ✅        | ✅                 |
| AutoML                              | ⚠️ Basic                        | ✅ Advanced   | ✅     | ✅        | ⚠️                |
| Clustering + Visualization          | ✅ Integrated                   | ❌            | ⚠️     | ⚠️        | ✅                 |
| Graph Customization                 | ✅ Advanced                     | ❌            | ❌     | ❌        | ✅                 |
| Report Export                       | ✅ Excel + PDF + Word           | ❌            | ❌     | ⚠️        | ❌                |
| UI Simplicity                       | ✅ Very Simple                  | ❌            | ❌     | ❌        | ❌                |
| Beginner Friendly                   | ✅ High                         | ❌            | ❌     | ❌        | ⚠️                |
| Deployment Ready                    | ❌                              | ✅            | ✅     | ✅        | ⚠️                |
| Real-time Prediction API            | ❌                              | ✅            | ✅     | ✅        | ❌                |
| Enterprise Scalability              | ❌                              | ✅            | ✅     | ✅        | ⚠️                |
---

# 🧠 How It Works (Pipeline)

```
Upload → Profiling → Cleaning → Preprocessing → Training → Clustering → Export
```

---

# 📂 Project Structure

```
project/

├── modules/
│   ├── model_export/
│   │   ├── export_manager.py
│   │   ├── joblib_exporter.py
│   │   ├── pickle_exporter.py
│   │   ├── onnx_exporter.py
│   │   ├── tensorflow_exporter.py
│   │   ├── torch_exporter.py
│   │
│   ├── ai_recommender.py
│   ├── automl.py
│   ├── clustering.py
│   ├── duplicate_handler.py
│   ├── evaluation.py
│   ├── exporter.py
│   ├── file_loader.py
│   ├── missing_handler.py
│   ├── profiling.py
│
├── utils/
│   └── helpers.py
│
├── reports/
├── app.py
├── requirements.txt
```

---

# 🔍 File Explanations (Deep)

## 📂 app.py  
Main Streamlit application  
- UI + Dashboard  
- Full pipeline control  
- Handles user interaction  
- Runs complete ML workflow  

👉 Example: :contentReference[oaicite:0]{index=0}  

---

## 📂 file_loader.py  
Handles multi-format file loading  
- CSV, Excel, JSON, XML, YAML, SQLite  
- Auto-detects format  
- Converts into Pandas DataFrame  

👉 :contentReference[oaicite:1]{index=1}  

---

## 📂 profiling.py  
Performs data analysis  
- Column summary  
- Missing values  
- Numeric & categorical stats  

👉 :contentReference[oaicite:2]{index=2}  

---

## 📂 missing_handler.py  
Handles missing data intelligently  
- Mean / Median / Mode  
- Forward / Backward fill  
- Custom values  

👉 :contentReference[oaicite:3]{index=3}  

---

## 📂 duplicate_handler.py  
Duplicate detection & removal  
- Full row detection  
- Column-based duplicates  
- Safe removal  

👉 :contentReference[oaicite:4]{index=4}  

---

## 📂 automl.py  
Core ML engine  
- Detects problem type  
- Splits dataset  
- Trains multiple models  
- Selects best model  

Models included:
- Logistic Regression  
- Random Forest  
- Decision Tree  
- SVM  
- KNN  
- Naive Bayes  

👉 :contentReference[oaicite:5]{index=5}  

---

## 📂 evaluation.py  
Evaluates model performance  

### Classification:
- Accuracy  
- Precision  
- Recall  
- F1 Score  

### Regression:
- R² Score  
- MAE  
- RMSE  
- MAPE  

👉 :contentReference[oaicite:6]{index=6}  

---

## 📂 clustering.py  
Unsupervised learning module  

Supports:
- K-Means (auto cluster detection)
- DBSCAN (auto eps detection)
- Agglomerative clustering  

Also includes:
- PCA (dimensionality reduction)
- Silhouette score evaluation  

👉 :contentReference[oaicite:7]{index=7}  

---

## 📂 ai_recommender.py  
AI-based suggestions  

- Missing value strategies  
- Model recommendations  
- Feature importance  
- Clustering suggestions  

👉 :contentReference[oaicite:8]{index=8}  

---

## 📂 exporter.py  
Export cleaned dataset  

Formats:
- CSV  
- Excel  
- JSON  
- XML  
- YAML  
- SQLite  

👉 :contentReference[oaicite:9]{index=9}  

---

## 📂 model_export/  
Advanced model export system  

### Supported Formats:
- Pickle  
- Joblib  
- ONNX  
- TensorFlow  
- PyTorch  

### Central Manager:
- export_manager.py handles all exports  

👉 :contentReference[oaicite:10]{index=10}  

---

## 📂 utils/helpers.py  
Utility functions  

- Column cleaning  
- Encoding helpers  
- Safe operations  
- Memory optimization  

👉 :contentReference[oaicite:11]{index=11}  

---

# 🤖 AI Intelligence

MLForge provides AI suggestions for:

- Missing value handling  
- Feature selection  
- Model selection  
- Clustering approach  

---

# 📊 Clustering Features

- Auto optimal cluster detection  
- Elbow method  
- Silhouette scoring  
- 2D visualization (PCA)  

---

# 💾 Model Export System

Export trained models into:

- `.pkl`
- `.joblib`
- `.onnx`
- `.zip` (TensorFlow)
- `.pt` (PyTorch)

---

# ▶️ Run Project

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

# 📊 Use Cases

- Data Science Projects  
- Machine Learning Automation  
- Business Analytics  
- Customer Segmentation  
- Financial Analysis  
- Research Work  

---

# 🚀 Future Enhancements

- Deep Learning models  
- NLP integration  
- Auto feature engineering  
- Cloud deployment  
- API support  

---

# 👨‍💻 Author

Daksh Vasani  
Machine Learning Developer  

---

# ⭐ Support

If you like this project, give it a ⭐ on GitHub!

---

<p align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,100:2c5364&height=170&section=footer&text=Thanks%20for%20Visiting!&fontSize=28&fontColor=ffffff&animation=twinkling&fontAlignY=65"/>
</p>
