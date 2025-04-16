# Predicting Clinical Trial Outcomes using Machine Learning and Neo4j
---

##  Overview

This project aims to improve the accuracy and efficiency of predicting **clinical trial outcomes** by integrating **machine learning models** with a **Neo4j graph database**. We propose a relationship-based data modeling approach that leverages publicly available biomedical datasets and advanced data engineering methods to construct a scalable and high-performing prediction system.

---

##  Objective

The goal is to develop a predictive system that uses **drug bioactivity**, **disease associations**, and **trial metadata** to estimate the probability of clinical trial success or failure. This is achieved by:

- Structuring the data into a graph database (Neo4j)
- Performing relationship-based queries to build data features
- Applying ensemble ML algorithms like XGBoost
- Optimizing data integration and model training pipelines

---

##  Key Highlights

- **Data Sources**: ChEMBL, DrugBank, CTD, AACT, UniProt  
- **Graph Modeling**: Neo4j for managing drug–target–disease–trial relationships  
- **Machine Learning Models**:  
  - XGBoost (achieved 74.53% accuracy)  
  - Support Vector Machine (SVM)  
  - Random Forest (for benchmark comparisons)
- **Data Preprocessing**: SMOTE for class imbalance, feature engineering for chemical descriptors  
- **Query Optimization**: Graph partitioning and indexing to reduce response time by 62%

---

##  Files in Repository

- `Clinical_trial_predictor.ipynb` – Jupyter notebook for ML pipeline, training, and evaluation  
- `csv_to_neo4j.py` – Python script to ingest structured data (CSV/TSV) into the Neo4j graph database  
- `README.md` – Project documentation

---

##  Results Summary

| Metric                | Value          |
|----------------------|----------------|
| Model Accuracy        | 74.53%         |
| Precision (weighted)  | 60%            |
| Recall (weighted)     | 75%            |
| F1-score (weighted)   | 64%            |
| Query Time Reduction  | 62% via Neo4j  |

- **Best performance** was seen in the "Other" outcome class (F1 = 0.85)  
- **Worst recall** occurred in rare categories like "biomarkers" and "mortality"  

---

##  Technologies Used

- **Language**: Python 3  
- **Database**: Neo4j Graph DB  
- **Libraries**: Pandas, NumPy, Scikit-learn, XGBoost, imbalanced-learn (SMOTE)  
- **Visualization**: Matplotlib, Seaborn  
- **Tools**: Jupyter Notebook, Cypher, CSV/TSV preprocessing

---

##  Future Enhancements

- Add **Graph Neural Networks (GNNs)** to capture higher-order graph features  
- Use **Explainable AI (XAI)** frameworks (SHAP, LIME) to improve transparency  
- Implement **adaptive learning** and **real-time retraining** with new clinical data  
- Address ethical and regulatory considerations for real-world deployment

---

## Getting Started

### Requirements

- Python 3.x  
- Neo4j Desktop or Community Edition  
- Jupyter Notebook  
- Required Python packages (install using pip):

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn neo4j
Steps to Run
Clone the repo
```
git clone https://github.com/yourusername/clinical-trial-predictor.git
cd clinical-trial-predictor
```
Start Neo4j and run the Cypher scripts (optional if you already have the data loaded)

Run csv_to_neo4j.py to import data into the database

Open and run Clinical_trial_predictor.ipynb to train and evaluate the model

### Acknowledgements
Special thanks to Dr. Shafaq Khan for mentorship and guidance throughout the course.

Let me know if you’d like me to export this into a downloadable `README.md` file, or help you generate additional content such as `requirements.txt`, Cypher scripts, or a dataset directory structure.







