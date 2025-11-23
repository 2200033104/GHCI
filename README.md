# GHCI
IntelliClassify – Autonomous AI-Based Financial Transaction Categorisation
# IntelliClassify – AI-Based Financial Transaction Categorisation

A scalable, transparent, and fully in-house ML system that classifies financial transaction strings into meaningful categories without any third-party API dependencies.

---

## 🚀 Features
- End-to-end ML-based transaction categorisation  
- Confidence scoring for every prediction  
- Editable category taxonomy (JSON/YAML)  
- Explainability using SHAP  
- User feedback loop for continuous model improvement  
- Secure, fast, and scalable API  
- Reproducible training pipeline with full evaluation metrics  

---

## 🧠 Technology Stack
**Programming & ML**
- Python 3.10  
- Scikit-learn, XGBoost  
- Pandas, NumPy  
- spaCy for NLP  
- SHAP for explainability  

**Backend**
- FastAPI  
- UVicorn  

**Storage**
- PostgreSQL / SQLite  
- taxonomy.json (configurable categories)  

**Deployment**
- Docker  
- GitHub  

---

## 🏗️ System Architecture
User Input
↓
Preprocessing → Vectorization → ML Model
↓
Categorisation Engine → Confidence Score
↓
Explainability (SHAP)
↓
Storage + Feedback → Retraining

---

## 🗄️ Data Model

### **transactions**
| column | type | description |
|--------|------|-------------|
| id | UUID | primary key |
| raw_text | text | original transaction |
| processed_text | text | cleaned NLP text |
| predicted_category | text | model output |
| confidence | float | probability score |
| created_at | datetime | timestamp |

### **feedback**
Stores user corrections.

### **taxonomy.json**
Defines all categories:

```json
{
  "Food & Dining": ["starbucks", "zomato"],
  "Shopping": ["amazon", "flipkart"],
  "Fuel": ["shell", "indian oil"]
}


