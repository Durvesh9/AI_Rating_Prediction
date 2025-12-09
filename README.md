
# ⭐ AI Rating Prediction using LLMs (Gemini API)

This project predicts **1–5 star ratings** from free-form text reviews using **Large Language Models (LLMs)** via **Google Gemini API**, without training any custom ML model.
It explores how **prompt engineering** affects accuracy and JSON response reliability across three approaches:

* **V1:** Simple Prompt
* **V2:** Structured Prompt
* **V3:** Few-Shot Prompt (with examples)

The project also includes evaluation of **JSON validity**, **prediction accuracy**, and safely parsing LLM outputs.

---



## 🔧 Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/Durvesh9/AI_Rating_Prediction
cd AI_Rating_Prediction
```

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. **Add your API key to `.env` (IMPORTANT)**

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_api_key_here
```

If `.env` is missing, the notebook will throw:

```
RuntimeError: GEMINI_API_KEY environment variable is not set
```

---

## 📄 About the Dataset

The project uses a Yelp-style dataset containing:

* `text` — review text
* `stars` — actual rating (1–5)

The notebook samples a subset of rows to avoid API overuse.

---

## 🤖 LLM-Based Rating Prediction

The core logic is in `RatingPredictor`, which uses three prompting strategies:

### **V1 — Simple Prompt**

A short, direct instruction asking the model to predict star rating.

### **V2 — Structured Prompt**

Adds explicit instructions:

* return only JSON
* no extra text
* integer rating 1–5
* short explanation

This increases JSON validity.

### **V3 — Few-Shot Prompt**

Includes:

* 3 example inputs/outputs
* rating rubric
* strict JSON rules

This produces the **best and most consistent accuracy**.

---

## 📊 Evaluation

After generating predictions, the notebook evaluates:

### 1️⃣ JSON Validity

Whether the model returned:

* correct JSON format
* required fields
* a numeric star rating in 1–5

### 2️⃣ Accuracy

How often `predicted_stars == actual_stars`.

The results are summarized in a comparison table.

---

## ⚠️ Important Note on API Quota

Gemini **free-tier quota = ~20 requests/day per project per model**.

Your loop may easily exceed quota:

```
200 reviews × 3 prompts = 600 API calls ❌
```

If quota is exceeded, the notebook raises:

```
ResourceExhausted: 429 You exceeded your current quota
```

You can reduce usage by:

* Lowering `sample_size`
* Disabling some prompt approaches
* Setting a max API call limit
* Using paid tier or multiple projects

---

## ▶️ Running the Notebook

Start Jupyter:

```bash
jupyter notebook
```

Open:

```
AI_Rating_Prediction/rating_prediction.ipynb
```

Make sure:

✔ `.env` contains your API key
✔ Internet is ON
✔ You reduce sample size if using a free-tier API key

---

## 🖥️ Streamlit Dashboard (Optional)

To run the UI:

```bash
streamlit run dashboards/dashboards.py
```

This provides:

* Input text → LLM → predicted rating
* Quick visual validation
* Interactive usage without Jupyter

---

## 🚀 Future Improvements

* Train classical ML baselines for comparison
* Add confusion matrix & misclassification analysis
* Increase dataset size (paid API tier)
* Cache responses to avoid repeated API calls
* Add batch processing with retry handlers


✅ Create a PDF-formatted report
✅ Write a more detailed README with images, tables, and diagrams
