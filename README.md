# 🤖 Robot Adviser App – From Reviews to Product Summaries

Robot Adviser is an NLP-powered product advisor that transforms raw customer reviews into buyer-ready insights.  
The pipeline:

1. **Sentiment classification** – label each review as Negative / Neutral / Positive.  
2. **Category clustering** – group products into clear meta-categories.  
3. **Review summarization** – generate concise, buyer-friendly summaries of the top options per category.  
4. **Web app (Streamlit)** – interactively explore categories, sentiment, products and summaries.

---

## 📁 Repository Structure

- `1_sentiment_calssifier_i.ipynb` – data prep + sentiment model (TF-IDF + LogReg/SVM)  
- `2_category_clustering_i.ipynb` – product clustering (k-means + silhouette)  
- `3_review_summerizer_i.ipynb` – baseline summarization experiments (BART/T5)  
- `4_finetuning_additional_i.ipynb` – soft-prompt / adapter fine-tuning for summarization  
- `app.py` – Streamlit app (**Robot Adviser** UI)  
- `clustered_reviews_kmeans_tfidf (3).csv` – processed dataset with clusters & metadata  
- `softprompt_adapter.zip` – optional adapter for the summarization model  
- `agent.jpg` – avatar image used in the UI  
- `content/` – additional resources (e.g., pitch, notes)  
- `requirements.txt` – Python dependencies

---

## 🚀 Quick Start

1. **Create & activate a virtual environment** (optional but recommended).
2. Install dependencies:

   ```bash
   pip install -r requirements.txt

3. Run the web app:

   ```bash
   streamlit run app.py

5. Open the Streamlit URL in your browser, choose a category and sentiment, and Robot Adviser will:
- show a representative product, and
- generate a concise summary for that category.
  
5. Notebooks
- Run the notebooks in order (1_… → 4_…) if you want to:
- retrain the sentiment classifier,
- recompute clusters, or
- fine-tune the summarization model and regenerate softprompt_adapter.zip.
