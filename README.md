ML NLP LLM Projects

This repository contains a set of small machine learning, natural language processing, and large language model experiments.  
All were built and tested in Google Colab.  
These are not production systems but exploratory notebooks with reproducible results, metrics, and configurations.

Projects:

1. Baseline QA to SBERT-RAG  
   Dataset: SQuAD v2 (100 sample questions)  
   Baseline: GPT-2 QA generation  ROUGE ~ 0.037 / 0.002 / 0.032  
   With SBERT embeddings and retrieval augmentation → ROUGE ~ 0.041 / 0.0048 / 0.036  

2. Named Entity Recognition Fine-Tuning  
   Dataset: CoNLL-2003  
   Model: BERT-base-cased, 3 epochs, batch size 8  
   Metrics: F1 ≈ 0.895, Accuracy ≈ 0.97  

3. Quora QA RAG Pipeline  
   Embedding model: all-MiniLM-L6-v2  
   Vector store: Chroma  
   RAG via LangChain and ChromaDB, generation via FLAN-T5-base  
   Evaluation on 1k samples: P@1 ≈ 0.053, P@5 ≈ 0.238  

4. Mini Semantic Search on SQuAD  
   Dataset: SQuAD (first 20 Q&A pairs)  
   Embedding model: all-MiniLM-L6-v2  
   Retrieval by cosine similarity  
   Example: query “which building in Notre Dame has the biggest age” → Old College  

5. Abstractive Summarization with BART-large-cnn  
   Dataset: CNN/DailyMail  
   ROUGE scores: rouge1 = 0.6136, rouge2 = 0.3720, rougeL = 0.5454  
   Semantic similarity between generated and reference ≈ 0.842  

6. MRPC Sentence Pair Classification with Optuna Tuning  
   Dataset: GLUE MRPC  
   Model: DistilBERT with Hugging Face Trainer  
   Baseline F1 ≈ 0.894  
   Hyperparameter tuning with Optuna for learning rate, batch size, epochs, weight decay  
   Final test results: Accuracy ≈ 0.831, F1 ≈ 0.876  

How to run

1. Clone the repository
   git clone https://github.com/imankh7/NLP_LLM.git
   cd NLP_LLM

2. Install dependencies
   pip install -r requirements.txt

3. Open and run notebooks in Colab or Jupyter

Dependencies

transformers  
datasets  
sentence-transformers  
evaluate  
optuna  
scikit-learn  
langchain  
chromadb  
spacy

License
MIT License
