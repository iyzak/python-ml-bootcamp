\# Customer Review Semantic Search



\## Overview



A small natural language processing (NLP) project that retrieves relevant information from a document collection using sentence embeddings and cosine similarity.



The project demonstrates semantic search, reusable Python functions, ranked retrieval results, and a basic evaluation method.



\## Problem Statement



Keyword searches may miss relevant information when a user's query uses different wording from the source document. This project explores how sentence embeddings can retrieve text based on semantic similarity.



\## How It Works



1\. Load the pretrained `all-MiniLM-L6-v2` sentence embedding model.

2\. Convert documents into numerical embeddings.

3\. Encode each user query.

4\. Calculate cosine similarity between the query and document embeddings.

5\. Return the top three matching documents.

6\. Evaluate retrieval using Recall@3.



Document embeddings are computed once at startup and reused for subsequent searches.



\## Technology Stack



\- Python

\- Sentence Transformers

\- PyTorch (used by the embedding model)

\- Cosine similarity

\- Git and GitHub



\## Setup



Run these commands from the project directory.



\### 1. Create and activate a virtual environment



```cmd

python -m venv .venv

.venv\\Scripts\\activate

```



\### 2. Install the dependency



```cmd

python -m pip install sentence-transformers

```



\### 3. Run the application



```cmd

python semantic\_search.py

```



The pretrained model may download the first time the application runs. An internet connection may be required.



\## Example Queries



\- How can I identify positive and negative reviews?

\- How do I compare text using numerical vectors?

\- What tools can classify customer feedback?



The application returns the top three matching documents with their titles, text, IDs, and similarity scores.



\## Evaluation



The initial evaluation contains two test queries with manually designated relevant document IDs.



| Test query | Relevant documents found | Recall@3 |

|---|---:|---:|

| Review identification | 3 of 3 | 1.00 |

| Text vector comparison | 1 of 1 | 1.00 |



These results demonstrate that the current test cases retrieve all designated relevant documents within the top three results. Because the test set is small and manually defined, the results should not be interpreted as a general performance guarantee.



\## Limitations and Future Improvements



\- Expand the document collection with realistic customer feedback and product documentation.

\- Build a larger evaluation dataset with independently reviewed relevance labels.

\- Compare embedding models and retrieval configurations.

\- Measure additional metrics, such as Precision@k and Mean Reciprocal Rank.

\- Explore integrating sentiment classification to label actual customer reviews as positive, negative, or neutral.



\## Learning Outcomes



This project provides practical experience with pretrained sentence embedding models, semantic similarity, ranked information retrieval, Python functions, and basic retrieval evaluation.

