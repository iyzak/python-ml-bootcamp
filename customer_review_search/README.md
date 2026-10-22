Customer Review Semantic Search



A Python-based semantic search project that retrieves relevant documents from a small knowledge collection using sentence embeddings and cosine similarity.



The project demonstrates core machine learning engineering concepts, including text embeddings, vector similarity, ranked retrieval, evaluation metrics, and exporting evaluation results to CSV.



Project Overview



Traditional keyword search often relies on exact word matches. Semantic search uses numerical representations of text to identify passages that are meaningfully related to a query, even when the wording differs.



This project uses a pretrained Sentence Transformers model to encode documents and search queries into vector representations. It then ranks documents by cosine similarity and returns the top matching results.



&#x20;Example use case



A user searches:



> Which framework helps train neural networks?



The system should rank the PyTorch document highly because its description explains that PyTorch is used to build and train neural networks.



Features



\- Semantic retrieval: Search using natural-language queries.

\- Pretrained sentence embeddings: Encode documents and queries using `all-MiniLM-L6-v2`.

\- Cosine similarity: Compare query and document embeddings.

\- Ranked results: Return the top three matching documents by default.

\- Retrieval evaluation: Measure Precision@1, Precision@3, Recall@3, and Reciprocal Rank.

\- Aggregate metrics: Calculate mean scores across five evaluation queries.

\- CSV export: Save per-query evaluation results to `evaluation\_results.csv`.

\- Interactive command-line interface:\*\* Search repeatedly or run the evaluation from the terminal.



&#x20;Tech Stack



\- Python

\- Sentence Transformers

\- PyTorch (used by the embedding model)

\- CSV module from the Python standard library

\- Git and GitHub



&#x20;How It Works



1\. Load the model:Initialize the pretrained `sentence-transformers/all-MiniLM-L6-v2` model.

2\. Encode documents: Convert each document into a numerical vector.

3\. Encode the query: Convert the user's search query into a vector using the same model.

4\. Calculate similarity: Compare the query vector with document vectors using cosine similarity.

5\. Rank results: Sort documents by similarity and return the top results.

6\. Evaluate retrieval: Compare retrieved document IDs against manually assigned relevance labels.

7\. Export results: Write individual query metrics and retrieved IDs to a CSV file.



Document embeddings are calculated once when the application starts, rather than being recalculated for every search query.



&#x20;Project Structure



```text

customer\_review\_search/

├── semantic\_search.py

├── README.md

└── evaluation\_results.csv

```



`evaluation\_results.csv` is generated when the evaluation is run. It may not exist in a fresh checkout until the evaluation command has been executed.



The project also uses the root-level `requirements.txt` and virtual environment from the parent `python-ml-bootcamp` repository.

&#x20;Getting Started



&#x20;Prerequisites



\- Python installed on your computer

\- Git

\- Internet access for downloading the pretrained model on its first run



&#x20;1. Clone the repository



```cmd

git clone https://github.com/iyzak/python-ml-bootcamp.git

cd python-ml-bootcamp

```



If you already cloned the repository, navigate to your existing local folder instead.



2\. Create and activate a virtual environment



If you already have the bootcamp virtual environment, activate it:



```cmd

.venv\\Scripts\\activate

```



If you do not have one yet, create it first:



```cmd

python -m venv .venv

.venv\\Scripts\\activate

```



&#x20;3. Install dependencies



From the repository root, run:



```cmd

python -m pip install -r requirements.txt

```



&#x20;4. Navigate to the project



```cmd

cd customer\_review\_search

```



5\. Start semantic search



```cmd

python semantic\_search.py

```



Enter a natural-language query at the prompt. Type `quit` to exit.



Example queries:



\- `How can I identify positive and negative reviews?`

\- `How do I compare text using numerical vectors?`

\- `Which framework helps train neural networks?`

\- `What tool supports numerical calculations and arrays?`

\- `What foods make up a healthy diet?`



6\. Run the evaluation



Start the program and enter:



```text

eval

```



The program prints the per-query retrieval metrics and overall averages. It also generates `evaluation\_results.csv` in the project directory.



&#x20;Evaluation



The current evaluation uses five test queries and manually assigned relevant document IDs.



| Metric | Mean score | Interpretation |

|---|---:|---|

| Precision@1 | 1.00 | The first result was relevant for all five test queries. |

| Precision@3 | 0.47 | About 47% of the retrieved top-three results were labelled relevant, averaged across queries. |

| Recall@3 | 1.00 | All documents labelled relevant for each test query appeared in its top three results. |

| Mean Reciprocal Rank (MRR) | 1.00 | The first relevant result ranked first for every test query. |



Scores are based on the current small document collection and its manually assigned relevance labels. They are an initial experiment, not evidence of production-level retrieval performance.



Metric definitions



\- \*\*Precision@1:\*\* Whether the highest-ranked result is relevant.

\- \*\*Precision@3:\*\* The fraction of the top three retrieved documents that are relevant.

\- \*\*Recall@3:\*\* The fraction of all labelled relevant documents retrieved within the top three.

\- \*\*MRR:\*\* The average reciprocal rank of the first relevant result.



Example Output



```text

Customer Review Semantic Search

Type 'eval' to evaluate retrieval.

Type 'quit' to exit.



Enter a search query: Which framework helps train neural networks?



Most relevant documents:



ID: DOC005

Title: PyTorch

Similarity: <score depends on the model output>

Text: PyTorch is a machine learning framework used to build and train neural networks.

```



The example illustrates the expected result structure. The displayed similarity score is intentionally not fixed because the program calculates it at runtime.



Current Limitations



\- The document collection contains only seven manually written documents.

\- Evaluation uses five predefined test queries.

\- Relevance labels are manually assigned and may not reflect every user's search intent.

\- The system always returns up to three results, even when some matches are weak.

\- Similarity scores are not calibrated probabilities of relevance.

\- The current application searches in-memory documents and does not use a vector database.

\- The evaluation results should not be generalized to larger or real-world datasets without additional testing.



Future Improvements



\- Expand the test dataset with realistic customer-review queries.

\- Add more diverse documents and relevance judgements.

\- Experiment with similarity thresholds to filter weak matches.

\- Compare embedding models using the same evaluation dataset.

\- Add automated tests and input validation.

\- Support loading documents from CSV files.

\- Explore a vector database for larger collections.

\- Build an API or web interface for interactive search.



&#x20;Learning Outcomes



This project provides hands-on practice with:



\- Text embeddings and vector representations

\- Semantic similarity and ranked retrieval

\- Reusing pretrained machine learning models

\- Retrieval evaluation and relevance labels

\- CSV reporting and reproducible experiments

\- Python project organization and Git-based version control



Author



Isaac Munene



GitHub: \[@iyzak](https://github.com/iyzak)



This project is part of my hands-on Python and machine learning learning journey.



