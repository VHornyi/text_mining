# Lord of the Rings - Text Mining Project

This project performs various text mining analyses on the dialogue scripts from the three books of "The Lord of the Rings" trilogy: "The Fellowship of the Ring," "The Two Towers," and "The Return of the King."

## Project Structure


TEXT_MINING_PROJECT/
├──results/
├── text_mining/
│ ├── data/
│ │ ├── 01 Fellowship.txt # Dialogue script for Book 1
│ │ ├── 02 Two Towers.txt # Dialogue script for Book 2
│ │ ├── 03 Return of the King.txt # Dialogue script for Book 3
│ │ └── races.csv # Character race information (currently not fully utilized)
│ ├── main.py # Main script to run the analysis
│ ├── utils.py # Utility functions (loading, preprocessing)
│ └── analysis.py # Functions for specific analyses
└── requirements.txt # Python package dependencies

## Technical Setup

### Prerequisites
*   Python 3.9+ (tested with Python 3.12)
*   pip (Python package installer)

### Installation
1.  **Clone the repository (or download the files):**
    ```bash
    # If you are using Git
    # git clone <repository_url>
    # cd TEXT_MINING_PROJECT
    ```
2.  **Install required Python packages:**
    Navigate to the root directory of the project (`TEXT_MINING_PROJECT/`) where `requirements.txt` is located and run:
    ```bash
    # Replace 'python' with your specific python executable if needed
    # (e.g., python3, or the full path to your python.exe)
    python -m pip install -r requirements.txt
    ```
    This will install all necessary libraries, including `pandas`, `nltk`, `scikit-learn`, `matplotlib`, `wordcloud`, and `gensim`.

3.  **NLTK Data Download:**
    The first time you run the script, it will attempt to download necessary NLTK resources (`wordnet`, `punkt`, `stopwords`, `omw-1.4`, `vader_lexicon`). Ensure you have an internet connection.

### Running the Analysis
To execute the text mining pipeline, run the `main.py` script from the `text_mining` directory:
```bash
# Navigate to the text_mining directory
cd text_mining

# Run the main script
# Replace 'python' with your specific python executable if needed
python main.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END

Alternatively, from the root TEXT_MINING_PROJECT directory:

# Replace 'python' with your specific python executable if needed
python text_mining/main.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END
Data Encoding

The script currently assumes the input .txt files are encoded in cp1252. This was determined to correctly read the provided sample files. If you use different files and encounter encoding errors, you might need to:

Modify the encoding parameter in the open() function within text_mining/utils.py (in the load_lotr_texts() function).

Alternatively, convert your .txt files to UTF-8 or cp1252 encoding using a text editor like VS Code or Notepad++.

Analysis Performed and Results

The script performs the following analyses on the combined text of all three books, as well as on individual books where appropriate. Graphical outputs (word clouds, frequency plots, sentiment bar chart, cluster plot) are displayed in separate windows during execution.

1. Text Preprocessing

Steps: Lowercasing, punctuation and number removal, tokenization, stop word removal (English), lemmatization.

Output Example:

--- Preprocessing Text ---
Total tokens after preprocessing: 16321
Sample processed tokens: ['galadriel', 'world', 'changed', 'feel', 'water', 'feel', 'earth', 'smell', 'air', 'much']
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END
2. Word Frequency Analysis

Identifies and visualizes the most frequent words and generates word clouds for the entire corpus.

3. N-gram Analysis

Identifies common sequences of N words.

Top Bigrams (2-grams) Example:

frodo sam: 58
mister frodo: 43
sam frodo: 37
gandalf gandalf: 35
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END

Interpretation: Highlights key character interactions (frodo sam), forms of address (mister frodo), and important plot elements or locations. Repeated names often indicate emphasis or direct address in dialogues.

Top Trigrams (3-grams) Example:

sam mister frodo: 17
frodo sam sam: 10
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END

Interpretation: Further emphasizes character interactions and common conversational patterns.

4. Document-Term Matrix (DTM) and TF-IDF

DTM: Term frequencies per document (book).

TF-IDF: Term importance per document, weighted by inverse document frequency.

Output Example:

DTM shape: (3, 1000)
TF-IDF matrix shape: (3, 1000)
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END

Interpretation: Each of the 3 books is represented by a vector of 1000 word features.

5. Sentiment Analysis (VADER)

Analyzes the overall sentiment of each book.

Output Example:

Sentiment for Fellowship: Compound=0.9999 (POSITIVE)
Sentiment for Two Towers: Compound=0.7198 (POSITIVE)
Sentiment for Return of the King: Compound=-0.9994 (NEGATIVE)
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END

Interpretation: "Fellowship" and "Two Towers" are rated positively. "Return of the King" is rated negatively, likely due to a high prevalence of conflict-related vocabulary, despite its ultimately positive resolution. VADER is sensitive to word choice over complex narrative sentiment.

6. Topic Modeling

Discovers abstract topics within the books.

Latent Dirichlet Allocation (LDA):

Example Topics (Top 7 words):

Topic #1: sam, frodo, gandalf, aragorn, gollum, pippin, théoden (Main Characters)

Topic #2: frodo, gandalf, sam, aragorn, bilbo, ring, boromir (Fellowship & Ring)

Topic #3: breaking, cool, heed, heh, anor, toby, courtesy (Less coherent, possibly noise)
Interpretation: LDA identifies key character-centric themes. The third topic is less clear, potentially due to the small number of documents (3).

Latent Semantic Analysis (LSA/SVD):

Example Topics (Top 7 words, reduced to 2 topics):

Topic #1: frodo, gandalf, sam, aragorn, pippin, gollum, merry (Core Protagonists)

Topic #2: bilbo, frodo, gandalf, ring, boromir, galadriel, mr (Early Journey & Ring Focus)
Interpretation: LSA also identifies character and plot-focused topics. The number of topics was automatically reduced due to data dimensions.

7. Document Clustering (K-Means)

Groups books based on TF-IDF textual similarity.

Output Example:

Fellowship: Cluster 1
Two Towers: Cluster 0
Return of the King: Cluster 2
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END

Interpretation: With k=3 (number of books), each book forms its own cluster, indicating they are sufficiently distinct textually.

8. Word Embeddings (Word2Vec)

Learns vector representations for words.

Example - Words most similar to 'frodo':

- gandalf: 0.9997
- aragorn: 0.9996
- thoden: 0.9995
- gimli: 0.9995
- sam: 0.9995
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END

Interpretation: Words semantically close to "frodo" are other major characters he interacts with. High similarity scores are typical for small, domain-specific corpora.

Example - Analogy king - man + woman = ?:

('saruman', 0.9924...)
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END

Interpretation: The result "saruman" (instead of "queen") highlights the model's learning from the limited vocabulary and specific contexts within the LoTR dialogues. It likely picks up on "power" or "authority" attributes.

Limitations

Small Corpus Size: The analysis is based on only three documents (books). This can limit the robustness and generalizability of some methods like topic modeling and word embeddings.

Dialogue Only: The analysis is performed on dialogue scripts, not the full narrative text of the books.

Simple Preprocessing: While effective, the preprocessing is standard. More advanced techniques could be explored.

VADER's Lexicon-Based Sentiment: VADER does not capture complex narrative sentiment or sarcasm effectively.

Further Work / Potential Improvements

Character-Specific Analysis: Parse dialogue to associate lines with specific characters and then use races.csv to analyze text by race or character.

Advanced Preprocessing: Custom stop-word lists, handling of character names as single tokens, part-of-speech tagging for more nuanced analysis.

Hyperparameter Tuning: Optimize parameters for LDA, LSA, K-Means for potentially more insightful results.

Evaluation Metrics: Implement quantitative metrics for topic modeling (e.g., coherence score) and clustering.

Network Analysis: Create character interaction networks based on co-occurrence in dialogue.

Alternative Sentiment Analysis Models: Explore transformer-based models for more context-aware sentiment analysis.

Full Narrative Text: Apply the pipeline to the complete books for a richer analysis.

IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END