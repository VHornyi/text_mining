import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.cluster import KMeans
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import seaborn as sns 
import pandas as pd 
from utils import create_ngrams 

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
    print("NLTK resource 'vader_lexicon' found.")
except LookupError:
    print("NLTK resource 'vader_lexicon' not found. Downloading...")
    nltk.download('vader_lexicon')
    print("NLTK resource 'vader_lexicon' downloaded.")


def plot_top_words(tokens, top_n=20, title="Top Words"):
    if not tokens:
        print("No tokens to plot.")
        return
    word_counts = Counter(tokens)
    most_common = word_counts.most_common(top_n)
    words, counts = zip(*most_common)

    plt.figure(figsize=(12, 6))
    plt.bar(words, counts)
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

def generate_wordcloud(tokens_list, title="Word Cloud"):
    if not tokens_list:
        print("No tokens to generate word cloud.")
        return
    text_for_wordcloud = " ".join(tokens_list)
    if not text_for_wordcloud.strip():
        print("Cannot generate word cloud from empty text.")
        return

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_for_wordcloud)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

def analyze_word_frequencies(processed_tokens):
    print("\n--- Word Frequency Analysis ---")
    if not processed_tokens:
        print("No processed tokens for frequency analysis.")
        return
    plot_top_words(processed_tokens, top_n=25, title="Overall Top 25 Words (Lemmatized)")
    generate_wordcloud(processed_tokens, title="Overall Word Cloud (Lemmatized)")

def analyze_ngrams(processed_tokens, n=2, top_n=15):
    print(f"\n--- {n}-gram Analysis ---")
    if not processed_tokens or len(processed_tokens) < n:
        print(f"Not enough tokens for {n}-gram analysis.")
        return

    ngrams_list = create_ngrams(processed_tokens, n) 
    if not ngrams_list:
        print(f"No {n}-grams generated.")
        return

    ngram_counts = Counter(ngrams_list)
    most_common_ngrams = ngram_counts.most_common(top_n)

    print(f"Top {top_n} {n}-grams:")
    for ngram, count in most_common_ngrams:
        print(f"{' '.join(ngram)}: {count}")

def plot_matrix_heatmap(matrix, doc_names, feature_names, title="Matrix Heatmap", max_features_to_show=50):
    if matrix is None or not doc_names or not feature_names:
        print(f"Cannot plot heatmap for {title}: Missing data.")
        return

    if hasattr(matrix, "toarray"):
        dense_matrix = matrix.toarray()
    else:
        dense_matrix = matrix

    num_features = dense_matrix.shape[1]
    if num_features > max_features_to_show:
        print(f"Displaying heatmap for the first {max_features_to_show} features of {title} (out of {num_features}).")
        dense_matrix_subset = dense_matrix[:, :max_features_to_show]
        feature_names_subset = feature_names[:max_features_to_show]
    else:
        dense_matrix_subset = dense_matrix
        feature_names_subset = feature_names

    if dense_matrix_subset.shape[1] == 0:
        print(f"No features to display in heatmap for {title} after subsetting.")
        return

    df_matrix = pd.DataFrame(dense_matrix_subset, index=doc_names, columns=feature_names_subset)

    plt.figure(figsize=(max(15, len(feature_names_subset) * 0.3), max(4, len(doc_names) * 0.5)))
    sns.heatmap(df_matrix, annot=False, cmap="viridis", linewidths=.5)
    plt.title(title)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()

def build_dtm_tfidf(raw_corpus_list, doc_names_for_heatmap):
    print("\n--- DTM and TF-IDF ---")
    if not raw_corpus_list or not any(raw_corpus_list):
        print("Raw corpus list is empty or contains only empty documents. Cannot build DTM/TF-IDF.")
        return None, None, None

    MAX_FEATURES_FOR_VECTORIZERS = 50 

    dtm = None
    dtm_feature_names = None
    tfidf_matrix = None
    tfidf_feature_names = None

    try:
        count_vectorizer = CountVectorizer(stop_words='english', max_features=MAX_FEATURES_FOR_VECTORIZERS)
        dtm = count_vectorizer.fit_transform(raw_corpus_list)
        dtm_feature_names = count_vectorizer.get_feature_names_out()
        print("DTM shape:", dtm.shape)
        if dtm is not None and dtm_feature_names is not None and doc_names_for_heatmap:
            plot_matrix_heatmap(dtm, doc_names_for_heatmap, list(dtm_feature_names), title=f"DTM Heatmap (Top {MAX_FEATURES_FOR_VECTORIZERS} Features)", max_features_to_show=MAX_FEATURES_FOR_VECTORIZERS)
    except ValueError as e:
        print(f"Error building DTM: {e}.")
        dtm = None
        dtm_feature_names = None

    try:
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=MAX_FEATURES_FOR_VECTORIZERS)
        tfidf_matrix = tfidf_vectorizer.fit_transform(raw_corpus_list)
        tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
        print("TF-IDF matrix shape:", tfidf_matrix.shape)
        if tfidf_matrix is not None and tfidf_feature_names is not None and doc_names_for_heatmap:
             plot_matrix_heatmap(tfidf_matrix, doc_names_for_heatmap, list(tfidf_feature_names), title=f"TF-IDF Heatmap (Top {MAX_FEATURES_FOR_VECTORIZERS} Features)", max_features_to_show=MAX_FEATURES_FOR_VECTORIZERS)
    except ValueError as e:
        print(f"Error building TF-IDF: {e}.")
        tfidf_matrix = None
        tfidf_feature_names = None

    return dtm, tfidf_matrix, (dtm_feature_names, tfidf_feature_names)


def sentiment_analysis_vader(corpus_dict):
    print("\n--- Sentiment Analysis (VADER) ---")
    analyzer = SentimentIntensityAnalyzer()
    sentiments = {}
    for doc_name, text in corpus_dict.items():
        if not text.strip():
            print(f"Skipping sentiment analysis for empty document: {doc_name}")
            sentiments[doc_name] = {'compound': 0, 'overall': 'NEUTRAL'}
            continue

        vs = analyzer.polarity_scores(text)
        compound_score = vs['compound']
        sentiments[doc_name] = {'compound': compound_score}
        if compound_score >= 0.05:
            sentiments[doc_name]['overall'] = 'POSITIVE'
        elif compound_score <= -0.05:
            sentiments[doc_name]['overall'] = 'NEGATIVE'
        else:
            sentiments[doc_name]['overall'] = 'NEUTRAL'
        print(f"Sentiment for {doc_name}: Compound={compound_score:.4f} ({sentiments[doc_name]['overall']})")

    names = list(sentiments.keys())
    compound_scores = [s['compound'] for s in sentiments.values()]

    plt.figure(figsize=(10, 5))
    colors = ['green' if s > 0 else 'red' if s < 0 else 'blue' for s in compound_scores]
    plt.bar(names, compound_scores, color=colors)
    plt.ylabel("Compound Sentiment Score (VADER)")
    plt.title("Sentiment Analysis of LOTR Books")
    plt.axhline(0, color='grey', lw=0.8)
    plt.show()

    return sentiments


def topic_modeling_lda(dtm, feature_names, n_topics=3, n_top_words=10):
    print("\n--- Topic Modeling (LDA) ---")
    if dtm is None or feature_names is None or dtm.shape[0] == 0 or dtm.shape[1] == 0:
        print(f"DTM is None, has no documents, or no features. Skipping LDA.")
        return
    if dtm.shape[0] < n_topics:
        print(f"Warning: Number of documents ({dtm.shape[0]}) is less than requested n_topics ({n_topics}). Reducing n_topics for LDA to {dtm.shape[0]}.")
        n_topics = dtm.shape[0]
    if n_topics == 0 : 
        print("Cannot perform LDA with 0 topics. Skipping.")
        return


    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(dtm)

    print(f"\nTop {n_top_words} words per topic (LDA):")
    for topic_idx, topic in enumerate(lda.components_):
        top_words_indices = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_words_indices]
        print(f"Topic #{topic_idx + 1}: {', '.join(top_words)}")

def topic_modeling_lsa(tfidf_matrix, feature_names, n_topics=3, n_top_words=10):
    print("\n--- Topic Modeling (LSA/SVD) ---")
    if tfidf_matrix is None or feature_names is None or tfidf_matrix.shape[0] < 1 or tfidf_matrix.shape[1] < 1:
        print(f"TF-IDF matrix or features are None, or matrix has zero dimensions. Skipping LSA.")
        return

    max_possible_topics = min(tfidf_matrix.shape)
    if tfidf_matrix.shape[0] == tfidf_matrix.shape[1]:
         max_possible_topics -=1


    if max_possible_topics < 1:
        print(f"Cannot perform LSA: not enough samples or features to select components. Max possible topics: {max_possible_topics}")
        return

    actual_n_topics = min(n_topics, max_possible_topics)
    if actual_n_topics < n_topics:
        print(f"Warning: Reducing number of LSA topics from {n_topics} to {actual_n_topics} due to data dimensions.")
    if actual_n_topics < 1:
        print(f"Cannot perform LSA with {actual_n_topics} topics after adjustment. Skipping.")
        return

    lsa = TruncatedSVD(n_components=actual_n_topics, random_state=42)
    lsa.fit(tfidf_matrix)

    print(f"\nTop {n_top_words} words per topic (LSA):")
    for topic_idx, topic in enumerate(lsa.components_):
        top_words_indices = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_words_indices]
        print(f"Topic #{topic_idx + 1}: {', '.join(top_words)}")


def document_clustering_kmeans(tfidf_matrix, doc_names, n_clusters=2):
    print("\n--- Document Clustering (K-Means) ---")
    if tfidf_matrix is None or tfidf_matrix.shape[0] == 0:
        print("TF-IDF matrix is None or has no documents. Skipping K-Means.")
        return

    
    actual_n_clusters = min(n_clusters, tfidf_matrix.shape[0])
    if actual_n_clusters < n_clusters:
        print(f"Warning: Reducing number of K-Means clusters from {n_clusters} to {actual_n_clusters} (number of documents).")
    if actual_n_clusters < 1: 
        print("Cannot perform K-Means with 0 clusters. Skipping.")
        return
    


    kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init='auto')
    kmeans.fit(tfidf_matrix)
    clusters = kmeans.labels_

    print("\nDocument clusters:")
    for doc_name, cluster_id in zip(doc_names, clusters):
        print(f"{doc_name}: Cluster {cluster_id}")

    
    if tfidf_matrix.shape[0] > 1 and tfidf_matrix.shape[1] > 1: 
        from sklearn.decomposition import PCA
       
        pca_n_components = min(2, tfidf_matrix.shape[0], tfidf_matrix.shape[1])

        if pca_n_components < 1 :
             print("PCA not possible for visualization with current data dimensions.")
        else:
            pca = PCA(n_components=pca_n_components, random_state=42)
            reduced_features = pca.fit_transform(tfidf_matrix.toarray()) 

            plt.figure(figsize=(8,6))
            if pca_n_components == 1:
                print("PCA reduced to 1 component. Visualization will be 1D.")
                plt.scatter(reduced_features[:,0], [0] * len(reduced_features), c=clusters)
                for i, txt in enumerate(doc_names):
                    plt.annotate(txt, (reduced_features[i,0], 0.01)) 
                plt.yticks([]) 
                plt.ylabel("")
            else: 
                scatter = plt.scatter(reduced_features[:,0], reduced_features[:,1], c=clusters)
                for i, txt in enumerate(doc_names):
                    plt.annotate(txt, (reduced_features[i,0], reduced_features[i,1]))
                plt.ylabel('PCA Component 2')

            plt.title('Document Clusters (K-Means after PCA)')
            plt.xlabel('PCA Component 1')
            plt.show()
    elif tfidf_matrix.shape[0] == 1:
        print("PCA visualization skipped: Only one document.")


def word_embeddings_example(processed_tokens_list_of_lists, target_word='frodo'):
    print("\n--- Word Embeddings (Word2Vec Example) ---")
    if not any(processed_tokens_list_of_lists):
        print("No processed tokens for Word2Vec.")
        return

    from gensim.models import Word2Vec

    try:
        valid_sentences_for_w2v = [s for s in processed_tokens_list_of_lists if s]
        if not valid_sentences_for_w2v:
            print("All token lists are empty. Cannot train Word2Vec model.")
            return

        
        model = Word2Vec(sentences=valid_sentences_for_w2v, vector_size=100, window=5, min_count=1, workers=4, epochs=10)
        print("Word2Vec model trained.")

        if target_word in model.wv:
            print(f"\nWords most similar to '{target_word}':")
            similar_words = model.wv.most_similar(target_word, topn=5)
            for word, score in similar_words:
                print(f"- {word}: {score:.4f}")
        else:
            print(f"Word '{target_word}' not in Word2Vec vocabulary.")

        try:
            if all(w in model.wv for w in ['king', 'man', 'woman']):
                analogy = model.wv.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
                print(f"\nAnalogy 'king - man + woman = ?': {analogy[0] if analogy else 'Not found'}")
            else:
                missing_analogy_words = [w for w in ['king', 'man', 'woman'] if w not in model.wv]
                if missing_analogy_words:
                    print(f"Cannot perform analogy, words not in vocab: {', '.join(missing_analogy_words)}")
        except KeyError as e: 
            print(f"Could not perform analogy, word not in vocab or no similar vectors: {e}")
        except Exception as e_sim: 
            print(f"Could not perform analogy due to an error: {e_sim}")


    except Exception as e:
        print(f"Error training or using Word2Vec model: {e}")
