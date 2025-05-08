import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.cluster import KMeans
import nltk # Добавим импорт nltk для nltk.data.find
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Загрузка VADER лексикона (если еще не загружен) - ИСПРАВЛЕНО
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
    print("NLTK resource 'vader_lexicon' found.")
except LookupError:
    print("NLTK resource 'vader_lexicon' not found. Downloading...")
    nltk.download('vader_lexicon')
    print("NLTK resource 'vader_lexicon' downloaded.")


def plot_top_words(tokens, top_n=20, title="Top Words"):
    """Визуализация наиболее частых слов."""
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
    """Генерация облака слов."""
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
    """Анализ частоты слов."""
    print("\n--- Word Frequency Analysis ---")
    if not processed_tokens:
        print("No processed tokens for frequency analysis.")
        return
    plot_top_words(processed_tokens, top_n=25, title="Overall Top 25 Words (Lemmatized)")
    generate_wordcloud(processed_tokens, title="Overall Word Cloud (Lemmatized)")

def analyze_ngrams(processed_tokens, n=2, top_n=15):
    """Анализ N-грамм."""
    print(f"\n--- {n}-gram Analysis ---")
    if not processed_tokens or len(processed_tokens) < n: # Импортируем create_ngrams из utils в main.py
        print(f"Not enough tokens for {n}-gram analysis.")
        return
    
    # create_ngrams будет импортирован и вызван из main.py, здесь мы ожидаем список n-грамм
    # Для этого анализа, если он вызывается из analysis.py, ему нужно передать токены, а не n-граммы
    # Либо вызывать utils.create_ngrams здесь. Предположим, что n-граммы уже созданы и переданы,
    # или, что более вероятно, эта функция должна принимать токены и вызывать create_ngrams.
    # Исходя из вашего плана, create_ngrams вызывается в utils, а analyze_ngrams в analysis.
    # Тогда analyze_ngrams должен получать уже готовые n-граммы или создавать их сам.
    # В предыдущей версии `main.py` `create_ngrams` не вызывался явно перед `analyze_ngrams`
    # Лучше, если `analyze_ngrams` принимает токены и вызывает `create_ngrams` из `utils`
    # Но для этого ему нужен доступ к `create_ngrams`.
    # Я оставлю как в предыдущем варианте, где `analyze_ngrams` ожидает токены
    # и использует `create_ngrams` из `nltk.util` (который был импортирован в utils)
    # Перенесем импорт `create_ngrams` в `analysis.py` или передадим его из `main.py`
    # Проще всего, если `analyze_ngrams` использует `nltk.ngrams` напрямую или
    # импортирует `create_ngrams` из `utils`.

    # Чтобы избежать путаницы, давайте `analyze_ngrams` будет использовать `nltk.util.ngrams` напрямую.
    # или лучше импортировать `create_ngrams` из `utils`
    # Для простоты я изменю `analyze_ngrams`, чтобы он принимал `processed_tokens`.
    # `create_ngrams` должен быть доступен (импортирован из `utils` в `main` и затем, если нужно, передан,
    # или `analysis` импортирует `create_ngrams` из `utils`).
    # В текущей структуре main.py:
    # from utils import ... create_ngrams ...
    # from analysis import ... analyze_ngrams ...
    # То есть `analyze_ngrams` не видит `create_ngrams` из `utils` напрямую.
    # Проще всего сделать так, чтобы `analyze_ngrams` сам вызывал `nltk.ngrams`.

    # Вернемся к версии, где analyze_ngrams вызывает nltk.util.ngrams, если он был импортирован
    # `from nltk.util import ngrams` (или `import nltk` и `nltk.ngrams`)
    # Я добавлю `from utils import create_ngrams` в начало `analysis.py`

    # Для чистоты кода, лучше если `main.py` передаст `create_ngrams` или `analyze_ngrams`
    # будет импортировать из `utils`.
    # Я добавлю импорт в `analysis.py`
    from utils import create_ngrams # Добавим импорт

    ngrams_list = create_ngrams(processed_tokens, n) # Используем импортированную функцию
    if not ngrams_list:
        print(f"No {n}-grams generated.")
        return
        
    ngram_counts = Counter(ngrams_list)
    most_common_ngrams = ngram_counts.most_common(top_n)
    
    print(f"Top {top_n} {n}-grams:")
    for ngram, count in most_common_ngrams:
        print(f"{' '.join(ngram)}: {count}")


def build_dtm_tfidf(raw_corpus_list):
    """Построение DTM и TF-IDF матриц."""
    print("\n--- DTM and TF-IDF ---")
    if not raw_corpus_list or not any(raw_corpus_list):
        print("Raw corpus list is empty or contains only empty documents. Cannot build DTM/TF-IDF.")
        return None, None, None

    try:
        count_vectorizer = CountVectorizer(stop_words='english', max_features=1000)
        dtm = count_vectorizer.fit_transform(raw_corpus_list)
        dtm_feature_names = count_vectorizer.get_feature_names_out()
        print("DTM shape:", dtm.shape)
    except ValueError as e:
        print(f"Error building DTM: {e}. This might happen if all documents are empty after stopword removal.")
        dtm = None
        dtm_feature_names = None

    try:
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = tfidf_vectorizer.fit_transform(raw_corpus_list)
        tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
        print("TF-IDF matrix shape:", tfidf_matrix.shape)
    except ValueError as e:
        print(f"Error building TF-IDF: {e}. This might happen if all documents are empty after stopword removal.")
        tfidf_matrix = None
        tfidf_feature_names = None

    return dtm, tfidf_matrix, (dtm_feature_names, tfidf_feature_names)


def sentiment_analysis_vader(corpus_dict):
    """Анализ тональности текста с помощью VADER."""
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
    """Тематическое моделирование с помощью LDA."""
    print("\n--- Topic Modeling (LDA) ---")
    if dtm is None or feature_names is None or dtm.shape[0] < n_topics :
        print(f"Not enough documents ({dtm.shape[0] if dtm is not None else 0}) for {n_topics} topics, or DTM/features are None. Skipping LDA.")
        return

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(dtm)

    print(f"\nTop {n_top_words} words per topic (LDA):")
    for topic_idx, topic in enumerate(lda.components_):
        top_words_indices = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_words_indices]
        print(f"Topic #{topic_idx + 1}: {', '.join(top_words)}")

def topic_modeling_lsa(tfidf_matrix, feature_names, n_topics=3, n_top_words=10):
    """Тематическое моделирование с помощью LSA (SVD)."""
    print("\n--- Topic Modeling (LSA/SVD) ---")
    if tfidf_matrix is None or feature_names is None or tfidf_matrix.shape[0] < 1: # Проверка на существование матрицы и признаков
        print(f"TF-IDF matrix or features are None, or no documents. Skipping LSA.")
        return
    
    # n_components должен быть < min(n_samples, n_features)
    # n_samples = tfidf_matrix.shape[0], n_features = tfidf_matrix.shape[1]
    max_possible_topics = min(tfidf_matrix.shape) -1 # TruncatedSVD n_components must be < min(n_samples, n_features)
    if max_possible_topics < 1:
        print(f"Cannot perform LSA: not enough samples or features. Max possible topics: {max_possible_topics}")
        return

    actual_n_topics = min(n_topics, max_possible_topics)
    if actual_n_topics < n_topics:
        print(f"Warning: Reducing number of LSA topics from {n_topics} to {actual_n_topics} due to data dimensions.")
    if actual_n_topics < 1: # Еще одна проверка после корректировки
        print(f"Cannot perform LSA with {actual_n_topics} topics. Skipping.")
        return


    lsa = TruncatedSVD(n_components=actual_n_topics, random_state=42)
    lsa.fit(tfidf_matrix)

    print(f"\nTop {n_top_words} words per topic (LSA):")
    for topic_idx, topic in enumerate(lsa.components_):
        top_words_indices = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_words_indices]
        print(f"Topic #{topic_idx + 1}: {', '.join(top_words)}")


def document_clustering_kmeans(tfidf_matrix, doc_names, n_clusters=2):
    """Кластеризация документов с помощью K-Means."""
    print("\n--- Document Clustering (K-Means) ---")
    if tfidf_matrix is None or tfidf_matrix.shape[0] < n_clusters:
        print(f"Not enough documents ({tfidf_matrix.shape[0] if tfidf_matrix is not None else 0}) for {n_clusters} clusters, or TF-IDF matrix is None. Skipping K-Means.")
        return

    # K-Means может выбросить ошибку, если n_clusters > n_samples
    actual_n_clusters = min(n_clusters, tfidf_matrix.shape[0])
    if actual_n_clusters < n_clusters:
        print(f"Warning: Reducing number of K-Means clusters from {n_clusters} to {actual_n_clusters} (number of documents).")
    if actual_n_clusters < 1: # Если после корректировки кластеров 0 (например, 0 документов)
        print("Cannot perform K-Means with 0 clusters. Skipping.")
        return


    kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init='auto')
    kmeans.fit(tfidf_matrix)
    clusters = kmeans.labels_

    print("\nDocument clusters:")
    for doc_name, cluster_id in zip(doc_names, clusters):
        print(f"{doc_name}: Cluster {cluster_id}")

    if tfidf_matrix.shape[0] > 2 and tfidf_matrix.shape[1] > 1:
        from sklearn.decomposition import PCA
        # PCA n_components must be < min(n_samples, n_features)
        pca_n_components = min(2, tfidf_matrix.shape[0], tfidf_matrix.shape[1])
        if pca_n_components < 2 and tfidf_matrix.shape[1] == 1: # Если только 1 признак, PCA до 2 невозможен
            print("PCA to 2 components not possible with current data dimensions for visualization.")
        elif pca_n_components < 1:
            print("PCA not possible for visualization.")
        else:
            # Если pca_n_components стал 1, то график будет одномерным, что не очень информативно
            # Для 2D графика нужно pca_n_components=2
            if pca_n_components == 1 and tfidf_matrix.shape[0] > 1: # если можем уменьшить до 1 компонента
                print("PCA will reduce to 1 component. Visualization might not be 2D.")
                pca = PCA(n_components=1, random_state=42)
                reduced_features = pca.fit_transform(tfidf_matrix.toarray())
                plt.figure(figsize=(8,6))
                # Одномерный scatter plot
                plt.scatter(reduced_features[:,0], [0] * len(reduced_features), c=clusters)
                for i, txt in enumerate(doc_names):
                    plt.annotate(txt, (reduced_features[i,0], 0))

            elif pca_n_components >= 2: # Стандартный случай для 2D
                pca = PCA(n_components=2, random_state=42)
                reduced_features = pca.fit_transform(tfidf_matrix.toarray())
                plt.figure(figsize=(8,6))
                scatter = plt.scatter(reduced_features[:,0], reduced_features[:,1], c=clusters)
                for i, txt in enumerate(doc_names):
                    plt.annotate(txt, (reduced_features[i,0], reduced_features[i,1]))
            
            plt.title('Document Clusters (K-Means after PCA)')
            plt.xlabel('PCA Component 1')
            if pca_n_components >=2 : plt.ylabel('PCA Component 2')
            plt.show()

def word_embeddings_example(processed_tokens_list_of_lists, target_word='frodo'):
    """Пример использования Word Embeddings (Word2Vec)."""
    print("\n--- Word Embeddings (Word2Vec Example) ---")
    if not any(processed_tokens_list_of_lists):
        print("No processed tokens for Word2Vec.")
        return

    from gensim.models import Word2Vec

    try:
        # Убедимся, что каждый элемент processed_tokens_list_of_lists не пустой
        valid_sentences_for_w2v = [s for s in processed_tokens_list_of_lists if s]
        if not valid_sentences_for_w2v:
            print("All token lists are empty. Cannot train Word2Vec model.")
            return

        model = Word2Vec(sentences=valid_sentences_for_w2v, vector_size=100, window=5, min_count=1, workers=4)
        # model.train(valid_sentences_for_w2v, total_examples=model.corpus_count, epochs=10) # train уже вызывается в конструкторе Word2Vec если есть sentences
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
            print(f"Could not perform analogy, word not in vocab: {e}")

    except Exception as e:
        print(f"Error training or using Word2Vec model: {e}")