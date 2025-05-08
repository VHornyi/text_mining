from utils import load_races, load_lotr_texts, preprocess_text, create_ngrams
from analysis import (
    analyze_word_frequencies,
    analyze_ngrams,
    build_dtm_tfidf,
    sentiment_analysis_vader,
    topic_modeling_lda,
    topic_modeling_lsa,
    document_clustering_kmeans,
    word_embeddings_example
)

def main():
    print("Starting Text Mining Project for Lord of the Rings...")


    lotr_texts_dict, combined_lotr_text = load_lotr_texts()

    if not combined_lotr_text.strip():
        print("No text data loaded. Exiting.")
        return

    # 2. Czyszczenie danych tekstowych 
    print("\n--- Preprocessing Text ---")
    all_processed_tokens = preprocess_text(combined_lotr_text)
    print(f"Total tokens after preprocessing: {len(all_processed_tokens)}")
    if all_processed_tokens:
        print(f"Sample processed tokens: {all_processed_tokens[:10]}")

    
    processed_texts_by_book_tokens = {}
    raw_texts_by_book_list = [] 
    doc_names = list(lotr_texts_dict.keys())

    for book_name, text_content in lotr_texts_dict.items():
        processed_texts_by_book_tokens[book_name] = preprocess_text(text_content)
        raw_texts_by_book_list.append(text_content if text_content else "") 
    list_of_token_lists_for_word2vec = [tokens for tokens in processed_texts_by_book_tokens.values() if tokens]


    # 3. Podstawowe metody analizy danych tekstowych
    analyze_word_frequencies(all_processed_tokens)

    analyze_ngrams(all_processed_tokens, n=2, top_n=15)
    analyze_ngrams(all_processed_tokens, n=3, top_n=10)


    dtm, tfidf_matrix, feature_names_tuple = build_dtm_tfidf(raw_texts_by_book_list)
    dtm_feature_names, tfidf_feature_names = (None, None)
    if feature_names_tuple:
        dtm_feature_names, tfidf_feature_names = feature_names_tuple


    # 4. Nienadzorowane uczenie maszynowe
   
    sentiment_analysis_vader(lotr_texts_dict)

    if dtm is not None and dtm_feature_names is not None:
        topic_modeling_lda(dtm, dtm_feature_names, n_topics=3, n_top_words=7)
    else:
        print("Skipping LDA due to issues with DTM.")
        
    if tfidf_matrix is not None and tfidf_feature_names is not None:
        topic_modeling_lsa(tfidf_matrix, tfidf_feature_names, n_topics=3, n_top_words=7)
    else:
        print("Skipping LSA due to issues with TF-IDF matrix.")


    # n_clusters 
    if tfidf_matrix is not None:
        document_clustering_kmeans(tfidf_matrix, doc_names, n_clusters=min(3, len(doc_names))) # Кластеризуем книги
    else:
        print("Skipping K-Means clustering due to issues with TF-IDF matrix.")


    # Word Embeddings 
    if list_of_token_lists_for_word2vec:
         word_embeddings_example(list_of_token_lists_for_word2vec, target_word='frodo')
    else:
        print("Skipping Word Embeddings example as there are no processed tokens lists.")


    print("\n--- Text Mining Project Finished ---")

if __name__ == "__main__":
    main()