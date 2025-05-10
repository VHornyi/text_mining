from utils import load_races, load_lotr_texts, preprocess_text
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

    
    analyze_word_frequencies(all_processed_tokens)

    analyze_ngrams(all_processed_tokens, n=2, top_n=15)
    analyze_ngrams(all_processed_tokens, n=3, top_n=10)

    
    dtm, tfidf_matrix, feature_names_tuple = build_dtm_tfidf(raw_texts_by_book_list, doc_names)
    dtm_feature_names, tfidf_feature_names = (None, None)
    if feature_names_tuple:
        dtm_feature_names, tfidf_feature_names = feature_names_tuple

    
    sentiment_analysis_vader(lotr_texts_dict)

    if dtm is not None and dtm_feature_names is not None:
        actual_lda_topics = min(3, dtm.shape[0]) if dtm.shape[0] > 0 else 1
        if actual_lda_topics > 0: 
             topic_modeling_lda(dtm, dtm_feature_names, n_topics=actual_lda_topics, n_top_words=7)
        else:
            print("Skipping LDA due to insufficient documents for topic modeling.")
    else:
        print("Skipping LDA due to issues with DTM or feature names.")

    if tfidf_matrix is not None and tfidf_feature_names is not None:
        if tfidf_matrix.shape[0] > 0 and tfidf_matrix.shape[1] > 0:
            
            topic_modeling_lsa(tfidf_matrix, tfidf_feature_names, n_topics=3, n_top_words=7)
        else:
            print("Skipping LSA due to TF-IDF matrix dimensions being too small after feature reduction.")
    else:
        print("Skipping LSA due to issues with TF-IDF matrix or feature names.")

    if tfidf_matrix is not None and doc_names:
        num_docs = tfidf_matrix.shape[0]
        
        actual_n_clusters = min(3, num_docs) if num_docs > 0 else 0
        if actual_n_clusters > 0 : 
             if actual_n_clusters == 1 and num_docs == 1:
                 print("Skipping K-Means: Only one document, clustering is trivial.")
             else:
                document_clustering_kmeans(tfidf_matrix, doc_names, n_clusters=actual_n_clusters)
        else:
            print("Skipping K-Means clustering as there are no documents or clusters to form.")
    else:
        print("Skipping K-Means clustering due to issues with TF-IDF matrix or no document names.")

    if list_of_token_lists_for_word2vec:
         word_embeddings_example(list_of_token_lists_for_word2vec, target_word='frodo')
    else:
        print("Skipping Word Embeddings example as there are no processed tokens lists.")

    print("\n--- Text Mining Project Finished ---")

if __name__ == "__main__":
    main()
