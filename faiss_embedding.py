from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Get the backend directory path
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = SentenceTransformer("omarelshehy/Arabic-STS-Matryoshka")

def get_embedding(text):
    """
    Encode a given text string into a vector embedding using the pre-trained model.

    Args:
        text (str): Input text to encode.

    Returns:
        np.ndarray: Vector embedding representation of the input text.
    """
    return model.encode(text)

def faiss_update_cached_embeddings(max_threads=8):
    """
    Load professional vocabulary terms from JSON, compute their normalized embeddings in parallel,
    and save the resulting word embeddings along with their translated forms to disk for FAISS.

    Args:
        max_threads (int): Number of worker threads to use for parallel embedding computation.
    """
    translated_icd10_path = os.path.join("archive", "icd10data", "translated_icd10.json")
    with open(translated_icd10_path, "r", encoding="utf-8") as f:
        translation_dict = json.load(f)
        icd10_vocab = list(translation_dict.keys())

    print(f"Loaded {len(icd10_vocab)} terms.")
    total = len(icd10_vocab)
    embedding_dict = {}

    def process_term(term):
        embedding = get_embedding(term)
        # Normalize the embedding for FAISS (required for cosine similarity with Inner Product)
        embedding = embedding / np.linalg.norm(embedding)
        return term, embedding.tolist()

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {executor.submit(process_term, term): term for term in icd10_vocab}

        for idx, future in enumerate(as_completed(futures), start=1):
            term, embedding = future.result()
            embedding_dict[term] = embedding

            if idx % 10 == 0 or idx == total:
                print(f"Progress: {idx}/{total} terms processed ({(idx/total)*100:.1f}%)")

    words = [translation_dict.get(word, word) for word in embedding_dict.keys()]
    embeddings = np.array(list(embedding_dict.values()), dtype=np.float32)  # Ensure float32 for FAISS

    # Save to backend directory
    vectors_path = os.path.join("icd10_vectors_faiss.npz")
    np.savez(vectors_path, words=words, embeddings=embeddings)
    print("\033[92mSUCCESS: FAISS Cached Embeddings Updated and stored with translated words.\033[0m")

_embedding_cache = {
    "embedding_dict": None,
    "icd10_vocab": None,
    "pro_vectors": None,
    "faiss_index": None,
}

def faiss_load_embeddings():
    """
    Load the cached normalized embeddings, vocabulary, and FAISS index from disk if not already loaded,
    store them in memory for efficient reuse.

    Returns:
        tuple: (embedding_dict, icd10_vocab, pro_vectors, faiss_index)
            - embedding_dict (dict): Maps terms to their numpy vector embeddings.
            - icd10_vocab (list): List of ICD-10 vocabulary terms.
            - pro_vectors (np.ndarray): Array of embeddings corresponding to icd10_vocab.
            - faiss_index (faiss.Index): FAISS index for similarity search.
    """
    if _embedding_cache["embedding_dict"] is None:
        vectors_path = os.path.join("icd10_vectors_faiss.npz")
        data = np.load(vectors_path, allow_pickle=True)
        words = data["words"]
        embeddings = data["embeddings"].astype(np.float32)  # Ensure float32 for FAISS

        embedding_dict = dict(zip(words, embeddings))
        embedding_dict = {term: np.array(vec) for term, vec in embedding_dict.items()}

        icd10_vocab = list(embedding_dict.keys())
        pro_vectors = np.array([embedding_dict[term] for term in icd10_vocab], dtype=np.float32)

        # Create FAISS index (IndexFlatIP for normalized vectors = cosine similarity)
        dimension = pro_vectors.shape[1]
        faiss_index = faiss.IndexFlatIP(dimension)
        faiss_index.add(pro_vectors)  # Add normalized vectors to the index

        _embedding_cache["embedding_dict"] = embedding_dict
        _embedding_cache["icd10_vocab"] = icd10_vocab
        _embedding_cache["pro_vectors"] = pro_vectors
        _embedding_cache["faiss_index"] = faiss_index

    return (_embedding_cache["embedding_dict"],
            _embedding_cache["icd10_vocab"],
            _embedding_cache["pro_vectors"],
            _embedding_cache["faiss_index"])

def faiss_get_icd10_similarities(query, top_k=3):
    """
    Given a query string, compute its normalized embedding and find the top-k most similar
    ICD-10 vocabulary terms using FAISS.

    Args:
        query (str): Layman or input query string.
        top_k (int): Number of top similar terms to return.

    Returns:
        tuple: (list of terms, list of similarity scores)
            - terms (list): Top-k most similar ICD-10 terms.
            - similarity scores (list): Corresponding similarity scores.
    """
    embedding_dict, icd10_vocab, pro_vectors, faiss_index = faiss_load_embeddings()

    layman_vec = get_embedding(query)
    # Normalize query vector for FAISS
    layman_vec = layman_vec / np.linalg.norm(layman_vec)
    layman_vec = layman_vec.reshape(1, -1).astype(np.float32)  # Reshape for FAISS

    # Search FAISS index
    similarities, indices = faiss_index.search(layman_vec, top_k)

    sorted_vocab = [icd10_vocab[i] for i in indices[0]]
    sorted_similarities = similarities[0]

    return sorted_vocab, sorted_similarities

def faiss_search_icd10(query, top_k=3, verbose=False):
    """
    Search the ICD-10 vocabulary for terms similar to the query using FAISS.
    Optionally return icd10 descriptions alongside codes and scores.

    Args:
        query (str): Input query to search.
        top_k (int): Number of top matches to return.
        verbose (bool): If True, also return ICD-10 descriptions.

    Returns:
        If verbose:
            tuple: (codes, scores, descriptions)
        Else:
            tuple: (codes, scores)
    """
    codes, scores = faiss_get_icd10_similarities(query, top_k)
    codes = [x.item() if hasattr(x, 'item') else x for x in codes]
    scores = [x.item() if hasattr(x, 'item') else x for x in scores]

    if verbose:
        descriptions_path = os.path.join("icd10_descriptions.json")
        
        if not os.path.exists(descriptions_path):
            raise FileNotFoundError(f"Could not find icd10_descriptions.json at {descriptions_path}")
        
        with open(descriptions_path, "r", encoding="utf-8") as f:
            icd10_description_dict = json.load(f)
            descriptions = [icd10_description_dict.get(code) for code in codes]
        return codes, scores, descriptions

    return codes, scores

if __name__ == "__main__":
    faiss_update_cached_embeddings()
    