from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
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


def update_cached_embeddings(max_threads=8):
    """
    Load professional vocabulary terms from JSON, compute their embeddings in parallel,
    and save the resulting word embeddings along with their translated forms to disk.

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
        return term, get_embedding(term).tolist()

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {executor.submit(process_term, term): term for term in icd10_vocab}

        for idx, future in enumerate(as_completed(futures), start=1):
            term, embedding = future.result()
            embedding_dict[term] = embedding

            if idx % 10 == 0 or idx == total:
                print(f"Progress: {idx}/{total} terms processed ({(idx/total)*100:.1f}%)")

    words = [translation_dict.get(word, word) for word in embedding_dict.keys()]
    embeddings = list(embedding_dict.values())

    # Save to backend directory
    vectors_path = os.path.join("icd10_vectors.npz")
    np.savez(vectors_path, words=words, embeddings=embeddings)
    print("\033[92mSUCCESS: Cached Embeddings Updated and stored with translated words.\033[0m")


_embedding_cache = {
    "embedding_dict": None,
    "icd10_vocab": None,
    "pro_vectors": None,
}


def load_embeddings():
    """
    Load the cached embeddings and corresponding vocabulary from disk if not already loaded,
    store them in memory for efficient reuse.

    Returns:
        tuple: (embedding_dict, icd10_vocab, pro_vectors)
            - embedding_dict (dict): Maps terms to their numpy vector embeddings.
            - icd10_vocab (list): List of ICD-10 vocabulary terms.
            - pro_vectors (np.ndarray): Array of embeddings corresponding to icd10_vocab.
    """
    if _embedding_cache["embedding_dict"] is None:
        vectors_path = os.path.join("icd10_vectors.npz")
        data = np.load(vectors_path, allow_pickle=True)
        words = data["words"]
        embeddings = data["embeddings"]

        embedding_dict = dict(zip(words, embeddings))
        embedding_dict = {term: np.array(vec) for term, vec in embedding_dict.items()}

        icd10_vocab = list(embedding_dict.keys())
        pro_vectors = np.array([embedding_dict[term] for term in icd10_vocab])

        _embedding_cache["embedding_dict"] = embedding_dict
        _embedding_cache["icd10_vocab"] = icd10_vocab
        _embedding_cache["pro_vectors"] = pro_vectors

    return (_embedding_cache["embedding_dict"],
            _embedding_cache["icd10_vocab"],
            _embedding_cache["pro_vectors"])


def get_icd10_similarities(query, top_k=3):
    """
    Given a query string, compute its embedding and find the top-k most similar
    ICD-10 vocabulary terms based on cosine similarity.

    Args:
        query (str): Layman or input query string.
        top_k (int): Number of top similar terms to return.

    Returns:
        tuple: (list of terms, list of similarity scores)
            - terms (list): Top-k most similar ICD-10 terms.
            - similarity scores (list): Corresponding similarity scores.
    """
    embedding_dict, icd10_vocab, pro_vectors = load_embeddings()

    layman_vec = get_embedding(query)

    projected_vec = layman_vec

    similarities = cosine_similarity([projected_vec], pro_vectors).flatten()

    sorted_indices = np.argsort(similarities)[::-1]

    sorted_vocab = [icd10_vocab[i] for i in sorted_indices]
    sorted_similarities = similarities[sorted_indices]

    return sorted_vocab[:top_k], sorted_similarities[:top_k]


def search_icd10(query, top_k=3, verbose=False):
    """
    Search the ICD-10 vocabulary for terms similar to the query.
    Optionally return icd10 descriptions alongside codes and scores.

    Args:
        query (str): input query to search.
        top_k (int): Number of top matches to return.
        verbose (bool): If True, also return ICD-10 descriptions.

    Returns:
        If verbose:
            tuple: (codes, scores, descriptions)
        Else:
            tuple: (codes, scores)
    """
    codes, scores = get_icd10_similarities(query, top_k)
    codes = [x.item() if hasattr(x, 'item') else x for x in codes]
    scores = [x.item() if hasattr(x, 'item') else x for x in scores]

    if verbose:
        # Try multiple possible paths for the descriptions file
        descriptions_path = os.path.join("icd10_descriptions.json")
        
        for path in possible_paths:
            if os.path.exists(path):
                descriptions_path = path
                break
        
        if descriptions_path is None:
            raise FileNotFoundError(f"Could not find icd10_descriptions.json in any of these locations: {possible_paths}")
        
        with open(descriptions_path, "r", encoding="utf-8") as f:
            icd10_description_dict = json.load(f)
            descriptions = [icd10_description_dict.get(code) for code in codes]
        return codes, scores, descriptions

    return codes, scores


# >>>>>>>>>>>>>> NOT IN USE <<<<<<<<<<<<<<<<<<<

# def update_transformation_vector():
#     """
#     Compute and cache a transformation vector from layman embeddings to professional embeddings
#     by averaging the difference vectors between mapped term pairs.
#     """
#     transformations = []

#     for layman_term, pro_term in layman_to_pro.items():
#         lay_vec = get_embedding(layman_term)
#         pro_vec = get_embedding(pro_term)
#         diff = pro_vec - lay_vec
#         transformations.append(diff)

#     transformation_vector = np.mean(transformations, axis=0)
#     np.save("transformation_vector.npy", transformation_vector)
#     print("\033[92mSUCCESS: Cached Transformation Vector Updated and stored.\033[0m")


# update_cached_embeddings()