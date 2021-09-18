import typing
from sentence_transformers import SentenceTransformer, util

__all__ = ["get_similar_commits"]


def get_similar_commits(
    model: SentenceTransformer,
    query: str,
    commits: typing.List[typing.Dict[str, str]],
    n_matches: int = 10,
) -> typing.List[typing.Dict[str, str]]:
    """
    Get the n_matches most similar commits to the query commit.

    Arguments
    ---------
    model: SentenceTransformer
        The model to use for the similarity computation.
    query: str
        The query string to compare to.
    commits: typing.List[typing.Dict[str, str]]
        A list of commits to compare to the query.
    n_matches: int (default=10)
        The number of matches to return.

    Returns
    -------
    typing.List[typing.Dict[str, str]]
        A list of the n_matches most similar commits to the query.
    """

    # Get all the commit messages.
    commit_messages = [commit["message"] for commit in commits]

    # Encode all the commit messages.
    commit_embeddings = model.encode(
        commit_messages, convert_to_tensor=True, normalize_embeddings=True
    )

    # Encode the query.
    query_embedding = model.encode(
        [query], convert_to_tensor=True, normalize_embeddings=True
    )

    # Use cosine similarity to find the most similar commits.
    # Since the returned tensors are normalized, we can use the faster dot product here. Reference:
    # https://www.sbert.net/examples/applications/computing-embeddings/README.html?highlight=faster%20dot-product
    cosine_scores = util.dot_score(query_embedding, commit_embeddings).squeeze()

    # Get the indices of the most similar commits.
    sorted_scores_indices = cosine_scores.argsort().tolist()[::-1][:n_matches]

    # Get the n_matches most similar commits.
    sorted_commits = [commits[i] for i in sorted_scores_indices]
    for i, commit in enumerate(sorted_commits):
        commit["score"] = f"{cosine_scores[sorted_scores_indices[i]].numpy():.2f}"

    return sorted_commits
