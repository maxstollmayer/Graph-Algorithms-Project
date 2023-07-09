"""utility functions for generating and analysing graphs of books"""

from typing import Any

import matplotlib as mpl
from matplotlib import pyplot as plt
from netgraph import Graph
import networkx as nx
from nltk import word_tokenize, pos_tag, download
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import numpy as np
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extract_text(book_path: str) -> list[str]:
    """Extracts all text from a pdf as a list of page texts"""
    text_list = []
    with pdfplumber.open(book_path + ".pdf") as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            text_list.append(text)
    return text_list


def save_text(pages: list[str], file_path: str) -> None:
    """Saves a list of page texts to a txt file"""
    with open(file_path + ".txt", "w", encoding="utf-8") as file:
        for page in pages:
            cleaned = page.strip()
            cleaned = cleaned.replace("\n", " ")
            cleaned = " ".join(cleaned.split())
            file.write(cleaned)
            file.write("\n\n")


def get_pos(tag: str) -> str | None:
    """Converts between position tags and wordnet tags."""
    match tag[0]:
        case "J":
            return wordnet.ADJ
        case "R":
            return wordnet.ADV
        case "N":
            return wordnet.NOUN
        case "V":
            return wordnet.VERB
        case _:
            return None


def preprocess(text: str) -> str:
    """Tokenizes and lemmatizes the text."""
    tokens = [token for token in word_tokenize(text.lower())]
    tagged = [(word, get_pos(tag)) for word, tag in pos_tag(tokens)]
    lemmatizer = WordNetLemmatizer()
    lemmatized = [
        lemmatizer.lemmatize(word, pos=tag) for (word, tag) in tagged if tag is not None
    ]
    return " ".join(lemmatized)


def check_nltk_datasets() -> None:
    """Downloads relevant nltk datasets if necessary"""
    download("wordnet", quiet=True)
    download("omw-1.4", quiet=True)
    download("punkt", quiet=True)
    download("averaged_perceptron_tagger", quiet=True)
    download("stopwords", quiet=True)


def get_pages(book_path: str, delimiter: str) -> list[str]:
    """Reads the book from the extracted pdf txt file"""

    # check for necessary datasets
    check_nltk_datasets()

    # load pages
    with open(book_path, "r", encoding="utf-8") as f:
        pages = f.read().split(delimiter)

    return pages


def get_weights(pages: list[str]) -> np.ndarray:
    """Calculates cosine similarities between pages"""

    # preprocess pages
    data = [preprocess(page) for page in pages]

    # vectorize tokens to frequencies
    vectorizer = TfidfVectorizer()
    freqs = vectorizer.fit_transform(data)

    # calculate cosine similarities between pages
    n = len(pages)
    weights = np.zeros((n, n))
    for i, j in np.ndindex(weights.shape):
        weights[i, j] = cosine_similarity(freqs.getrow(i), freqs.getrow(j)).flatten()[0]

    return weights


def generate_graph(book_path: str, delimiter: str = "\n\n") -> nx.Graph:
    """
    Generates graph of the given book with pages as nodes
    and edges weighted by the cosine similarity
    """

    # load book and calculate weights
    pages = get_pages(book_path, delimiter)
    weights = get_weights(pages)

    # generate fully connected, weighted graph
    G = nx.complete_graph(len(pages))
    for i, j in G.edges:
        G.edges[i, j]["weight"] = weights[i, j]

    return G


def get_graph(
    book: str,
    book_folder: str = "books/",
    graph_folder: str = "graphs/",
    save: bool = True,
    delimiter: str = "\n\n",
) -> nx.Graph:
    """Retrieves graph if it exists or generates it"""
    graph_path = graph_folder + book + ".gml"
    book_path = book_folder + book + ".txt"
    try:
        G = nx.read_gml(graph_path)
        remap = {str(i): i for i in range(len(G))}
        G = nx.relabel_nodes(G, remap)
    except (FileNotFoundError, nx.NetworkXError):
        G = generate_graph(book_path, delimiter)
        if save:
            nx.write_gml(G, graph_path)
    return G


def generate_threshold_graph(
    book_path: str, threshold: float, delimiter: str = "\n\n"
) -> nx.Graph:
    """Returns graph with only unweighted edges that lie above the threshold"""

    # load book and calculate weights
    pages = get_pages(book_path, delimiter)
    weights = get_weights(pages)
    n = len(pages)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if weights[i, j] >= threshold:
                G.add_edge(i, j)
    return G


def display_weight_stats(G: nx.Graph, bins: int = 100) -> None:
    """Display min, mean and max values and a histogram of the weights."""
    stats = np.array([G.edges[edge]["weight"] for edge in G.edges])

    print("minimum", stats.min())
    print("   mean", stats.mean(), "+-", stats.std())
    print("maximum", stats.max())

    plt.hist(stats, bins=bins)
    plt.title("weight histogram")
    plt.show()


def get_comm_dict(communities: list[set[int]]) -> dict[Any, int]:
    """Convert community set to node dictionary."""
    return {node: i for (i, community) in enumerate(communities) for node in community}


def plot_communities(
    G: nx.Graph, communities: list[set[int]], title: str | None = None
) -> None:
    """Plot the communities of a graph."""
    comm_dict = get_comm_dict(communities)
    colors = mpl.colormaps["tab10"].colors[: len(communities)]  # type: ignore
    node_color = {node: colors[i] for node, i in comm_dict.items()}
    max_weight = np.array([G.edges[edge]["weight"] for edge in G.edges]).max()
    edge_dict = {(i, j): G.edges[(i, j)]["weight"] / max_weight for (i, j) in G.edges}

    Graph(
        G,
        node_color=node_color,
        node_edge_width=0,
        edge_alpha=edge_dict,
        edge_width=edge_dict,
        node_layout="community",
        node_layout_kwargs=dict(node_to_community=comm_dict),
        node_labels=True,
    )
    if title is not None:
        plt.title(title)
    plt.show()


def naive_sequence(G: nx.Graph, root: Any):
    """Returns reading sequence that starts at root and goes to highest similarity in each step"""
    visited = [root]
    for i in range(len(G) - 1):
        weights = [
            (node, G.edges[(visited[i], node)]["weight"])
            for node in G
            if node not in visited
        ]
        new = max(weights, key=(lambda x: x[1]))
        visited.append(new[0])
    return visited


def cosine_error(sequence: list[int]) -> float:
    """
    Calculates the cosine similarity of a page sequence
    to the standard reading sequence 1 2 3...
    """
    ref = np.array(range(len(sequence))).reshape(1, -1)
    seq = np.array(sequence).reshape(1, -1)
    return cosine_similarity(seq, ref)[0, 0]


def l1_error(sequence: list[int]) -> float:
    """
    Calculates the L1 distance of a page sequence
    to the standard reading sequence 1 2 3...
    """
    ref = np.arange(len(sequence))
    seq = np.array(sequence)
    return np.sum(np.abs(ref - seq))


def l2_error(sequence: list[int]) -> float:
    """
    Calculates the Euclidean distance of a page sequence
    to the standard reading sequence 1 2 3...
    """
    ref = np.arange(len(sequence))
    seq = np.array(sequence)
    return np.sqrt(np.sum((ref - seq) ** 2))


def linf_error(sequence: list[int]) -> float:
    """
    Calculates the Euclidean distance of a page sequence
    to the standard reading sequence 1 2 3...
    """
    ref = np.arange(len(sequence))
    seq = np.array(sequence)
    return np.max(np.abs(ref - seq))


def plot_sequence_errors(books: list[str], sequences: list[list[int]]) -> None:
    """Plots different errors from the generated reading sequences"""
    l1_errors = np.array([l1_error(sequence) for sequence in sequences])
    l2_errors = np.array([l2_error(sequence) for sequence in sequences])
    linf_errors = np.array([linf_error(sequence) for sequence in sequences])
    cosine_errors = np.array([cosine_error(sequence) for sequence in sequences])

    plt.title("normalized errors")
    plt.plot(books, l1_errors / np.max(l1_errors), label="l1")
    plt.plot(books, l2_errors / np.max(l2_errors), label="l2")
    plt.plot(books, linf_errors / np.max(linf_errors), label="linf")
    plt.plot(books, cosine_errors / np.max(cosine_errors), label="cosine")
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()
