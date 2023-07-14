"""utility functions for generating and analysing graphs of books"""

from random import sample
from typing import Any

from matplotlib import pyplot as plt
import networkx as nx
from nltk import word_tokenize, pos_tag, download
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import numpy as np
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


NODE_ATTR_NUM = "page_num"
NODE_ATTR_CHARS = "num_chars"
NODE_ATTR_WORDS = "num_words"
EDGE_ATTR = "similarity"


def extract_text(book_name: str) -> list[str]:
    """Extracts all text from a pdf as a list of page texts"""
    text_list = []
    with pdfplumber.open(book_name + ".pdf") as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            text_list.append(text)
    return text_list


def save_text(pages: list[str], file_name: str) -> None:
    """Saves a list of page texts to a txt file"""
    with open(file_name + ".txt", "w", encoding="utf-8") as file:
        for page in pages:
            cleaned = page.strip()
            cleaned = cleaned.replace("\n", " ")
            cleaned = " ".join(cleaned.split())
            file.write(cleaned)
            file.write("\n\n")


def split_text(words: list[str], chars_per_page: int) -> list[str]:
    """Splits list of words into pages that do not surpass the character limit"""

    pages = []
    current_page = ""

    for word in words:
        # append word to current page if there is space
        if len(current_page) + len(word) + 1 <= chars_per_page:
            current_page += word + " "

        # start new page otherwise
        else:
            pages.append(current_page.strip())
            current_page = word + " "

    # add remaining text as last page
    if current_page:
        pages.append(current_page.strip())

    return pages


def split_into_pages(text: str, num_pages: int) -> list[str]:
    """
    Returns text split into the specified number of pages
    Does not surpass the specified number but may be lower
    """

    assert (
        num_pages > 0
    ), f"The number of pages has to be a positive integer, not {num_pages}."
    chars_per_page = len(text) // num_pages + 1
    words = text.split()
    pages = split_text(words, chars_per_page)

    while len(pages) > num_pages:
        overflow = len(pages) - num_pages
        chars_per_page += len(" ".join(pages[-overflow])) // 100 + 1
        pages = split_text(words, chars_per_page)

    return pages


def get_pages(
    book_path: str, num_pages: int | None = None, page_delimiter: str = "\n\n"
) -> list[str]:
    """Reads the book from the extracted pdf txt file"""

    # load book
    with open(book_path, "r", encoding="utf-8") as file:
        text = file.read()

    # split into specified number of pages
    if num_pages is not None:
        return split_into_pages(text, num_pages)

    # keep original pages
    return text.split(page_delimiter)


def check_nltk_datasets() -> None:
    """Downloads relevant nltk datasets if necessary"""
    download("wordnet", quiet=True)
    download("omw-1.4", quiet=True)
    download("punkt", quiet=True)
    download("averaged_perceptron_tagger", quiet=True)
    download("stopwords", quiet=True)


def to_wordnet(tag: str) -> str | None:
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

    # check for necessary datasets
    check_nltk_datasets()

    tokens = [token for token in word_tokenize(text.lower())]
    tagged = [(word, to_wordnet(tag)) for word, tag in pos_tag(tokens)]
    lemmatizer = WordNetLemmatizer()
    lemmatized = [
        lemmatizer.lemmatize(word, pos=tag) for (word, tag) in tagged if tag is not None
    ]
    return " ".join(lemmatized)


def get_weights(pages: list[str]) -> np.ndarray:
    """Calculates cosine similarities between pages"""

    # preprocess pages
    data = [preprocess(page) for page in pages]

    # vectorize tokens to frequencies
    vectorizer = TfidfVectorizer()
    freqs = vectorizer.fit_transform(data)

    # calculate cosine similarities between pages
    num_pages = len(pages)
    weights = np.zeros((num_pages, num_pages))
    for i, j in np.ndindex(weights.shape):
        weights[i, j] = cosine_similarity(freqs.getrow(i), freqs.getrow(j)).flatten()[0]

    return weights


def generate_graph(pages: list[str]) -> nx.Graph:
    """
    Generates graph of the given pages as nodes
    with the following attributes:
    - page number
    - number of characters
    - number of words
    and edges weighted by the cosine similarity
    """

    weights = get_weights(pages)

    # generate fully connected, weighted graph
    graph = nx.complete_graph(len(pages))
    for node in graph.nodes:
        graph.nodes[node][NODE_ATTR_NUM] = int(node)
        graph.nodes[node][NODE_ATTR_CHARS] = len(pages[node])
        graph.nodes[node][NODE_ATTR_WORDS] = len(pages[node].split())
        # TODO: more attributes? e.g. page ends with punctuation or not
    for i, j in graph.edges:
        graph.edges[i, j][EDGE_ATTR] = weights[i, j]

    return graph


def get_graph(
    text: str, num_pages: int | None = None, page_delimiter: str = "\n\n"
) -> nx.Graph:
    """Generates whole graph of given book in the specified number of pages"""

    if num_pages is not None:
        # split into specified number of pages
        pages = split_into_pages(text, num_pages)
    else:
        # keep original pages
        pages = text.split(page_delimiter)

    return generate_graph(pages)


def get_subgraphs(
    text: str, num_nodes: int, page_delimiter: str = "\n\n"
) -> list[nx.Graph]:
    """Returns list of subgraphs of required number of nodes if book is longer"""
    pages = text.split(page_delimiter)
    num_pages = len(pages)
    if num_nodes <= num_pages:
        return [generate_graph(pages)]

    graphs = []
    max_offset = num_pages - num_nodes
    for offset in range(max_offset):
        subset = pages[offset : num_nodes + offset - 1]
        graphs.append(generate_graph(subset))
    return graphs


def permute_graph(graph: nx.Graph) -> tuple[nx.Graph, list[int]]:
    """Shuffles nodes into random order"""

    num_nodes = len(graph)
    new_order = sample(range(num_nodes), num_nodes)
    mapping = dict(zip(graph.nodes, new_order))
    new_graph = nx.relabel_nodes(graph, mapping)
    for node in new_graph.nodes:
        new_graph.nodes[node][NODE_ATTR_NUM] = int(node)
    return new_graph, new_order


def get_weight_stats(graph: nx.Graph) -> np.ndarray:
    """Returns weight statistics"""
    return np.array([graph.edges[edge][EDGE_ATTR] for edge in graph.edges])


def plot_weight_stats(
    graph_whole: nx.Graph,
    graph_split: nx.Graph,
    title: str | None = None,
    bins: int = 100,
) -> None:
    """Plots a histogram of the weights"""
    stats_whole = get_weight_stats(graph_whole)
    stats_split = get_weight_stats(graph_split)

    plt.hist(stats_whole, bins=bins, alpha=0.5, label="whole")
    plt.hist(stats_split, bins=bins, alpha=0.5, label="split")
    plt.xlabel("cosine similarity")
    plt.ylabel("binned count")
    plt.legend()
    if title is not None:
        plt.title(title)
    else:
        plt.title("weight histogram")
    plt.show()


def naive_sequence(graph: nx.Graph, root: Any):
    """Returns reading sequence that starts at root and goes to highest similarity in each step"""
    visited = [root]
    for i in range(len(graph) - 1):
        weights = [
            (node, graph.edges[(visited[i], node)][EDGE_ATTR])
            for node in graph
            if node not in visited
        ]
        new = max(weights, key=lambda x: x[1])
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
