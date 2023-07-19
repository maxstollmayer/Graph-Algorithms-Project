"""utility functions for generating and analysing graphs of books"""

from functools import partial
import os
from random import sample

import dgl
from dgl.data import DGLDataset
import networkx as nx
from nltk import word_tokenize, pos_tag, download
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import numpy as np
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch

TRAINING_BOOKS = [
    "Alice's Adventures in Wonderland",
    "Animal Farm",
    "Cloud Atlas",
    "If on a winter's night a traveler",
    "Of Mice and Men",
    "The Big Sleep",
    "The Maltese Falcon",
    "The Metamorphosis",
    "The Murder of Roger Ackroyd",
    "The Nine Tailors",
    "The Sound and the Fury",
    "The Strange Case of Dr. Jekyll and Mr. Hyde",
]
SAVE_DIR = "books"  # directory to save dataset in
SAVE_NAME = "_dataset.bin"  # name of saved dataset
PATHS = [os.path.join(SAVE_DIR, book + ".txt") for book in TRAINING_BOOKS]
ATTR = "attr"  # name of attributes
NUM_PAGES = 100  # number of pages in Cain's Jawbone
NUM_PERMS = 10  # number of permutations to add for each generated graph


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
    - sum of all adjacent edge weights
    - edges weighted by the cosine similarity
    """

    weights = get_weights(pages)

    # generate fully connected, weighted graph
    graph = nx.complete_graph(len(pages))

    for i, j in graph.edges:
        graph.edges[i, j][ATTR] = weights[i, j]

    for node in graph.nodes:
        graph.nodes[node][ATTR] = (
            float(node),
            float(len(pages[node])),
            float(len(pages[node].split())),
            sum(
                graph.edges[(node, other)][ATTR]
                for other in graph.nodes
                if node != other
            ),
        )
        # TODO: more attributes? e.g. page ends with punctuation or not

    return graph


def to_dgl(graph: nx.Graph) -> dgl.DGLGraph:
    """Converts a NetworkX graph to a DGL graph"""
    return dgl.from_networkx(
        nx.to_directed(graph), node_attrs=[ATTR], edge_attrs=[ATTR]
    )


def load_dgl_graph(book: str, page_delimiter: str = "\n\n") -> dgl.DGLGraph:
    """Generates a DGL graph of given book"""
    path = os.path.join(SAVE_DIR, book + ".txt")
    with open(path, "r", encoding="utf-8") as file:
        pages = file.read().split(page_delimiter)

    return to_dgl(generate_graph(pages))


def get_windows(num_pages: int, window_size: int) -> list[tuple[int, int]]:
    """Returns equally spaced windows for given range"""
    if num_pages <= window_size:
        return [(0, num_pages)]

    if num_pages <= 2 * window_size:
        return [(0, window_size), (num_pages - window_size, num_pages)]

    q = num_pages / window_size
    n = int(q)
    num_windows = n if q == n else n + 1

    offset = (num_windows * window_size - num_pages) / (num_windows - 1)
    overlap = int(offset)
    missing = int(round((offset - overlap) * (num_windows - 1)))

    windows = []
    for i in range(num_windows):
        start = i * (window_size - overlap)
        end = (i + 1) * window_size - i * overlap
        if num_windows - i <= missing:
            start -= missing
            end -= missing
        windows.append((start, end))

    return windows


def permute_graph(graph: nx.Graph) -> tuple[nx.Graph, list[int]]:
    """Shuffles nodes into random order"""

    num_nodes = len(graph)
    new_order = sample(range(num_nodes), num_nodes)
    mapping = dict(zip(graph.nodes, new_order))
    new_graph = nx.relabel_nodes(graph, mapping)
    for node in new_graph.nodes:
        _, attrs = new_graph.nodes[node][ATTR]
        new_graph.nodes[node][ATTR] = (float(node), *attrs)
    return new_graph, new_order


def sparsify_graph(graph: nx.Graph, k: int):
    """Returns graph that only consists of the k nearest neighbor edges"""
    new_graph = graph.copy()
    new_graph.clear_edges()

    def get_weight(node, other):
        return graph.edges[(node, other)][ATTR]

    for node in graph.nodes:
        k_neighbors = sorted(
            graph.neighbors(node), key=partial(get_weight, node), reverse=True
        )[:k]
        for other in k_neighbors:
            new_graph.add_edge(node, other)
            new_graph.edges[(node, other)][ATTR] = graph.edges[(node, other)][ATTR]
    return new_graph


class BookDataset(DGLDataset):
    """Dataset for graph of books"""

    def __init__(
        self,
        force_reload: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            name=SAVE_DIR, save_dir="", force_reload=force_reload, verbose=verbose
        )

    def process(self) -> None:
        # load books

        texts = []
        for path in PATHS:
            with open(path, "r", encoding="utf-8") as book:
                texts.append(book.read())

        nxgraphs = []
        for text in texts:
            # resplit into 100 pages
            split_pages = split_into_pages(text, NUM_PAGES)
            nxgraphs.append(generate_graph(split_pages))

            # subgraphs of 100 consecutive pages
            true_pages = text.split("\n\n")
            if len(true_pages) > NUM_PAGES:
                windows = get_windows(len(true_pages), NUM_PAGES)
                subgraphs = [
                    generate_graph(true_pages[start:end]) for (start, end) in windows
                ]
                nxgraphs.extend(subgraphs)

        # permute graphs, add labels and convert to dgl
        graphs = []
        labels = []
        for graph in nxgraphs:
            for _ in range(NUM_PERMS):
                permuted_graph, label = permute_graph(graph)
                graphs.append(to_dgl(permuted_graph))
                labels.append(label)

        self.graphs = graphs
        self.labels = torch.LongTensor(labels)

    def __getitem__(self, i: int) -> tuple[dgl.DGLGraph, torch.Tensor]:
        return self.graphs[i], self.labels[i]

    def __len__(self) -> int:
        return len(self.graphs)

    def save(self) -> None:
        graph_path = os.path.join(self.save_path, SAVE_NAME)
        dgl.save_graphs(graph_path, self.graphs, {"labels": self.labels})

    def load(self) -> None:
        graph_path = os.path.join(self.save_path, SAVE_NAME)
        self.graphs, label_dict = dgl.load_graphs(graph_path)
        self.labels = label_dict["labels"]

    def has_cache(self):
        graph_path = os.path.join(self.save_path, SAVE_NAME)
        return os.path.exists(graph_path)
