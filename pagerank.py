import os
import random
import re
import sys
import math
import numpy as np

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a set of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}
    
    for filename in pages:
        pages[filename] = set(link for link in pages[filename] if link in pages)
    
    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.
    """
    n = len(corpus)
    probabilities = {p: (1 - damping_factor) / n for p in corpus}
    links = corpus[page]
    
    if links:
        for link in links:
            probabilities[link] += damping_factor / len(links)
    else:
        for p in probabilities:
            probabilities[p] = 1 / n  # Distribute equally if no links exist
    
    return probabilities


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.
    """
    pagerank = {page: 0 for page in corpus}
    sample = random.choice(list(corpus.keys()))  # Select initial page randomly
    
    for _ in range(n):
        pagerank[sample] += 1
        probabilities = transition_model(corpus, sample, damping_factor)
        pages, weights = zip(*probabilities.items())
        sample = random.choices(pages, weights=weights, k=1)[0]
    
    total_samples = sum(pagerank.values())
    return {page: count / total_samples for page, count in pagerank.items()}


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.
    """
    n = len(corpus)
    pagerank = {page: 1 / n for page in corpus}  # Initial equal probability
    threshold = 0.001
    
    while True:
        new_pagerank = {}
        
        for page in corpus:
            rank_sum = sum(
                pagerank[linking_page] / len(corpus[linking_page])
                for linking_page in corpus if page in corpus[linking_page]
            )
            rank_sum += sum(
                pagerank[no_link_page] / n for no_link_page in corpus if len(corpus[no_link_page]) == 0
            )
            new_pagerank[page] = (1 - damping_factor) / n + damping_factor * rank_sum
        
        # Check for convergence
        if all(math.isclose(new_pagerank[p], pagerank[p], abs_tol=threshold) for p in pagerank):
            break
        
        pagerank = new_pagerank
    
    return pagerank


if __name__ == "__main__":
    main()
