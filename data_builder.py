"""# Hybrig RAG MODEL FOR Flan-T5-base"""

import wikipediaapi
import re
import time

wiki = wikipediaapi.Wikipedia(
    language="en",
    extract_format=wikipediaapi.ExtractFormat.WIKI,
    user_agent="HybridRAG-POC/1.0 (academic demo)"
)

def fetch_valid_article(title, min_words=200):
    time.sleep(0.5)  # rate limiting

    page = wiki.page(title)
    if not page.exists():
        return None

    text = re.sub(r'\s+', ' ', page.text).strip()
    if len(text.split()) < min_words:
        return None

    return {
        "url": page.fullurl,
        "title": page.title,
        "text": text
    }

import json
import random

# SEED_TITLES = [
#     "Artificial intelligence", "World War II", "Quantum mechanics",
#     "Indian cuisine", "Climate change", "Machine learning",
#     "Blockchain", "Ancient Rome", "Solar energy", "Neural networks"
# ]

# fixed_articles = []
# visited = set()

# while len(fixed_articles) < 200:
#     seed = random.choice(SEED_TITLES)
#     page = wiki.page(seed)

#     for link in page.links.values():
#         if link.title in visited:
#             continue
#         visited.add(link.title)

#         article = fetch_valid_article(link.title)
#         if article:
#             fixed_articles.append(article)
#         if len(fixed_articles) == 200:
#             break

# # Save ONLY URLs (requirement)
# with open("fixed_urls.json", "w") as f:
#     json.dump([a["url"] for a in fixed_articles], f, indent=2)

import requests

HEADERS = {
    "User-Agent": "HybridRAG-POC/1.0 (academic demo)"
}

def get_random_wikipedia_title():
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "random",
        "rnnamespace": 0,
        "rnlimit": 1
    }

    try:
        response = requests.get(
            url,
            params=params,
            headers=HEADERS,
            timeout=10
        )
        response.raise_for_status()

        data = response.json()
        return data["query"]["random"][0]["title"]

    except requests.exceptions.RequestException as e:
        print(f"HTTP error fetching random title: {e}")
        return None
    except (KeyError, ValueError) as e:
        print(f"Parsing error fetching random title: {e}")
        return None

def get_random_articles(n=300, max_attempts=8000):
    articles = []
    visited = set()
    attempts = 0

    while len(articles) < n and attempts < max_attempts:
        attempts += 1
        time.sleep(0.5)  # keep this slow & safe

        title = get_random_wikipedia_title()
        if not title or title in visited:
            continue

        visited.add(title)

        article = fetch_valid_article(title)
        if article:
            articles.append(article)

        if attempts % 200 == 0:
            print(f"Attempts: {attempts}, Collected: {len(articles)}")

    if len(articles) < n:
        raise RuntimeError(
            f"Only collected {len(articles)} articles after {attempts} attempts"
        )

    return articles

# Load fixed URLs
with open("data/fixed_urls.json") as f:
    fixed_urls = json.load(f)

fixed_articles = []

for i, url in enumerate(fixed_urls, start=1):
    title = url.split("/wiki/")[-1]
    article = fetch_valid_article(title)
    if article:
        fixed_articles.append(article)

    if i % 20 == 0:
        print(f"Loaded {i}/{len(fixed_urls)} fixed articles")

print("Fixed articles fetched:", len(fixed_articles))

# Fetch random articles
random_articles = get_random_articles(300)

# Combine
corpus = fixed_articles + random_articles
print("Total corpus size:", len(corpus))

import tiktoken
import uuid

tokenizer = tiktoken.get_encoding("cl100k_base")

def chunk_text(text, min_tokens=200, max_tokens=400, overlap=50):
    tokens = tokenizer.encode(text)
    chunks = []

    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]

        if len(chunk_tokens) >= min_tokens:
            chunks.append(tokenizer.decode(chunk_tokens))

        start += max_tokens - overlap

    return chunks

processed_chunks = []

for article in corpus:
    chunks = chunk_text(article["text"])

    for chunk in chunks:
        processed_chunks.append({
            "chunk_id": str(uuid.uuid4()),
            "url": article["url"],
            "title": article["title"],
            "text": chunk
        })
print("Total chunks created:", len(processed_chunks))
print("1")
with open("wikipedia_corpus_chunks.json", "w") as f:
    json.dump(processed_chunks, f, indent=2)
