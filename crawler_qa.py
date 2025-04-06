
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from sentence_transformers import SentenceTransformer, util
import torch

def crawl_help_site(base_url, max_depth=2):
    visited = set()
    to_visit = [(base_url, 0)]
    pages = {}

    while to_visit:
        url, depth = to_visit.pop()
        if depth > max_depth or url in visited:
            continue
        visited.add(url)

        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                continue
            soup = BeautifulSoup(response.text, 'html.parser')
            pages[url] = soup

            for link in soup.find_all('a', href=True):
                href = urljoin(url, link['href'])
                if href.startswith(base_url):
                    to_visit.append((href, depth + 1))

        except Exception as e:
            print(f"[!] Failed to crawl {url}: {e}")

    return pages

def extract_clean_content(pages):
    content_data = []
    for url, soup in pages.items():
        for tag in soup(['nav', 'header', 'footer', 'script', 'style']):
            tag.decompose()
        text = soup.get_text(separator=' ', strip=True)
        content_data.append({"url": url, "content": text})
    return content_data

model = SentenceTransformer('all-MiniLM-L6-v2')

def build_index(documents):
    texts = [doc['content'] for doc in documents]
    embeddings = model.encode(texts, convert_to_tensor=True)
    metadata = [doc['url'] for doc in documents]
    return embeddings, texts, metadata

def question_answer_loop(index, texts, metadata):
    print("🔍 Q&A Agent Ready! Ask your question below (type 'exit' to stop):")
    while True:
        query = input("\n> ")
        if query.lower() in ['exit', 'quit']:
            break
        query_embedding = model.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, index)[0]
        top_result = torch.topk(cos_scores, k=1)
        score = top_result.values.item()
        idx = top_result.indices.item()

        if score < 0.3:
            print("❌ Sorry, I couldn't find anything relevant in the documentation.")
        else:
            print(f"\n✅ Answer (confidence: {score:.2f}):\n")
            print(texts[idx][:1000])  # Display snippet
            print(f"\n🔗 Source: {metadata[idx]}")

if __name__ == "__main__":
    url = "https://help.zluri.com"  # Replace with your help site
    print("🌐 Crawling the website...")
    pages = crawl_help_site(url)
    print(f"✅ Crawled {len(pages)} pages. Cleaning content...")
    documents = extract_clean_content(pages)
    print("🔎 Building semantic index...")
    index, texts, metadata = build_index(documents)
    question_answer_loop(index, texts, metadata)
