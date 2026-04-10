import os
import re
import io
import json
import time
import math
import queue
import hashlib
import threading
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.robotparser import RobotFileParser

import requests
import pandas as pd
import streamlit as st
import trafilatura
from bs4 import BeautifulSoup

try:
    import chromadb
    from chromadb.config import Settings
except Exception:
    chromadb = None
    Settings = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import pypdf
except Exception:
    pypdf = None

try:
    import pydeck as pdk
except Exception:
    pdk = None


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Wildfire Knowledge Lab Pro",
    page_icon="🔥",
    layout="wide",
)

st.markdown("""
<style>
.block-container {
    max-width: 100% !important;
    padding-left: 2.2rem;
    padding-right: 2.2rem;
    padding-top: 1.0rem;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.hero-card {
    background: linear-gradient(135deg, rgba(239,68,68,0.14), rgba(251,113,133,0.09));
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 24px;
    padding: 1.2rem 1.3rem 1.15rem 1.3rem;
    box-shadow: 0 10px 34px rgba(0,0,0,0.16);
    margin-bottom: 1rem;
}
.metric-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 0.9rem 1rem;
}
.source-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 0.95rem 1rem;
    margin-bottom: 0.75rem;
}
.incident-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 0.95rem 1rem;
    margin-bottom: 0.75rem;
}
.big-button button {
    background: linear-gradient(90deg, #ef4444, #fb7185) !important;
    color: white !important;
    border-radius: 14px !important;
    padding: 0.80rem 1.25rem !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    font-weight: 900 !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.20) !important;
}
.small-muted {
    color: rgba(255,255,255,0.68);
    font-size: 0.92rem;
    line-height: 1.35;
}
textarea, input, [data-baseweb="textarea"] textarea, [data-baseweb="input"] input {
    color: #111827 !important;
    background: #ffffff !important;
    -webkit-text-fill-color: #111827 !important;
    caret-color: #111827 !important;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# CONFIG
# =========================================================
APP_TITLE = "Wildfire Knowledge Lab Pro"
APP_SUBTITLE = "Incident feed + outlook ingestion + fire weather parsing + map + daily briefing + grounded wildfire QA"

CHROMA_DIR = os.getenv("WILDFIRE_CHROMA_DIR", "./wildfire_chroma")
COLLECTION_NAME = os.getenv("WILDFIRE_COLLECTION_NAME", "wildfire_knowledge_pro")
EMBED_MODEL_NAME = os.getenv("WILDFIRE_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = st.secrets.get("openai", {}).get("api_key") or os.getenv("OPENAI_API_KEY", "")

USER_AGENT = "WildfireKnowledgeLabPro/2.0 (+research)"
REQUEST_TIMEOUT = 30
TOP_K = 8
CHUNK_SIZE_WORDS = 220
CHUNK_OVERLAP_WORDS = 40

INCIWEB_RSS_URL = "https://inciweb.wildfire.gov/feeds/rss/incidents/"
NIFC_MONTHLY_OUTLOOK_PDF = "https://www.nifc.gov/nicc-files/predictive/outlooks/monthly_seasonal_outlook.pdf"
NIFC_NA_OUTLOOK_PDF = "https://www.nifc.gov/nicc-files/predictive/outlooks/NA_Outlook.pdf"
NIFC_WEATHER_PAGE = "https://www.nifc.gov/nicc/predictive-services/weather"

DEFAULT_SEEDS = [
    "https://www.nifc.gov/",
    "https://www.nwcg.gov/",
    "https://www.fs.usda.gov/managing-land/fire",
    "https://www.fire.ca.gov/",
    "https://www.readyforwildfire.org/",
    "https://www.weather.gov/",
]

DEFAULT_ALLOWED_DOMAINS = [
    "nifc.gov",
    "nwcg.gov",
    "fs.usda.gov",
    "fire.ca.gov",
    "readyforwildfire.org",
    "weather.gov",
]

DEFAULT_FIRE_WEATHER_URLS = [
    "https://www.weather.gov/gjt/fire",
    "https://www.weather.gov/unr/brief_fire",
]

if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""
if "last_hits" not in st.session_state:
    st.session_state.last_hits = []
if "last_briefing" not in st.session_state:
    st.session_state.last_briefing = ""
if "last_incidents" not in st.session_state:
    st.session_state.last_incidents = []
if "last_map_df" not in st.session_state:
    st.session_state.last_map_df = pd.DataFrame()
if "ingest_log" not in st.session_state:
    st.session_state.ingest_log = []
if "latest_stats" not in st.session_state:
    st.session_state.latest_stats = {"pages": 0, "chunks": 0, "sources": 0}
if "latest_docs_preview" not in st.session_state:
    st.session_state.latest_docs_preview = []


# =========================================================
# HELPERS
# =========================================================
def utc_now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_url(url: str) -> str:
    url = urldefrag(url)[0].strip()
    parsed = urlparse(url)
    return parsed._replace(
        scheme=parsed.scheme.lower(),
        netloc=parsed.netloc.lower(),
        path=re.sub(r"/{2,}", "/", parsed.path),
        fragment=""
    ).geturl()


def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def split_into_word_chunks(text: str, chunk_size_words: int = CHUNK_SIZE_WORDS, overlap_words: int = CHUNK_OVERLAP_WORDS) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(1, chunk_size_words - overlap_words)
    for start in range(0, len(words), step):
        piece = words[start:start + chunk_size_words]
        if piece:
            chunks.append(" ".join(piece))
        if start + chunk_size_words >= len(words):
            break
    return chunks


def html_title(html: str, fallback: str = "") -> str:
    soup = BeautifulSoup(html, "html.parser")
    if soup.title and soup.title.string:
        t = clean_text(soup.title.string)
        if t:
            return t
    h1 = soup.find("h1")
    if h1:
        t = clean_text(h1.get_text(" ", strip=True))
        if t:
            return t
    return fallback


def looks_like_html_url(url: str) -> bool:
    blocked = (
        ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".zip", ".rar", ".7z",
        ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".tif", ".tiff",
        ".csv", ".xml", ".rss", ".atom", ".geojson", ".shp"
    )
    return not url.lower().endswith(blocked)


def is_allowed_domain(url: str, allowed_domains: List[str]) -> bool:
    netloc = urlparse(url).netloc.lower()
    if not netloc:
        return False
    for dom in allowed_domains:
        dom = dom.lower().strip()
        if netloc == dom or netloc.endswith("." + dom):
            return True
    return False


def extract_links(base_url: str, html: str) -> List[str]:
    out = []
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a.get("href", "").strip()
        if not href:
            continue
        full = normalize_url(urljoin(base_url, href))
        if full.startswith("http://") or full.startswith("https://"):
            out.append(full)
    return list(dict.fromkeys(out))


def fetch_text_url(url: str, timeout: int = REQUEST_TIMEOUT) -> Tuple[str, str]:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    r.raise_for_status()
    ctype = r.headers.get("Content-Type", "").lower()
    return r.text if "text" in ctype or "html" in ctype or "xml" in ctype else r.content.decode("utf-8", errors="ignore"), ctype


def fetch_bytes_url(url: str, timeout: int = REQUEST_TIMEOUT) -> bytes:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    r.raise_for_status()
    return r.content


def read_pdf_bytes(file_bytes: bytes) -> str:
    if pypdf is None:
        return ""
    try:
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            if txt.strip():
                pages.append(txt)
        return clean_text("\n\n".join(pages))
    except Exception:
        return ""


def parse_datetime_loose(s: str) -> Optional[datetime]:
    if not s:
        return None
    s = s.strip()
    candidates = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in candidates:
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            pass
    return None


def estimate_cost_text(tokens_in: int, tokens_out: int, model_name: str) -> str:
    price = {
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4o": (2.50, 10.00),
    }
    if model_name not in price:
        return "est cost: n/a"
    pin, pout = price[model_name]
    usd = (tokens_in / 1_000_000.0) * pin + (tokens_out / 1_000_000.0) * pout
    return f"est cost: ${usd:.6f}"


def extract_lat_lon_from_text(text: str) -> Tuple[Optional[float], Optional[float]]:
    if not text:
        return None, None
    patterns = [
        r"(-?\d{1,3}\.\d+)\s*,\s*(-?\d{1,3}\.\d+)",
        r"lat(?:itude)?[:= ]+(-?\d{1,3}\.\d+).*?lon(?:gitude)?[:= ]+(-?\d{1,3}\.\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.I | re.S)
        if m:
            a = float(m.group(1))
            b = float(m.group(2))
            if -90 <= a <= 90 and -180 <= b <= 180:
                return a, b
            if -180 <= a <= 180 and -90 <= b <= 90:
                return b, a
    return None, None


def maybe_trim(text: str, n: int = 1200) -> str:
    text = text or ""
    return text[:n] + ("..." if len(text) > n else "")


# =========================================================
# DATA STRUCTURES
# =========================================================
@dataclass
class PageRecord:
    url: str
    title: str
    text: str
    source_domain: str
    fetched_at_utc: str
    content_hash: str
    source_type: str


@dataclass
class ChunkRecord:
    chunk_id: str
    url: str
    title: str
    source_domain: str
    chunk_index: int
    text: str
    content_hash: str
    fetched_at_utc: str
    source_type: str


# =========================================================
# ROBOTS
# =========================================================
class RobotsManager:
    def __init__(self, user_agent: str):
        self.user_agent = user_agent
        self.parsers: Dict[str, RobotFileParser] = {}
        self.lock = threading.Lock()

    def allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        root = f"{parsed.scheme}://{parsed.netloc}"
        with self.lock:
            if root not in self.parsers:
                rp = RobotFileParser()
                rp.set_url(urljoin(root, "/robots.txt"))
                try:
                    rp.read()
                except Exception:
                    pass
                self.parsers[root] = rp
            rp = self.parsers[root]
        try:
            return rp.can_fetch(self.user_agent, url)
        except Exception:
            return True


# =========================================================
# CRAWLER
# =========================================================
class WildfireCrawler:
    def __init__(
        self,
        seed_urls: List[str],
        allowed_domains: List[str],
        max_pages: int = 120,
        max_depth: int = 2,
        request_delay_sec: float = 0.5,
        max_workers: int = 6,
        user_agent: str = USER_AGENT,
    ):
        self.seed_urls = [normalize_url(u) for u in seed_urls if u.strip()]
        self.allowed_domains = [d.strip() for d in allowed_domains if d.strip()]
        self.max_pages = int(max_pages)
        self.max_depth = int(max_depth)
        self.request_delay_sec = float(request_delay_sec)
        self.max_workers = int(max_workers)
        self.user_agent = user_agent
        self.headers = {"User-Agent": self.user_agent}
        self.robots = RobotsManager(self.user_agent)
        self.q = queue.Queue()
        self.lock = threading.Lock()
        self.visited = set()
        self.page_records: List[PageRecord] = []
        self.last_request_time = 0.0

    def _rate_limit(self):
        with self.lock:
            now = time.time()
            wait = self.request_delay_sec - (now - self.last_request_time)
            if wait > 0:
                time.sleep(wait)
            self.last_request_time = time.time()

    def _fetch_html(self, url: str) -> Optional[requests.Response]:
        try:
            self._rate_limit()
            r = requests.get(url, headers=self.headers, timeout=REQUEST_TIMEOUT, allow_redirects=True)
            ctype = r.headers.get("Content-Type", "").lower()
            if r.status_code != 200:
                return None
            if "text/html" not in ctype:
                return None
            return r
        except Exception:
            return None

    def _extract_main_text(self, html: str) -> str:
        txt = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
        )
        if txt:
            return clean_text(txt)
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.decompose()
        return clean_text(soup.get_text("\n", strip=True))

    def _worker(self):
        while True:
            try:
                url, depth = self.q.get(timeout=1)
            except queue.Empty:
                return

            try:
                with self.lock:
                    if url in self.visited or len(self.visited) >= self.max_pages:
                        continue
                    self.visited.add(url)

                if not is_allowed_domain(url, self.allowed_domains):
                    continue
                if not looks_like_html_url(url):
                    continue
                if not self.robots.allowed(url):
                    continue

                resp = self._fetch_html(url)
                if resp is None:
                    continue

                html = resp.text
                text = self._extract_main_text(html)
                title = html_title(html, fallback=url)

                if len(text.split()) >= 80:
                    self.page_records.append(
                        PageRecord(
                            url=url,
                            title=title,
                            text=text,
                            source_domain=urlparse(url).netloc.lower(),
                            fetched_at_utc=utc_now_iso(),
                            content_hash=safe_hash(text),
                            source_type="web",
                        )
                    )

                if depth < self.max_depth:
                    for link in extract_links(url, html):
                        if is_allowed_domain(link, self.allowed_domains):
                            with self.lock:
                                if link not in self.visited and len(self.visited) < self.max_pages:
                                    self.q.put((link, depth + 1))
            finally:
                self.q.task_done()

    def crawl(self) -> List[PageRecord]:
        for s in self.seed_urls:
            self.q.put((s, 0))
        threads = []
        for _ in range(self.max_workers):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            threads.append(t)
        self.q.join()
        for t in threads:
            t.join(timeout=1)
        return self.page_records


# =========================================================
# VECTOR STORE
# =========================================================
class WildfireKnowledgeBase:
    def __init__(self, chroma_dir: str, collection_name: str, embed_model_name: str):
        if chromadb is None or SentenceTransformer is None:
            raise RuntimeError("Install chromadb and sentence-transformers.")
        self.client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedder = SentenceTransformer(embed_model_name)

    def page_to_chunks(self, page: PageRecord) -> List[ChunkRecord]:
        chunks = split_into_word_chunks(page.text)
        out = []
        for i, c in enumerate(chunks):
            cid = safe_hash(f"{page.url}|{page.content_hash}|{i}|{c[:120]}")
            out.append(
                ChunkRecord(
                    chunk_id=cid,
                    url=page.url,
                    title=page.title,
                    source_domain=page.source_domain,
                    chunk_index=i,
                    text=c,
                    content_hash=page.content_hash,
                    fetched_at_utc=page.fetched_at_utc,
                    source_type=page.source_type,
                )
            )
        return out

    def upsert_pages(self, pages: List[PageRecord]) -> int:
        chunks = []
        for p in pages:
            chunks.extend(self.page_to_chunks(p))
        if not chunks:
            return 0

        ids = [c.chunk_id for c in chunks]
        docs = [c.text for c in chunks]
        metas = [{
            "url": c.url,
            "title": c.title,
            "source_domain": c.source_domain,
            "chunk_index": c.chunk_index,
            "content_hash": c.content_hash,
            "fetched_at_utc": c.fetched_at_utc,
            "source_type": c.source_type,
        } for c in chunks]

        embeddings = self.embedder.encode(docs, show_progress_bar=False, normalize_embeddings=True).tolist()

        self.collection.upsert(
            ids=ids,
            documents=docs,
            metadatas=metas,
            embeddings=embeddings,
        )
        return len(chunks)

    def query(self, question: str, top_k: int = TOP_K) -> List[Dict]:
        qemb = self.embedder.encode([question], normalize_embeddings=True).tolist()[0]
        res = self.collection.query(
            query_embeddings=[qemb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        distances = res.get("distances", [[]])[0]
        hits = []
        for d, m, dist in zip(docs, metas, distances):
            hits.append({
                "text": d,
                "metadata": m,
                "distance": float(dist) if dist is not None else None,
            })
        return hits

    def peek(self, n: int = 40) -> List[Dict]:
        try:
            data = self.collection.get(limit=n, include=["documents", "metadatas"])
            out = []
            for d, m in zip(data.get("documents", []), data.get("metadatas", [])):
                out.append({"text": d, "metadata": m})
            return out
        except Exception:
            return []

    def count(self) -> int:
        try:
            return self.collection.count()
        except Exception:
            return 0

    def clear(self):
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)


@st.cache_resource(show_spinner=False)
def get_kb():
    return WildfireKnowledgeBase(
        chroma_dir=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embed_model_name=EMBED_MODEL_NAME,
    )


# =========================================================
# SOURCE INGESTORS
# =========================================================
def ingest_manual_text(title: str, text: str, source_type: str = "note") -> List[PageRecord]:
    text = clean_text(text)
    if len(text.split()) < 20:
        return []
    return [
        PageRecord(
            url=f"{source_type}://{safe_hash(title + text)[:12]}",
            title=title.strip() or "Untitled",
            text=text,
            source_domain=source_type,
            fetched_at_utc=utc_now_iso(),
            content_hash=safe_hash(text),
            source_type=source_type,
        )
    ]


def ingest_uploaded_files(files) -> List[PageRecord]:
    pages = []
    for f in files or []:
        file_bytes = f.read()
        fname = f.name
        suffix = os.path.splitext(fname)[1].lower()

        text = ""
        if suffix in [".txt", ".md", ".csv", ".json"]:
            try:
                text = clean_text(file_bytes.decode("utf-8", errors="ignore"))
            except Exception:
                text = ""
        elif suffix == ".pdf":
            text = read_pdf_bytes(file_bytes)
        else:
            try:
                text = clean_text(file_bytes.decode("utf-8", errors="ignore"))
            except Exception:
                text = ""

        if len(text.split()) < 30:
            continue

        pages.append(
            PageRecord(
                url=f"upload://{fname}",
                title=fname,
                text=text,
                source_domain="uploaded-file",
                fetched_at_utc=utc_now_iso(),
                content_hash=safe_hash(text),
                source_type="upload",
            )
        )
    return pages


def fetch_inciweb_incidents(rss_url: str = INCIWEB_RSS_URL, max_items: int = 30) -> List[Dict]:
    xml_text, _ = fetch_text_url(rss_url)
    root = ET.fromstring(xml_text)

    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "content": "http://purl.org/rss/1.0/modules/content/",
        "dc": "http://purl.org/dc/elements/1.1/",
        "georss": "http://www.georss.org/georss",
    }

    items = []
    for item in root.findall(".//item")[:max_items]:
        title = item.findtext("title", default="Untitled").strip()
        link = item.findtext("link", default="").strip()
        pub_date = item.findtext("pubDate", default="").strip()
        description_html = item.findtext("description", default="").strip()
        description_text = clean_text(BeautifulSoup(description_html, "html.parser").get_text("\n", strip=True))

        lat, lon = None, None
        pt = item.findtext("{http://www.georss.org/georss}point")
        if pt and " " in pt:
            try:
                lat_s, lon_s = pt.split()[:2]
                lat, lon = float(lat_s), float(lon_s)
            except Exception:
                lat, lon = None, None

        if lat is None or lon is None:
            lat, lon = extract_lat_lon_from_text(description_text)

        items.append({
            "title": title,
            "link": link,
            "pub_date": pub_date,
            "description": description_text,
            "lat": lat,
            "lon": lon,
        })
    return items


def incidents_to_pages(incidents: List[Dict]) -> List[PageRecord]:
    pages = []
    for x in incidents:
        text = clean_text(
            f"Incident Title: {x.get('title', '')}\n"
            f"Published: {x.get('pub_date', '')}\n"
            f"Link: {x.get('link', '')}\n"
            f"Latitude: {x.get('lat', '')}\n"
            f"Longitude: {x.get('lon', '')}\n"
            f"Summary:\n{x.get('description', '')}"
        )
        if len(text.split()) < 20:
            continue
        pages.append(
            PageRecord(
                url=x.get("link", "") or f"inciweb://{safe_hash(text)[:12]}",
                title=x.get("title", "InciWeb Incident"),
                text=text,
                source_domain="inciweb.wildfire.gov",
                fetched_at_utc=utc_now_iso(),
                content_hash=safe_hash(text),
                source_type="incident",
            )
        )
    return pages


def fetch_nifc_outlook_pages() -> List[PageRecord]:
    pages = []
    pdf_targets = [
        ("NIFC National Significant Wildland Fire Potential Outlook", NIFC_MONTHLY_OUTLOOK_PDF),
        ("NIFC North American Seasonal Fire Assessment and Outlook", NIFC_NA_OUTLOOK_PDF),
    ]
    for title, url in pdf_targets:
        try:
            pdf_bytes = fetch_bytes_url(url)
            text = read_pdf_bytes(pdf_bytes)
            if len(text.split()) >= 40:
                pages.append(
                    PageRecord(
                        url=url,
                        title=title,
                        text=text,
                        source_domain="nifc.gov",
                        fetched_at_utc=utc_now_iso(),
                        content_hash=safe_hash(text),
                        source_type="outlook",
                    )
                )
        except Exception:
            pass

    try:
        html, _ = fetch_text_url(NIFC_WEATHER_PAGE)
        txt = trafilatura.extract(html, include_comments=False, include_tables=False, no_fallback=False) or ""
        txt = clean_text(txt)
        if len(txt.split()) >= 40:
            pages.append(
                PageRecord(
                    url=NIFC_WEATHER_PAGE,
                    title="NIFC Predictive Services Weather",
                    text=txt,
                    source_domain="nifc.gov",
                    fetched_at_utc=utc_now_iso(),
                    content_hash=safe_hash(txt),
                    source_type="predictive-services",
                )
            )
    except Exception:
        pass

    return pages


def scrape_fire_weather_page(url: str) -> Optional[PageRecord]:
    try:
        html, _ = fetch_text_url(url)
        title = html_title(html, fallback=url)
        soup = BeautifulSoup(html, "html.parser")

        candidates = []

        for tag in soup.find_all(["h1", "h2", "h3", "p", "li", "pre", "div"]):
            txt = clean_text(tag.get_text(" ", strip=True))
            if len(txt.split()) >= 6:
                low = txt.lower()
                if any(k in low for k in [
                    "fire weather", "forecast discussion", "red flag", "humidity",
                    "gust", "wind", "dry", "critical fire weather", "spot forecast",
                    "briefing", "fire danger"
                ]):
                    candidates.append(txt)

        if not candidates:
            extracted = trafilatura.extract(html, include_comments=False, include_tables=False, no_fallback=False) or ""
            extracted = clean_text(extracted)
            candidates = [extracted] if len(extracted.split()) >= 40 else []

        text = clean_text("\n\n".join(dict.fromkeys(candidates)))
        if len(text.split()) < 40:
            return None

        return PageRecord(
            url=url,
            title=title,
            text=text,
            source_domain=urlparse(url).netloc.lower(),
            fetched_at_utc=utc_now_iso(),
            content_hash=safe_hash(text),
            source_type="fire-weather",
        )
    except Exception:
        return None


def fetch_fire_weather_pages(urls: List[str]) -> List[PageRecord]:
    pages = []
    for u in urls:
        rec = scrape_fire_weather_page(u)
        if rec:
            pages.append(rec)
    return pages


def fetch_nws_alerts_for_bbox_or_area(area: str = "") -> List[Dict]:
    base = "https://api.weather.gov/alerts/active"
    params = {}
    if area.strip():
        params["area"] = area.strip().upper()
    try:
        r = requests.get(base, headers={"User-Agent": USER_AGENT, "Accept": "application/geo+json"}, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        js = r.json()
        feats = js.get("features", [])
        out = []
        for f in feats:
            prop = f.get("properties", {})
            geom = f.get("geometry", {})
            event = prop.get("event", "")
            if "fire" in event.lower():
                out.append({
                    "event": event,
                    "headline": prop.get("headline", ""),
                    "severity": prop.get("severity", ""),
                    "areaDesc": prop.get("areaDesc", ""),
                    "sent": prop.get("sent", ""),
                    "description": prop.get("description", ""),
                    "instruction": prop.get("instruction", ""),
                    "response": prop.get("response", ""),
                    "url": prop.get("@id", ""),
                    "geometry": geom,
                })
        return out
    except Exception:
        return []


def alerts_to_pages(alerts: List[Dict]) -> List[PageRecord]:
    pages = []
    for a in alerts:
        text = clean_text(
            f"Event: {a.get('event', '')}\n"
            f"Headline: {a.get('headline', '')}\n"
            f"Severity: {a.get('severity', '')}\n"
            f"Area: {a.get('areaDesc', '')}\n"
            f"Sent: {a.get('sent', '')}\n"
            f"Description:\n{a.get('description', '')}\n\n"
            f"Instruction:\n{a.get('instruction', '')}\n"
        )
        if len(text.split()) < 15:
            continue
        pages.append(
            PageRecord(
                url=a.get("url", "") or f"nws-alert://{safe_hash(text)[:12]}",
                title=a.get("headline", "") or a.get("event", "NWS Fire Alert"),
                text=text,
                source_domain="api.weather.gov",
                fetched_at_utc=utc_now_iso(),
                content_hash=safe_hash(text),
                source_type="alert",
            )
        )
    return pages


# =========================================================
# LLM
# =========================================================
def get_openai_client():
    if OpenAI is None:
        raise RuntimeError("openai package not installed.")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set.")
    return OpenAI(api_key=OPENAI_API_KEY)


def build_grounded_prompt(question: str, hits: List[Dict]) -> Tuple[str, str]:
    blocks = []
    for i, h in enumerate(hits, start=1):
        md = h["metadata"]
        blocks.append(
            f"[Source {i}]\n"
            f"Title: {md.get('title', '')}\n"
            f"Type: {md.get('source_type', '')}\n"
            f"URL: {md.get('url', '')}\n"
            f"Domain: {md.get('source_domain', '')}\n"
            f"Excerpt:\n{h.get('text', '')}\n"
        )
    system_prompt = (
        "You are a wildfire knowledge analyst. Use only the provided sources. "
        "Be direct, practical, and grounded. If the sources are incomplete, say so."
    )
    user_prompt = f"Question:\n{question}\n\nRetrieved sources:\n\n" + "\n\n".join(blocks)
    return system_prompt, user_prompt


def answer_question_with_openai(question: str, hits: List[Dict]) -> Tuple[str, str]:
    client = get_openai_client()
    system_prompt, user_prompt = build_grounded_prompt(question, hits)

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    answer = resp.choices[0].message.content.strip()
    usage = getattr(resp, "usage", None)
    pt = getattr(usage, "prompt_tokens", 0) if usage else 0
    ct = getattr(usage, "completion_tokens", 0) if usage else 0
    return answer, f"model={OPENAI_MODEL} | prompt={pt} | completion={ct} | {estimate_cost_text(pt, ct, OPENAI_MODEL)}"


def answer_question_local(question: str, hits: List[Dict]) -> Tuple[str, str]:
    if not hits:
        return "No indexed wildfire sources are available yet. Ingest incidents, outlooks, weather pages, or web content first.", "local fallback"
    lines = [f"Question: {question}", "", "Top retrieved sources:", ""]
    for i, h in enumerate(hits, start=1):
        md = h["metadata"]
        lines.append(f"{i}. {md.get('title', '')} | {md.get('source_type', '')} | {md.get('url', '')}")
        lines.append(maybe_trim(h["text"], 900))
        lines.append("")
    lines.append("This is retrieval-only output because no OpenAI key was available.")
    return "\n".join(lines), "local fallback"


def build_daily_briefing_prompt(
    mode: str,
    incidents: List[Dict],
    outlook_pages: List[PageRecord],
    weather_pages: List[PageRecord],
    alert_pages: List[PageRecord],
    hits: List[Dict],
) -> Tuple[str, str]:
    incident_text = []
    for i, inc in enumerate(incidents[:15], start=1):
        incident_text.append(
            f"[Incident {i}] {inc.get('title', '')}\n"
            f"Published: {inc.get('pub_date', '')}\n"
            f"Link: {inc.get('link', '')}\n"
            f"Lat/Lon: {inc.get('lat', '')}, {inc.get('lon', '')}\n"
            f"Summary: {maybe_trim(inc.get('description', ''), 1000)}"
        )

    outlook_text = []
    for p in outlook_pages[:6]:
        outlook_text.append(f"[Outlook] {p.title}\nURL: {p.url}\n{maybe_trim(p.text, 1800)}")

    weather_text = []
    for p in weather_pages[:8]:
        weather_text.append(f"[Fire Weather] {p.title}\nURL: {p.url}\n{maybe_trim(p.text, 1400)}")

    alert_text = []
    for p in alert_pages[:8]:
        alert_text.append(f"[Alert] {p.title}\nURL: {p.url}\n{maybe_trim(p.text, 900)}")

    retrieval_text = []
    for i, h in enumerate(hits[:8], start=1):
        md = h["metadata"]
        retrieval_text.append(
            f"[Retrieved {i}] {md.get('title', '')}\n"
            f"Type: {md.get('source_type', '')}\n"
            f"URL: {md.get('url', '')}\n"
            f"{maybe_trim(h.get('text', ''), 1200)}"
        )

    mode_instructions = {
        "operations": (
            "Write a concise daily operations briefing. Focus on incident awareness, outlook signals, fire weather concerns, "
            "decision-relevant hazards, and actionable watch items. Use a command-brief style with clear sections."
        ),
        "research": (
            "Write a concise daily research briefing. Focus on patterns across incidents, recurring language in outlooks and weather, "
            "data signals, and hypotheses worth tracking. Use a research-synthesis style with structured observations."
        ),
    }

    system_prompt = (
        "You are a wildfire analyst building a daily intelligence briefing from provided incident, outlook, weather, alert, and retrieved knowledge inputs. "
        "Only use the provided material. Be concrete and structured."
    )

    user_prompt = (
        f"Mode: {mode}\n"
        f"Instruction: {mode_instructions.get(mode, mode_instructions['operations'])}\n\n"
        f"Incidents:\n\n" + "\n\n".join(incident_text) + "\n\n"
        f"Outlooks:\n\n" + "\n\n".join(outlook_text) + "\n\n"
        f"Fire weather pages:\n\n" + "\n\n".join(weather_text) + "\n\n"
        f"Active fire-related alerts:\n\n" + "\n\n".join(alert_text) + "\n\n"
        f"Retrieved knowledge context:\n\n" + "\n\n".join(retrieval_text) + "\n\n"
        "Output sections:\n"
        "1. Executive Summary\n"
        "2. Incident Picture\n"
        "3. Weather and Outlook Signals\n"
        "4. Operational or Research Watch Items\n"
        "5. Source Notes\n"
    )
    return system_prompt, user_prompt


def generate_daily_briefing(
    mode: str,
    incidents: List[Dict],
    outlook_pages: List[PageRecord],
    weather_pages: List[PageRecord],
    alert_pages: List[PageRecord],
    hits: List[Dict],
) -> Tuple[str, str]:
    if not OPENAI_API_KEY or OpenAI is None:
        text = (
            f"Daily briefing mode: {mode}\n\n"
            f"Incidents loaded: {len(incidents)}\n"
            f"Outlook pages loaded: {len(outlook_pages)}\n"
            f"Fire weather pages loaded: {len(weather_pages)}\n"
            f"Fire-related alerts loaded: {len(alert_pages)}\n"
            f"Retrieved context chunks: {len(hits)}\n\n"
            "OpenAI key was not available, so this is only a raw status summary."
        )
        return text, "local fallback"

    client = get_openai_client()
    system_prompt, user_prompt = build_daily_briefing_prompt(mode, incidents, outlook_pages, weather_pages, alert_pages, hits)

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    text = resp.choices[0].message.content.strip()
    usage = getattr(resp, "usage", None)
    pt = getattr(usage, "prompt_tokens", 0) if usage else 0
    ct = getattr(usage, "completion_tokens", 0) if usage else 0
    return text, f"model={OPENAI_MODEL} | prompt={pt} | completion={ct} | {estimate_cost_text(pt, ct, OPENAI_MODEL)}"


# =========================================================
# MAP HELPERS
# =========================================================
def incidents_to_df(incidents: List[Dict]) -> pd.DataFrame:
    rows = []
    for x in incidents:
        lat = x.get("lat")
        lon = x.get("lon")
        if lat is None or lon is None:
            continue
        rows.append({
            "title": x.get("title", ""),
            "link": x.get("link", ""),
            "pub_date": x.get("pub_date", ""),
            "description": maybe_trim(x.get("description", ""), 300),
            "lat": lat,
            "lon": lon,
            "kind": "incident",
        })
    return pd.DataFrame(rows)


def build_map_df(incidents: List[Dict]) -> pd.DataFrame:
    df = incidents_to_df(incidents)
    return df if not df.empty else pd.DataFrame(columns=["title", "link", "pub_date", "description", "lat", "lon", "kind"])


def render_pydeck_map(df: pd.DataFrame):
    if pdk is None:
        st.info("pydeck is not installed. Install it to enable the map.")
        return
    if df.empty:
        st.info("No incident points were found in the current feed.")
        return

    view_state = pdk.ViewState(
        latitude=float(df["lat"].mean()),
        longitude=float(df["lon"].mean()),
        zoom=4,
        pitch=30,
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position="[lon, lat]",
        get_radius=18000,
        get_fill_color="[239, 68, 68, 180]",
        get_line_color="[255,255,255,180]",
        line_width_min_pixels=1,
        pickable=True,
    )

    tooltip = {
        "html": "<b>{title}</b><br/>{pub_date}<br/>{description}",
        "style": {"backgroundColor": "rgba(30,30,30,0.95)", "color": "white"}
    }

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/dark-v11",
        tooltip=tooltip,
    )
    st.pydeck_chart(deck, use_container_width=True)


# =========================================================
# UI
# =========================================================
st.markdown(f"""
<div class="hero-card">
    <div style="font-size:2rem; font-weight:900; margin-bottom:0.25rem;">🔥 {APP_TITLE}</div>
    <div class="small-muted">{APP_SUBTITLE}</div>
</div>
""", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.markdown(f'<div class="metric-card"><div style="font-size:0.85rem; opacity:0.7;">Vector DB</div><div style="font-size:1.15rem; font-weight:800;">{COLLECTION_NAME}</div></div>', unsafe_allow_html=True)
m2.markdown(f'<div class="metric-card"><div style="font-size:0.85rem; opacity:0.7;">Embedding model</div><div style="font-size:1rem; font-weight:800;">{EMBED_MODEL_NAME}</div></div>', unsafe_allow_html=True)
m3.markdown(f'<div class="metric-card"><div style="font-size:0.85rem; opacity:0.7;">LLM</div><div style="font-size:1.15rem; font-weight:800;">{OPENAI_MODEL}</div></div>', unsafe_allow_html=True)
m4.markdown(f'<div class="metric-card"><div style="font-size:0.85rem; opacity:0.7;">OpenAI key</div><div style="font-size:1.15rem; font-weight:800;">{"set" if OPENAI_API_KEY else "not set"}</div></div>', unsafe_allow_html=True)

st.divider()

tabs = st.tabs(["Daily Briefing", "Incidents + Map", "Ask", "Ingest", "Library", "Settings"])


# =========================================================
# DAILY BRIEFING TAB
# =========================================================
with tabs[0]:
    kb_ok = True
    try:
        kb = get_kb()
    except Exception as e:
        kb_ok = False
        st.error(f"Knowledge base init failed: {e}")

    c1, c2 = st.columns([1.2, 1])

    with c1:
        briefing_mode = st.radio("Briefing mode", ["operations", "research"], horizontal=True)
        briefing_focus = st.text_area(
            "Optional focus prompt",
            value="Focus on incident awareness, outlook signals, weather concerns, and the most important watch items.",
            height=110,
        )
    with c2:
        alert_area = st.text_input("Optional NWS alert area code", value="CO")
        retrieve_query = st.text_input(
            "Retrieval query for added context",
            value="current wildfire operations outlook fire weather incident guidance",
        )

    st.markdown('<div class="big-button">', unsafe_allow_html=True)
    run_briefing = st.button("BUILD DAILY BRIEFING", use_container_width=True, type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

    if run_briefing:
        if not kb_ok:
            st.stop()

        with st.spinner("Pulling incident feed..."):
            incidents = fetch_inciweb_incidents()

        with st.spinner("Pulling outlooks and predictive services..."):
            outlook_pages = fetch_nifc_outlook_pages()

        with st.spinner("Pulling fire weather pages..."):
            weather_pages = fetch_fire_weather_pages(DEFAULT_FIRE_WEATHER_URLS)

        with st.spinner("Pulling fire-related alerts..."):
            alerts = fetch_nws_alerts_for_bbox_or_area(alert_area)
            alert_pages = alerts_to_pages(alerts)

        with st.spinner("Retrieving indexed context..."):
            hits = kb.query(f"{briefing_focus}\n\n{retrieve_query}", top_k=8) if kb_ok else []

        with st.spinner("Generating briefing..."):
            briefing, usage = generate_daily_briefing(
                mode=briefing_mode,
                incidents=incidents,
                outlook_pages=outlook_pages,
                weather_pages=weather_pages,
                alert_pages=alert_pages,
                hits=hits,
            )

        st.session_state.last_briefing = briefing
        st.session_state.last_incidents = incidents
        st.session_state.last_map_df = build_map_df(incidents)

        st.markdown("### Daily briefing")
        st.write(briefing)
        st.caption(usage)

        st.markdown("### Quick status")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Incidents", f"{len(incidents):,}")
        q2.metric("Outlook pages", f"{len(outlook_pages):,}")
        q3.metric("Weather pages", f"{len(weather_pages):,}")
        q4.metric("Fire alerts", f"{len(alert_pages):,}")

    elif st.session_state.last_briefing:
        st.markdown("### Last briefing")
        st.write(st.session_state.last_briefing)


# =========================================================
# INCIDENTS + MAP TAB
# =========================================================
with tabs[1]:
    left, right = st.columns([1.05, 1])

    with left:
        max_incidents = st.slider("Incident count", 5, 50, 20, 1)
        st.markdown('<div class="big-button">', unsafe_allow_html=True)
        run_incidents = st.button("REFRESH INCIDENT FEED", use_container_width=True, type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

        if run_incidents or not st.session_state.last_incidents:
            with st.spinner("Loading InciWeb incidents..."):
                try:
                    st.session_state.last_incidents = fetch_inciweb_incidents(max_items=max_incidents)
                    st.session_state.last_map_df = build_map_df(st.session_state.last_incidents)
                except Exception as e:
                    st.error(f"Failed to load incidents: {e}")

        incidents = st.session_state.last_incidents[:max_incidents]

        st.markdown("### Incident cards")
        for inc in incidents:
            lat = inc.get("lat")
            lon = inc.get("lon")
            coord_str = f"{lat}, {lon}" if lat is not None and lon is not None else "coordinates unavailable"

            st.markdown(
                f"""
                <div class="incident-card">
                    <div style="font-size:1.05rem; font-weight:800;">{inc.get("title", "Untitled")}</div>
                    <div style="font-size:0.86rem; opacity:0.72; margin-bottom:0.45rem;">
                        {inc.get("pub_date", "")} &nbsp;|&nbsp; {coord_str} &nbsp;|&nbsp;
                        <a href="{inc.get("link", "")}" target="_blank">open incident</a>
                    </div>
                    <div style="font-size:0.95rem; line-height:1.45;">{maybe_trim(inc.get("description", ""), 800)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        st.markdown("### Incident map")
        render_pydeck_map(st.session_state.last_map_df)

        if not st.session_state.last_map_df.empty:
            st.markdown("### Incident table")
            st.dataframe(
                st.session_state.last_map_df[["title", "pub_date", "lat", "lon", "link"]],
                use_container_width=True,
                hide_index=True,
            )


# =========================================================
# ASK TAB
# =========================================================
with tabs[2]:
    kb_ok = True
    try:
        kb = get_kb()
    except Exception as e:
        kb_ok = False
        st.error(f"Knowledge base init failed: {e}")

    a1, a2 = st.columns([1.4, 1])

    with a1:
        question = st.text_area(
            "Ask a wildfire question",
            value="What are the key themes across the indexed wildfire incident, outlook, weather, and policy sources?",
            height=150,
        )
        top_k = st.slider("Retrieved source count", 3, 12, 8, 1)
        st.markdown('<div class="big-button">', unsafe_allow_html=True)
        run_ask = st.button("ASK THE FIRE BRAIN", use_container_width=True, type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

    with a2:
        chunk_count = kb.count() if kb_ok else 0
        st.metric("Indexed chunks", f"{chunk_count:,}")
        st.metric("Recent pages", f"{st.session_state.latest_stats.get('pages', 0):,}")
        st.metric("Recent chunks", f"{st.session_state.latest_stats.get('chunks', 0):,}")
        st.metric("Recent sources", f"{st.session_state.latest_stats.get('sources', 0):,}")

    if run_ask:
        if not kb_ok:
            st.stop()

        with st.spinner("Searching indexed wildfire knowledge..."):
            hits = kb.query(question, top_k=top_k)
            st.session_state.last_hits = hits

        with st.spinner("Building grounded answer..."):
            try:
                if OPENAI_API_KEY and OpenAI is not None:
                    answer, usage = answer_question_with_openai(question, hits)
                else:
                    answer, usage = answer_question_local(question, hits)
            except Exception as e:
                answer, usage = answer_question_local(question, hits)
                answer += f"\n\nOpenAI call failed and the app fell back to retrieval-only output.\nError: {e}"

        st.session_state.last_answer = answer

        st.markdown("### Answer")
        st.write(answer)
        st.caption(usage)

        st.markdown("### Retrieved sources")
        for i, h in enumerate(hits, start=1):
            md = h["metadata"]
            st.markdown(
                f"""
                <div class="source-card">
                    <div style="font-size:1rem; font-weight:800;">[{i}] {md.get("title", "Untitled")}</div>
                    <div style="font-size:0.86rem; opacity:0.72; margin-bottom:0.45rem;">
                        {md.get("source_type", "")} &nbsp;|&nbsp; {md.get("source_domain", "")} &nbsp;|&nbsp;
                        <a href="{md.get("url", "")}" target="_blank">{md.get("url", "")}</a>
                    </div>
                    <div style="font-size:0.95rem; line-height:1.45;">{maybe_trim(h.get("text", ""), 900)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    elif st.session_state.last_answer:
        st.markdown("### Last answer")
        st.write(st.session_state.last_answer)


# =========================================================
# INGEST TAB
# =========================================================
with tabs[3]:
    kb_ok = True
    try:
        kb = get_kb()
    except Exception as e:
        kb_ok = False
        st.error(f"Knowledge base init failed: {e}")

    st.markdown("### Smart ingestion")
    i1, i2 = st.columns([1.1, 1])

    with i1:
        seed_text = st.text_area("Seed URLs", value="\n".join(DEFAULT_SEEDS), height=160)
        domains_text = st.text_area("Allowed domains", value="\n".join(DEFAULT_ALLOWED_DOMAINS), height=140)
        fire_weather_urls_text = st.text_area("Fire weather page URLs", value="\n".join(DEFAULT_FIRE_WEATHER_URLS), height=120)

    with i2:
        max_pages = st.number_input("Max pages", min_value=10, max_value=5000, value=120, step=10)
        max_depth = st.number_input("Max crawl depth", min_value=0, max_value=6, value=2, step=1)
        max_workers = st.number_input("Workers", min_value=1, max_value=24, value=6, step=1)
        request_delay = st.number_input("Delay between requests (sec)", min_value=0.0, max_value=5.0, value=0.5, step=0.1)
        alert_area = st.text_input("Alert area code for fire-related NWS alerts", value="CO")

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        run_crawl = st.button("CRAWL WEB", use_container_width=True)
    with b2:
        run_incident_ingest = st.button("INGEST INCIDENTS", use_container_width=True)
    with b3:
        run_outlook_ingest = st.button("INGEST OUTLOOKS", use_container_width=True)
    with b4:
        run_weather_ingest = st.button("INGEST WEATHER + ALERTS", use_container_width=True)

    st.divider()
    st.markdown("### Uploads and notes")
    u1, u2 = st.columns([1, 1])

    with u1:
        uploaded_files = st.file_uploader(
            "Upload PDFs, text, CSV, JSON",
            type=["pdf", "txt", "md", "csv", "json"],
            accept_multiple_files=True,
        )
        run_upload_ingest = st.button("INGEST FILES", use_container_width=True)

    with u2:
        manual_title = st.text_input("Note title", value="Wildfire note")
        manual_text = st.text_area("Paste note text", value="", height=180)
        run_note_ingest = st.button("INGEST NOTE", use_container_width=True)

    def finalize_ingest(pages: List[PageRecord], label: str):
        if not kb_ok:
            st.stop()
        added_chunks = kb.upsert_pages(pages) if pages else 0
        unique_sources = len({p.url for p in pages})
        st.session_state.latest_stats = {"pages": len(pages), "chunks": added_chunks, "sources": unique_sources}
        st.session_state.latest_docs_preview = [asdict(p) for p in pages[:20]]
        st.session_state.ingest_log.append(
            f"{utc_now_iso()} | {label} | pages={len(pages)} | chunks={added_chunks} | sources={unique_sources}"
        )
        if pages:
            st.success(f"Ingested {len(pages)} pages into {added_chunks} chunks.")
        else:
            st.warning("No usable text was found.")

    if run_crawl and kb_ok:
        with st.spinner("Crawling wildfire web sources..."):
            crawler = WildfireCrawler(
                seed_urls=[x.strip() for x in seed_text.splitlines() if x.strip()],
                allowed_domains=[x.strip() for x in domains_text.splitlines() if x.strip()],
                max_pages=max_pages,
                max_depth=max_depth,
                request_delay_sec=request_delay,
                max_workers=max_workers,
            )
            pages = crawler.crawl()
        finalize_ingest(pages, "crawl")

    if run_incident_ingest and kb_ok:
        with st.spinner("Fetching and indexing incidents..."):
            incidents = fetch_inciweb_incidents(max_items=40)
            st.session_state.last_incidents = incidents
            st.session_state.last_map_df = build_map_df(incidents)
            pages = incidents_to_pages(incidents)
        finalize_ingest(pages, "incidents")

    if run_outlook_ingest and kb_ok:
        with st.spinner("Fetching and indexing outlooks..."):
            pages = fetch_nifc_outlook_pages()
        finalize_ingest(pages, "outlooks")

    if run_weather_ingest and kb_ok:
        with st.spinner("Fetching and indexing fire weather pages and alerts..."):
            fw_pages = fetch_fire_weather_pages([x.strip() for x in fire_weather_urls_text.splitlines() if x.strip()])
            alerts = fetch_nws_alerts_for_bbox_or_area(alert_area)
            alert_pages = alerts_to_pages(alerts)
            pages = fw_pages + alert_pages
        finalize_ingest(pages, "weather+alerts")

    if run_upload_ingest and kb_ok:
        with st.spinner("Reading uploaded files..."):
            pages = ingest_uploaded_files(uploaded_files)
        finalize_ingest(pages, "uploads")

    if run_note_ingest and kb_ok:
        with st.spinner("Indexing note..."):
            pages = ingest_manual_text(manual_title, manual_text, source_type="note")
        finalize_ingest(pages, "note")

    st.markdown("### Latest ingest log")
    if st.session_state.ingest_log:
        for row in reversed(st.session_state.ingest_log[-14:]):
            st.code(row)
    else:
        st.info("Nothing ingested yet.")


# =========================================================
# LIBRARY TAB
# =========================================================
with tabs[4]:
    kb_ok = True
    try:
        kb = get_kb()
    except Exception as e:
        kb_ok = False
        st.error(f"Knowledge base init failed: {e}")

    if kb_ok:
        st.markdown("### Indexed library preview")
        preview = kb.peek(50)
        if preview:
            rows = []
            seen = set()
            for item in preview:
                md = item["metadata"]
                key = (md.get("url", ""), md.get("title", ""), md.get("source_type", ""))
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "title": md.get("title", ""),
                    "type": md.get("source_type", ""),
                    "domain": md.get("source_domain", ""),
                    "url": md.get("url", ""),
                    "fetched_at_utc": md.get("fetched_at_utc", ""),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("The library is empty.")

        st.markdown("### Recent ingest preview")
        if st.session_state.latest_docs_preview:
            st.dataframe(
                pd.DataFrame([{
                    "title": x.get("title", ""),
                    "type": x.get("source_type", ""),
                    "domain": x.get("source_domain", ""),
                    "url": x.get("url", ""),
                    "words": len((x.get("text", "") or "").split()),
                } for x in st.session_state.latest_docs_preview]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No recent ingest preview yet.")


# =========================================================
# SETTINGS TAB
# =========================================================
with tabs[5]:
    st.markdown("### Environment")
    st.code(
        f"CHROMA_DIR={CHROMA_DIR}\n"
        f"COLLECTION_NAME={COLLECTION_NAME}\n"
        f"EMBED_MODEL_NAME={EMBED_MODEL_NAME}\n"
        f"OPENAI_MODEL={OPENAI_MODEL}\n"
        f"OPENAI_KEY_SET={'yes' if bool(OPENAI_API_KEY) else 'no'}\n"
        f"INCIWEB_RSS_URL={INCIWEB_RSS_URL}\n"
        f"NIFC_MONTHLY_OUTLOOK_PDF={NIFC_MONTHLY_OUTLOOK_PDF}\n"
        f"NIFC_NA_OUTLOOK_PDF={NIFC_NA_OUTLOOK_PDF}\n"
        f"NIFC_WEATHER_PAGE={NIFC_WEATHER_PAGE}"
    )

    st.markdown("### Package checks")
    st.write({
        "chromadb": chromadb is not None,
        "sentence_transformers": SentenceTransformer is not None,
        "openai": OpenAI is not None,
        "pypdf": pypdf is not None,
        "pydeck": pdk is not None,
        "trafilatura": True,
        "beautifulsoup4": True,
    })

    kb_ok = True
    try:
        kb = get_kb()
    except Exception as e:
        kb_ok = False
        st.error(f"Knowledge base init failed: {e}")

    if kb_ok:
        st.metric("Current chunk count", f"{kb.count():,}")
        clear = st.button("CLEAR INDEX")
        if clear:
            kb.clear()
            st.session_state.last_answer = ""
            st.session_state.last_hits = []
            st.session_state.last_briefing = ""
            st.session_state.latest_stats = {"pages": 0, "chunks": 0, "sources": 0}
            st.session_state.latest_docs_preview = []
            st.session_state.ingest_log.append(f"{utc_now_iso()} | cleared index")
            st.success("Vector index cleared.")

st.divider()
st.caption("Wildfire Knowledge Lab Pro | incidents + outlooks + fire weather + alerts + map + grounded retrieval")


# =========================================================
# REQUIREMENTS
# =========================================================
# pip install streamlit requests beautifulsoup4 trafilatura chromadb sentence-transformers openai pypdf pandas pydeck
#
# streamlit secrets.toml
# [openai]
# api_key = "YOUR_KEY_HERE"
#
# run:
# streamlit run app.py
