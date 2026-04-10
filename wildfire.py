import os
import re
import io
import json
import requests
import pandas as pd
import streamlit as st
import trafilatura
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from urllib.parse import urlparse
from bs4 import BeautifulSoup

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import pypdf
except Exception:
    pypdf = None


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Wildfire Knowledge Lab",
    page_icon="🔥",
    layout="wide",
)

st.markdown("""
<style>
.block-container {
    max-width: 100% !important;
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
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
.source-card, .incident-card {
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
APP_TITLE = "Wildfire Knowledge Lab"
APP_SUBTITLE = "Incidents + outlooks + fire weather + alerts + uploads + grounded QA"

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = st.secrets.get("openai", {}).get("api_key") or os.getenv("OPENAI_API_KEY", "")

USER_AGENT = "WildfireKnowledgeLab/1.0 (+research)"
REQUEST_TIMEOUT = 30

INCIWEB_RSS_URL = "https://inciweb.wildfire.gov/feeds/rss/incidents/"
NIFC_MONTHLY_OUTLOOK_PDF = "https://www.nifc.gov/nicc-files/predictive/outlooks/monthly_seasonal_outlook.pdf"
NIFC_NA_OUTLOOK_PDF = "https://www.nifc.gov/nicc-files/predictive/outlooks/NA_Outlook.pdf"
NIFC_WEATHER_PAGE = "https://www.nifc.gov/nicc/predictive-services/weather"

DEFAULT_FIRE_WEATHER_URLS = [
    "https://www.weather.gov/gjt/fire",
    "https://www.weather.gov/unr/brief_fire",
]

if "doc_store" not in st.session_state:
    st.session_state.doc_store = []
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""
if "last_hits" not in st.session_state:
    st.session_state.last_hits = []
if "last_briefing" not in st.session_state:
    st.session_state.last_briefing = ""
if "last_incidents" not in st.session_state:
    st.session_state.last_incidents = []
if "ingest_log" not in st.session_state:
    st.session_state.ingest_log = []


# =========================================================
# HELPERS
# =========================================================
def utc_now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def maybe_trim(text: str, n: int = 1200) -> str:
    text = text or ""
    return text[:n] + ("..." if len(text) > n else "")


def fetch_text_url(url: str, timeout: int = REQUEST_TIMEOUT):
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    r.raise_for_status()
    ctype = r.headers.get("Content-Type", "").lower()
    if "text" in ctype or "html" in ctype or "xml" in ctype:
        return r.text, ctype
    return r.content.decode("utf-8", errors="ignore"), ctype


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


def extract_lat_lon_from_text(text: str):
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


def get_openai_client():
    if OpenAI is None:
        raise RuntimeError("openai package is not installed.")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set.")
    return OpenAI(api_key=OPENAI_API_KEY)


# =========================================================
# SIMPLE IN-MEMORY STORE
# =========================================================
def add_docs(docs):
    existing_keys = {(d["title"], d["url"], d["source_type"]) for d in st.session_state.doc_store}
    added = 0
    for doc in docs:
        key = (doc["title"], doc["url"], doc["source_type"])
        if key not in existing_keys:
            st.session_state.doc_store.append(doc)
            existing_keys.add(key)
            added += 1
    return added


def simple_search(query: str, top_k: int = 8):
    query_terms = [t.lower() for t in re.findall(r"\w+", query) if len(t) > 2]
    hits = []
    for doc in st.session_state.doc_store:
        text = f"{doc.get('title', '')}\n{doc.get('text', '')}".lower()
        score = 0
        for term in query_terms:
            score += text.count(term)
        if score > 0:
            hits.append({
                "score": score,
                "title": doc.get("title", ""),
                "url": doc.get("url", ""),
                "source_type": doc.get("source_type", ""),
                "source_domain": doc.get("source_domain", ""),
                "text": doc.get("text", "")
            })
    hits = sorted(hits, key=lambda x: x["score"], reverse=True)
    return hits[:top_k]


# =========================================================
# SOURCE INGESTORS
# =========================================================
def ingest_uploaded_files(files):
    docs = []
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

        if len(text.split()) < 20:
            continue

        docs.append({
            "title": fname,
            "url": f"upload://{fname}",
            "text": text,
            "source_domain": "uploaded-file",
            "source_type": "upload",
            "fetched_at_utc": utc_now_iso(),
        })
    return docs


def ingest_manual_text(title: str, text: str, source_type: str = "note"):
    text = clean_text(text)
    if len(text.split()) < 20:
        return []
    return [{
        "title": title.strip() or "Untitled",
        "url": f"{source_type}://manual",
        "text": text,
        "source_domain": source_type,
        "source_type": source_type,
        "fetched_at_utc": utc_now_iso(),
    }]


def fetch_inciweb_incidents(rss_url: str = INCIWEB_RSS_URL, max_items: int = 30):
    xml_text, _ = fetch_text_url(rss_url)
    root = ET.fromstring(xml_text)

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


def incidents_to_docs(incidents):
    docs = []
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
        docs.append({
            "title": x.get("title", "InciWeb Incident"),
            "url": x.get("link", "") or "inciweb://incident",
            "text": text,
            "source_domain": "inciweb.wildfire.gov",
            "source_type": "incident",
            "fetched_at_utc": utc_now_iso(),
        })
    return docs


def fetch_nifc_outlook_docs():
    docs = []
    pdf_targets = [
        ("NIFC National Significant Wildland Fire Potential Outlook", NIFC_MONTHLY_OUTLOOK_PDF),
        ("NIFC North American Seasonal Fire Assessment and Outlook", NIFC_NA_OUTLOOK_PDF),
    ]
    for title, url in pdf_targets:
        try:
            pdf_bytes = fetch_bytes_url(url)
            text = read_pdf_bytes(pdf_bytes)
            if len(text.split()) >= 40:
                docs.append({
                    "title": title,
                    "url": url,
                    "text": text,
                    "source_domain": "nifc.gov",
                    "source_type": "outlook",
                    "fetched_at_utc": utc_now_iso(),
                })
        except Exception:
            pass

    try:
        html, _ = fetch_text_url(NIFC_WEATHER_PAGE)
        txt = trafilatura.extract(html, include_comments=False, include_tables=False, no_fallback=False) or ""
        txt = clean_text(txt)
        if len(txt.split()) >= 40:
            docs.append({
                "title": "NIFC Predictive Services Weather",
                "url": NIFC_WEATHER_PAGE,
                "text": txt,
                "source_domain": "nifc.gov",
                "source_type": "predictive-services",
                "fetched_at_utc": utc_now_iso(),
            })
    except Exception:
        pass

    return docs


def scrape_fire_weather_page(url: str):
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

        return {
            "title": title,
            "url": url,
            "text": text,
            "source_domain": urlparse(url).netloc.lower(),
            "source_type": "fire-weather",
            "fetched_at_utc": utc_now_iso(),
        }
    except Exception:
        return None


def fetch_fire_weather_docs(urls):
    docs = []
    for u in urls:
        rec = scrape_fire_weather_page(u)
        if rec:
            docs.append(rec)
    return docs


def fetch_nws_alerts_for_area(area: str = ""):
    base = "https://api.weather.gov/alerts/active"
    params = {}
    if area.strip():
        params["area"] = area.strip().upper()
    try:
        r = requests.get(
            base,
            headers={"User-Agent": USER_AGENT, "Accept": "application/geo+json"},
            params=params,
            timeout=REQUEST_TIMEOUT
        )
        r.raise_for_status()
        js = r.json()
        feats = js.get("features", [])
        out = []
        for f in feats:
            prop = f.get("properties", {})
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
                    "url": prop.get("@id", ""),
                })
        return out
    except Exception:
        return []


def alerts_to_docs(alerts):
    docs = []
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
        docs.append({
            "title": a.get("headline", "") or a.get("event", "NWS Fire Alert"),
            "url": a.get("url", "") or "nws-alert://alert",
            "text": text,
            "source_domain": "api.weather.gov",
            "source_type": "alert",
            "fetched_at_utc": utc_now_iso(),
        })
    return docs


# =========================================================
# LLM
# =========================================================
def build_grounded_prompt(question: str, hits):
    blocks = []
    for i, h in enumerate(hits, start=1):
        blocks.append(
            f"[Source {i}]\n"
            f"Title: {h.get('title', '')}\n"
            f"Type: {h.get('source_type', '')}\n"
            f"URL: {h.get('url', '')}\n"
            f"Domain: {h.get('source_domain', '')}\n"
            f"Excerpt:\n{maybe_trim(h.get('text', ''), 2000)}\n"
        )
    system_prompt = (
        "You are a wildfire knowledge analyst. Use only the provided sources. "
        "Be direct, practical, and grounded. If the sources are incomplete, say so."
    )
    user_prompt = f"Question:\n{question}\n\nRetrieved sources:\n\n" + "\n\n".join(blocks)
    return system_prompt, user_prompt


def answer_question_with_openai(question: str, hits):
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
    return resp.choices[0].message.content.strip()


def answer_question_local(question: str, hits):
    if not hits:
        return "No indexed wildfire sources are available yet. Ingest incidents, outlooks, weather pages, alerts, or uploads first."
    lines = [f"Question: {question}", "", "Top retrieved sources:", ""]
    for i, h in enumerate(hits, start=1):
        lines.append(f"{i}. {h.get('title', '')} | {h.get('source_type', '')} | {h.get('url', '')}")
        lines.append(maybe_trim(h.get("text", ""), 800))
        lines.append("")
    lines.append("This is retrieval-only output because no OpenAI key was available.")
    return "\n".join(lines)


def generate_daily_briefing(mode: str, incidents, outlook_docs, weather_docs, alert_docs, hits):
    if not OPENAI_API_KEY or OpenAI is None:
        return (
            f"Daily briefing mode: {mode}\n\n"
            f"Incidents loaded: {len(incidents)}\n"
            f"Outlook docs loaded: {len(outlook_docs)}\n"
            f"Weather docs loaded: {len(weather_docs)}\n"
            f"Fire alert docs loaded: {len(alert_docs)}\n"
            f"Retrieved context docs: {len(hits)}\n\n"
            "OpenAI key was not available, so this is only a raw status summary."
        )

    client = get_openai_client()

    incident_text = []
    for i, inc in enumerate(incidents[:12], start=1):
        incident_text.append(
            f"[Incident {i}] {inc.get('title', '')}\n"
            f"Published: {inc.get('pub_date', '')}\n"
            f"Lat/Lon: {inc.get('lat', '')}, {inc.get('lon', '')}\n"
            f"Summary: {maybe_trim(inc.get('description', ''), 900)}"
        )

    outlook_text = [f"[Outlook] {d['title']}\n{maybe_trim(d['text'], 1600)}" for d in outlook_docs[:4]]
    weather_text = [f"[Weather] {d['title']}\n{maybe_trim(d['text'], 1200)}" for d in weather_docs[:6]]
    alert_text = [f"[Alert] {d['title']}\n{maybe_trim(d['text'], 800)}" for d in alert_docs[:6]]
    retrieval_text = [f"[Retrieved] {h['title']}\n{maybe_trim(h['text'], 1000)}" for h in hits[:6]]

    system_prompt = (
        "You are a wildfire analyst building a daily intelligence briefing from provided incident, outlook, weather, alert, and retrieved knowledge inputs. "
        "Only use the provided material. Be concrete and structured."
    )
    user_prompt = (
        f"Mode: {mode}\n\n"
        "Write a concise daily briefing with these sections:\n"
        "1. Executive Summary\n"
        "2. Incident Picture\n"
        "3. Weather and Outlook Signals\n"
        "4. Watch Items\n"
        "5. Source Notes\n\n"
        "Incidents:\n\n" + "\n\n".join(incident_text) + "\n\n"
        "Outlooks:\n\n" + "\n\n".join(outlook_text) + "\n\n"
        "Weather:\n\n" + "\n\n".join(weather_text) + "\n\n"
        "Alerts:\n\n" + "\n\n".join(alert_text) + "\n\n"
        "Retrieved context:\n\n" + "\n\n".join(retrieval_text)
    )

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


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
m1.markdown(f'<div class="metric-card"><div style="font-size:0.85rem; opacity:0.7;">Stored docs</div><div style="font-size:1.15rem; font-weight:800;">{len(st.session_state.doc_store)}</div></div>', unsafe_allow_html=True)
m2.markdown(f'<div class="metric-card"><div style="font-size:0.85rem; opacity:0.7;">LLM</div><div style="font-size:1.15rem; font-weight:800;">{OPENAI_MODEL}</div></div>', unsafe_allow_html=True)
m3.markdown(f'<div class="metric-card"><div style="font-size:0.85rem; opacity:0.7;">OpenAI key</div><div style="font-size:1.15rem; font-weight:800;">{"set" if OPENAI_API_KEY else "not set"}</div></div>', unsafe_allow_html=True)
m4.markdown(f'<div class="metric-card"><div style="font-size:0.85rem; opacity:0.7;">Time</div><div style="font-size:1.15rem; font-weight:800;">{datetime.now().strftime("%Y-%m-%d %H:%M")}</div></div>', unsafe_allow_html=True)

st.divider()

tabs = st.tabs(["Daily Briefing", "Incidents", "Ask", "Ingest", "Library", "Settings"])


# =========================================================
# DAILY BRIEFING TAB
# =========================================================
with tabs[0]:
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
        with st.spinner("Pulling incident feed..."):
            incidents = fetch_inciweb_incidents()

        with st.spinner("Pulling outlooks and predictive services..."):
            outlook_docs = fetch_nifc_outlook_docs()

        with st.spinner("Pulling fire weather pages..."):
            weather_docs = fetch_fire_weather_docs(DEFAULT_FIRE_WEATHER_URLS)

        with st.spinner("Pulling fire-related alerts..."):
            alerts = fetch_nws_alerts_for_area(alert_area)
            alert_docs = alerts_to_docs(alerts)

        with st.spinner("Retrieving stored context..."):
            hits = simple_search(f"{briefing_focus}\n{retrieve_query}", top_k=8)

        with st.spinner("Generating briefing..."):
            briefing = generate_daily_briefing(
                mode=briefing_mode,
                incidents=incidents,
                outlook_docs=outlook_docs,
                weather_docs=weather_docs,
                alert_docs=alert_docs,
                hits=hits,
            )

        st.session_state.last_briefing = briefing
        st.session_state.last_incidents = incidents

        st.markdown("### Daily briefing")
        st.write(briefing)

        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Incidents", f"{len(incidents):,}")
        q2.metric("Outlook docs", f"{len(outlook_docs):,}")
        q3.metric("Weather docs", f"{len(weather_docs):,}")
        q4.metric("Fire alerts", f"{len(alert_docs):,}")

    elif st.session_state.last_briefing:
        st.markdown("### Last briefing")
        st.write(st.session_state.last_briefing)


# =========================================================
# INCIDENTS TAB
# =========================================================
with tabs[1]:
    max_incidents = st.slider("Incident count", 5, 50, 20, 1)

    st.markdown('<div class="big-button">', unsafe_allow_html=True)
    run_incidents = st.button("REFRESH INCIDENT FEED", use_container_width=True, type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

    if run_incidents or not st.session_state.last_incidents:
        with st.spinner("Loading InciWeb incidents..."):
            try:
                st.session_state.last_incidents = fetch_inciweb_incidents(max_items=max_incidents)
            except Exception as e:
                st.error(f"Failed to load incidents: {e}")

    incidents = st.session_state.last_incidents[:max_incidents]

    if incidents:
        rows = []
        for inc in incidents:
            rows.append({
                "title": inc.get("title", ""),
                "pub_date": inc.get("pub_date", ""),
                "lat": inc.get("lat"),
                "lon": inc.get("lon"),
                "link": inc.get("link", ""),
            })

        map_rows = [r for r in rows if r["lat"] is not None and r["lon"] is not None]
        if map_rows:
            st.markdown("### Incident map")
            map_df = pd.DataFrame(map_rows).rename(columns={"lat": "latitude", "lon": "longitude"})
            st.map(map_df[["latitude", "longitude"]], use_container_width=True)

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

        st.markdown("### Incident table")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No incidents loaded yet.")


# =========================================================
# ASK TAB
# =========================================================
with tabs[2]:
    a1, a2 = st.columns([1.4, 1])

    with a1:
        question = st.text_area(
            "Ask a wildfire question",
            value="What are the key themes across the stored wildfire incident, outlook, weather, and policy sources?",
            height=150,
        )
        top_k = st.slider("Retrieved source count", 3, 12, 8, 1)
        st.markdown('<div class="big-button">', unsafe_allow_html=True)
        run_ask = st.button("ASK", use_container_width=True, type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

    with a2:
        st.metric("Stored docs", f"{len(st.session_state.doc_store):,}")
        type_counts = pd.Series([d["source_type"] for d in st.session_state.doc_store]).value_counts().to_dict() if st.session_state.doc_store else {}
        st.write(type_counts)

    if run_ask:
        with st.spinner("Searching stored wildfire knowledge..."):
            hits = simple_search(question, top_k=top_k)
            st.session_state.last_hits = hits

        with st.spinner("Building grounded answer..."):
            try:
                if OPENAI_API_KEY and OpenAI is not None:
                    answer = answer_question_with_openai(question, hits)
                else:
                    answer = answer_question_local(question, hits)
            except Exception as e:
                answer = answer_question_local(question, hits)
                answer += f"\n\nOpenAI call failed and the app fell back to retrieval-only output.\nError: {e}"

        st.session_state.last_answer = answer

        st.markdown("### Answer")
        st.write(answer)

        st.markdown("### Retrieved sources")
        for i, h in enumerate(hits, start=1):
            st.markdown(
                f"""
                <div class="source-card">
                    <div style="font-size:1rem; font-weight:800;">[{i}] {h.get("title", "Untitled")}</div>
                    <div style="font-size:0.86rem; opacity:0.72; margin-bottom:0.45rem;">
                        {h.get("source_type", "")} &nbsp;|&nbsp; {h.get("source_domain", "")} &nbsp;|&nbsp;
                        <a href="{h.get("url", "")}" target="_blank">{h.get("url", "")}</a>
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
    st.markdown("### Ingest wildfire sources")

    i1, i2 = st.columns([1.1, 1])

    with i1:
        fire_weather_urls_text = st.text_area("Fire weather page URLs", value="\n".join(DEFAULT_FIRE_WEATHER_URLS), height=120)

    with i2:
        alert_area = st.text_input("Alert area code for fire-related NWS alerts", value="CO")

    b1, b2, b3 = st.columns(3)
    with b1:
        run_incident_ingest = st.button("INGEST INCIDENTS", use_container_width=True)
    with b2:
        run_outlook_ingest = st.button("INGEST OUTLOOKS", use_container_width=True)
    with b3:
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

    if run_incident_ingest:
        with st.spinner("Fetching and storing incidents..."):
            incidents = fetch_inciweb_incidents(max_items=40)
            st.session_state.last_incidents = incidents
            docs = incidents_to_docs(incidents)
            added = add_docs(docs)
            st.session_state.ingest_log.append(f"{utc_now_iso()} | incidents | docs={added}")
            st.success(f"Added {added} incident docs.")

    if run_outlook_ingest:
        with st.spinner("Fetching and storing outlooks..."):
            docs = fetch_nifc_outlook_docs()
            added = add_docs(docs)
            st.session_state.ingest_log.append(f"{utc_now_iso()} | outlooks | docs={added}")
            st.success(f"Added {added} outlook docs.")

    if run_weather_ingest:
        with st.spinner("Fetching and storing weather and alerts..."):
            fw_docs = fetch_fire_weather_docs([x.strip() for x in fire_weather_urls_text.splitlines() if x.strip()])
            alerts = fetch_nws_alerts_for_area(alert_area)
            alert_docs = alerts_to_docs(alerts)
            docs = fw_docs + alert_docs
            added = add_docs(docs)
            st.session_state.ingest_log.append(f"{utc_now_iso()} | weather+alerts | docs={added}")
            st.success(f"Added {added} weather/alert docs.")

    if run_upload_ingest:
        with st.spinner("Reading uploaded files..."):
            docs = ingest_uploaded_files(uploaded_files)
            added = add_docs(docs)
            st.session_state.ingest_log.append(f"{utc_now_iso()} | uploads | docs={added}")
            st.success(f"Added {added} uploaded docs.")

    if run_note_ingest:
        with st.spinner("Storing note..."):
            docs = ingest_manual_text(manual_title, manual_text, source_type="note")
            added = add_docs(docs)
            st.session_state.ingest_log.append(f"{utc_now_iso()} | note | docs={added}")
            st.success(f"Added {added} note docs.")

    st.markdown("### Ingest log")
    if st.session_state.ingest_log:
        for row in reversed(st.session_state.ingest_log[-20:]):
            st.code(row)
    else:
        st.info("Nothing ingested yet.")


# =========================================================
# LIBRARY TAB
# =========================================================
with tabs[4]:
    st.markdown("### Stored document library")
    if st.session_state.doc_store:
        rows = []
        for d in st.session_state.doc_store:
            rows.append({
                "title": d.get("title", ""),
                "type": d.get("source_type", ""),
                "domain": d.get("source_domain", ""),
                "url": d.get("url", ""),
                "words": len((d.get("text", "") or "").split()),
                "fetched_at_utc": d.get("fetched_at_utc", ""),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("The library is empty.")


# =========================================================
# SETTINGS TAB
# =========================================================
with tabs[5]:
    st.markdown("### Environment")
    st.code(
        f"OPENAI_MODEL={OPENAI_MODEL}\n"
        f"OPENAI_KEY_SET={'yes' if bool(OPENAI_API_KEY) else 'no'}\n"
        f"INCIWEB_RSS_URL={INCIWEB_RSS_URL}\n"
        f"NIFC_MONTHLY_OUTLOOK_PDF={NIFC_MONTHLY_OUTLOOK_PDF}\n"
        f"NIFC_NA_OUTLOOK_PDF={NIFC_NA_OUTLOOK_PDF}\n"
        f"NIFC_WEATHER_PAGE={NIFC_WEATHER_PAGE}\n"
        f"STORED_DOCS={len(st.session_state.doc_store)}"
    )

    st.markdown("### Package checks")
    st.write({
        "openai": OpenAI is not None,
        "pypdf": pypdf is not None,
        "trafilatura": True,
        "beautifulsoup4": True,
        "requests": True,
        "pandas": True,
    })

    clear = st.button("CLEAR STORED DOCS")
    if clear:
        st.session_state.doc_store = []
        st.session_state.last_answer = ""
        st.session_state.last_hits = []
        st.session_state.last_briefing = ""
        st.session_state.ingest_log.append(f"{utc_now_iso()} | cleared docs")
        st.success("Stored docs cleared.")

st.divider()
st.caption("Wildfire Knowledge Lab | incidents + outlooks + fire weather + alerts + uploads + grounded retrieval")
