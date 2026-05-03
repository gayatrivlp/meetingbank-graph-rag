"""
MeetingBank GraphRAG — Streamlit Demo (Colab version)
Run with: streamlit run app.py
"""

import os
import streamlit as st
from neo4j import GraphDatabase
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# ── Credentials (Colab testing only — never push this to GitHub) ──────────────
NEO4J_URI      = os.environ.get("NEO4J_URI")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE")
GROQ_KEY = os.environ.get("GROQ_KEY")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MeetingBank GraphRAG",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

  :root {
    --ink:     #0f1117;
    --paper:   #f5f0e8;
    --accent:  #b5451b;
    --accent2: #2563a8;
    --rule:    #d4c9b0;
    --muted:   #6b6050;
    --card-bg: #faf7f2;
  }

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--paper) !important;
    color: var(--ink);
  }

  section[data-testid="stSidebar"] {
    background-color: var(--ink) !important;
    border-right: 3px solid var(--accent);
  }
  section[data-testid="stSidebar"] * { color: var(--paper) !important; }
  section[data-testid="stSidebar"] .stSelectbox label {
    color: var(--rule) !important;
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }

  h1 { font-family: 'DM Serif Display', serif; font-size: 2.6rem !important; color: var(--ink) !important; }
  h2 { font-family: 'DM Serif Display', serif; color: var(--ink) !important; }

  .result-card {
    background: var(--card-bg);
    border: 1px solid var(--rule);
    border-left: 4px solid var(--accent);
    border-radius: 2px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
  }
  .result-card:hover { border-left-color: var(--accent2); }

  .card-meta {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: var(--muted);
    margin-bottom: 0.5rem;
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
  }
  .card-meta span { background: var(--rule); padding: 2px 8px; border-radius: 2px; }
  .card-meta .city  { background: #dce8f5; color: var(--accent2); }
  .card-meta .score { background: #f5dcd4; color: var(--accent); }
  .card-meta .type  { background: #e8f0e0; color: #2d5a1b; }

  .evidence-text {
    font-size: 0.88rem;
    color: var(--muted);
    border-left: 2px solid var(--rule);
    padding-left: 0.8rem;
    margin: 0.6rem 0;
    font-style: italic;
    line-height: 1.6;
  }
  .summary-text { font-size: 0.92rem; color: var(--ink); line-height: 1.7; }

  .answer-box {
    background: var(--ink);
    color: var(--paper);
    border-radius: 2px;
    padding: 1.6rem 2rem;
    font-size: 0.97rem;
    line-height: 1.8;
    border-left: 5px solid var(--accent);
    margin-bottom: 1.5rem;
  }

  .stTextArea > div > div > textarea {
    background: white !important;
    border: 2px solid var(--rule) !important;
    border-radius: 2px !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--ink) !important;
  }
  .stTextArea > div > div > textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: none !important;
  }

  .stButton > button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 2rem !important;
  }
  .stButton > button:hover { background: var(--accent2) !important; }

  .divider { border: none; border-top: 1px solid var(--rule); margin: 1.5rem 0; }
  a { color: var(--accent2) !important; }

  .stage-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.3rem;
  }
</style>
""", unsafe_allow_html=True)


# ── Load resources ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_resources():
    os.environ["NEO4J_DATABASE"] = NEO4J_DATABASE

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )
    embeddings = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1.5",
        model_kwargs={"trust_remote_code": True}
    )

    # llama-3.1-8b-instant for fast filter extraction
    fast_llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature = 0,
        api_key=GROQ_KEY)


    #meta-llama/llama-4-scout-17b-16e-instruct for answer generation
    smart_llm = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature = 0,
        api_key = GROQ_KEY)

    return driver, embeddings, fast_llm, smart_llm


# ── Pipeline functions (mirrors notebook exactly) ─────────────────────────────

STAGE_1_QUERY = """
  CALL db.index.vector.queryNodes('transcript_chunk_index', $top_k, $query_embedding)
  YIELD node AS chunk, score
  MATCH (chunk)<-[:HAS_CHUNK]-(item:Item)
  RETURN item.item_id AS itemId,
         score,
         chunk.text AS evidence
  ORDER BY score DESC
"""

STAGE_2_QUERY = """
  UNWIND $item_data AS data
  MATCH (item:Item {item_id: data.itemId})
  MATCH (meeting:Meeting)-[:HAS_ITEM]->(item)
  MATCH (meeting:Meeting)-[:TAKES_PLACE_IN]->(city:City)
  WHERE $city_name IS NULL OR city.city_name = $city_name
  MATCH (item)-[:OF_TYPE]->(itemType:ItemType)
  WITH item, meeting, city, itemType, data
  WHERE $type_name IS NULL OR itemType.type_name = $type_name
  RETURN
    item.item_id               AS itemId,
    data.score                 AS score,
    data.evidence              AS evidence,
    item.summary               AS itemSummary,
    item.start_time            AS startTime,
    item.end_time              AS endTime,
    item.duration              AS duration,
    itemType.type_name         AS itemType,
    city.city_name             AS city,
    meeting.meeting_id         AS meetingId,
    meeting.webpage_link       AS webpageLink,
    meeting.video_link         AS videoLink,
    meeting.meetingdetail_link AS meetingDetailLink,
    meeting.meeting_date       AS meetingDate
  ORDER BY score DESC
"""
ENTITY_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a query parser for a U.S. city council meeting search system.
Extract structured filters from the user's question.
city_name: One of ["LongBeach","Seattle","Denver","KingCounty","Alameda","Boston"] or null.
  Normalize variants: "Long Beach" → "LongBeach", case-insensitive.
  Only set if the user explicitly mentions a city.
  If the question starts with "which city", "what city", or is asking you to identify the city — return null.
type_name: One of [
    "Agenda Item", "Public Hearing", "Ordinance", "Resolution", "Contract",
    "Emergency Ordinance", "Appointment", "ABC License", "Ordinance (Ord)",
    "Clerk File (CF)", "Resolution (Res)", "Council Bill (CB)",
    "Council Budget Action (CBA)", "Bill", "Proclamation", "Communication",
    "Executive Session", "Presentation", "Announcement", "Motion",
    "Consent Calendar Item", "Continued Agenda Item", "Regular Agenda Item",
    "Joint Agenda Item", "Council Communication", "Council Referral",
    "Joint Consent Item", "Proclamation/Special Order", "SACIC Consent Item",
    "Closed Session Item", "SACIC Regular Item", "Mayor Order",
    "Report of Public Officer", "Committee Reports", "Council Ordinance",
    "Council Hearing Order", "Council Legislative Resolution",
    "Personnel Orders", "Matters Recently Heard-For Possible Action",
    "Mayor Home Rule Petition", "Council Home Rule Petition", "Loan Order",
    "Council 17F Order", "Mayor Ordinance", "Council Order"
] or null.
  Only set if the user explicitly refers to a meeting item type.
Respond ONLY with valid JSON. No explanation, no markdown.
Example: {{"city_name": null, "type_name": null}}
Never return a list. If multiple or all cities apply, return null."""),
    ("human", "User question: {user_query}")
    
])

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert assistant analyzing U.S. city council meeting records
from Seattle, King County, Denver, Boston, Alameda, and Long Beach.
Your task is to answer user questions based the council meetings and the child items information
- Use only the provided context. Do not make up information.
- Do NOT add geographic, demographic, or any other details not stated in the context
- Present results in a clear, professional format.
- If the context doesn't contain enough information, say so explicitly.
- - Do NOT render markdown — no ## headers, no ** bold**, no bullet dashes. Write in plain prose only
- Be concise but complete."""),
    ("human", "User Query:\n{user_query}\n\nRetrieved Context:\n{context}\n\nPlease provide your analysis.")
])


def semantic_search(driver, query_embedding, top_k=20):
    db = os.getenv("NEO4J_DATABASE")
    with driver.session(database=db) as session:
        result = session.run(STAGE_1_QUERY,
                             query_embedding=query_embedding, top_k=top_k)
        return [record.data() for record in result]


def graph_filter_and_augment(driver, semantic_results, city_name=None, type_name=None):
    db = os.getenv("NEO4J_DATABASE")
    with driver.session(database=db) as session:
        result = session.run(STAGE_2_QUERY,
                             item_data=semantic_results,
                             city_name=city_name,
                             type_name=type_name)
        return [record.data() for record in result]


def build_context(enriched_results):
    if not enriched_results:
        return "No relevant meeting items found"
    parts = []
    for r in enriched_results:
        parts.append(
            f"[City: {r['city']} | Meeting: {r['meetingId']} | "
            f"Type: {r.get('itemType', 'Unknown')} | Score: {r['score']:.4f}]\n"
            f"Evidence: {r['evidence']}\n"
            f"Item Summary: {r.get('itemSummary', 'N/A')}"
        )
    return "\n\n---\n\n".join(parts)


def extract_query_filters(fast_llm, user_query):
    chain = ENTITY_EXTRACTION_PROMPT | fast_llm | JsonOutputParser()
    try:
        filters = chain.invoke({"user_query": user_query})
        return {
            "city_name": filters.get("city_name") or None,
            "type_name": filters.get("type_name") or None,
        }
    except Exception:
        return {"city_name": None, "type_name": None}


def generate_answer(smart_llm, user_query, context):
    chain = RAG_PROMPT | smart_llm
    response = chain.invoke({"user_query": user_query, "context": context})
    return response.content


# ── UI ────────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="border-bottom: 3px solid #0f1117; padding-bottom: 1rem; margin-bottom: 1.5rem;">
  <div style="font-family:'DM Mono',monospace; font-size:0.72rem; letter-spacing:0.15em;
              color:#b5451b; text-transform:uppercase; margin-bottom:0.2rem;">
    Portfolio Project · Graph RAG
  </div>
  <h1 style="margin:0; line-height:1.1;">MeetingBank<br>
    <span style="color:#b5451b;">Council Intelligence</span>
  </h1>
  <div style="font-family:'DM Sans',sans-serif; color:#6b6050; margin-top:0.5rem; font-size:0.95rem;">
    1,250 meetings · 6,894 agenda items · 133,536 transcript chunks across 6 U.S. cities
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'DM Serif Display',serif; font-size:1.4rem; margin-bottom:0.2rem;">
      Search Filters
    </div>
    <div style="font-size:0.75rem; color:#9a8f80; margin-bottom:1.2rem;">
      Optionally narrow results by city or item type.<br>
      Leave on Auto-detect to let the model decide.
    </div>
    """, unsafe_allow_html=True)

    city_filter = st.selectbox(
        "City",
        ["Auto-detect", "LongBeach","Seattle","Denver","KingCounty","Alameda","Boston"]
    )
    type_filter = st.selectbox(
        "Item Type",
        ["Auto-detect", "Agenda Item", "Committee Reports", "Proclamation",
         "Proclamation/Special Order", "Public Comment", "Consent Calendar"]
    )
    top_k = st.slider("Top-K chunks (Stage 1)", min_value=5, max_value=50,
                      value=20, step=5)

    st.markdown("<hr style='border-color:#2a2a2a; margin:1rem 0;'>",
                unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.7rem; color:#6b5f50; line-height:1.7;">
      <b style="color:#d4c9b0;">Two-Stage Graph RAG</b><br>
      ① Vector search on TranscriptChunks<br>
      ② Graph traversal for metadata &amp; filtering<br>
    </div>
    """, unsafe_allow_html=True)

# ── Sample queries ────────────────────────────────────────────────────────────
st.markdown("<div class='stage-label'>Try a sample query</div>",
            unsafe_allow_html=True)

samples = [
    "What housing affordability measures were discussed in Denver?",
    "Which city recognized Hispanic Heritage Month and what was said?",
    "What public comments were made about homelessness in Long Beach?",
    "Summarize proclamations made in Boston city council meetings.",
    "What infrastructure or transportation projects came up in Alameda?",
]

# Initialize session state
if "user_query" not in st.session_state:
    st.session_state["user_query"] = ""

cols = st.columns(len(samples))
chosen_sample = None
for i, (col, q) in enumerate(zip(cols, samples)):
    with col:
        if st.button(q[:40] + "…", key=f"sample_{i}", use_container_width=True):
            # chosen_sample = q
            st.session_state["user_query"] = q

# ── Query input ───────────────────────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)

user_query = st.text_area(
    "Your question",
    # value=chosen_sample or "",
    placeholder="Ask anything about U.S. city council meetings…",
    height=90,
    label_visibility="collapsed",
    key="user_query"
)

# Keep session state in sync with manual edits
# st.session_state["user_query"] = user_query

run = st.button("⬡  Search Knowledge Graph")

# ── Execute with step-by-step progress ───────────────────────────────────────
if run:
    if not user_query.strip():
        st.warning("Please enter a question.")
        st.stop()

    city_arg = None if city_filter == "Auto-detect" else city_filter
    type_arg = None if type_filter == "Auto-detect" else type_filter

    # Step 0: load resources
    with st.spinner("Connecting to Neo4j and loading models…"):
        try:
            driver, embeddings, fast_llm, smart_llm = load_resources()
        except Exception as e:
            st.error(f"Failed to load resources: {e}")
            st.stop()
    st.success("✅ Connected to Neo4j and loaded models.")

    # Step 1: extract filters (Haiku — fast)
    with st.spinner("Extracting filters from query…"):
        try:
            if city_arg is None and type_arg is None:
                filters = extract_query_filters(fast_llm, user_query)
                city_arg = filters["city_name"]
                type_arg = filters["type_name"]
        except Exception as e:
            st.error(f"Filter extraction failed: {e}")
            st.stop()
    st.success(f"✅ Filters — city: {city_arg or 'none'}, type: {type_arg or 'none'}")

    # Step 2: embed query
    with st.spinner("Generating query embedding…"):
        try:
            query_embedding = embeddings.embed_query(user_query)
        except Exception as e:
            st.error(f"Embedding failed: {e}")
            st.stop()
    st.success(f"✅ Embedding done — vector length: {len(query_embedding)}")

    # Step 3: vector search
    with st.spinner(f"Stage 1: vector search (top_k={top_k})…"):
        try:
            semantic_results = semantic_search(driver, query_embedding, top_k=top_k)
        except Exception as e:
            st.error(f"Vector search failed: {e}")
            st.stop()
    st.success(f"✅ Stage 1 done — {len(semantic_results)} chunks retrieved.")

    if not semantic_results:
        st.warning("No relevant chunks found for this query.")
        st.stop()

    # Step 4: graph traversal
    with st.spinner("Stage 2: graph traversal and enrichment…"):
        try:
            enriched_results = graph_filter_and_augment(
                driver, semantic_results,
                city_name=city_arg, type_name=type_arg
            )
        except Exception as e:
            st.error(f"Graph traversal failed: {e}")
            st.stop()
    st.success(f"✅ Stage 2 done — {len(enriched_results)} rows returned (items + graph context).")

    # Step 5: generate answer
    with st.spinner("Stage 3: generating answer with LLM…"):
        try:
            context = build_context(enriched_results)
            answer  = generate_answer(smart_llm, user_query, context)
        except Exception as e:
            st.error(f"Answer generation failed: {e}")
            st.stop()
    st.success("✅ Answer generated.")

    # ── Results ───────────────────────────────────────────────────────────────
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("<h2 style='margin-top:1rem;'>Answer</h2>", unsafe_allow_html=True)
    st.markdown(
        f'<div class="answer-box">{answer.replace(chr(10), "<br>")}</div>',
        unsafe_allow_html=True
    )

    # Stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stage 1 chunks", len(semantic_results))
    c2.metric("Stage 2 enriched rows",  len(enriched_results))
    c3.metric("City filter",    city_arg or "None")
    c4.metric("Type filter",    type_arg or "None")

    # Retrieved items
    if enriched_results:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown(
            f"<h2>Retrieved Items "
            f"<span style='font-size:1rem; color:#6b6050;'>({len(enriched_results)} results)</span></h2>",
            unsafe_allow_html=True
        )

        for r in enriched_results:
            links = []
            if r.get("webpageLink"):
                links.append(f'<a href="{r["webpageLink"]}" target="_blank">Webpage</a>')
            if r.get("videoLink"):
                links.append(f'<a href="{r["videoLink"]}" target="_blank">Video</a>')
            if r.get("meetingDetailLink"):
                links.append(f'<a href="{r["meetingDetailLink"]}" target="_blank">Detail</a>')
            links_html = " · ".join(links) if links else ""
            date_str   = r.get("meetingDate", "")
            evidence   = r.get("evidence", "")
            summary    = r.get("itemSummary", "")

            st.markdown(f"""
            <div class="result-card">
              <div class="card-meta">
                <span class="city">🏙 {r.get('city','?')}</span>
                <span class="score">⟐ {r['score']:.4f}</span>
                <span class="type">{r.get('itemType','?')}</span>
                <span>{r.get('meetingId','?')}</span>
                {f'<span>📅 {date_str}</span>' if date_str else ''}
              </div>
              <div class="evidence-text">
                "{evidence[:400]}{'…' if len(evidence) > 400 else ''}"
              </div>
              <div class="summary-text">
                {summary or '<em style="color:#9a8f80;">No summary available</em>'}
              </div>
              {f'<div style="margin-top:0.6rem; font-size:0.8rem;">{links_html}</div>'
               if links_html else ''}
            </div>
            """, unsafe_allow_html=True)

    # Raw context
    with st.expander("View raw LLM context"):
        st.code(context, language="text")
