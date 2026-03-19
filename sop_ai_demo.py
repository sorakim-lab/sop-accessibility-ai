"""
SOP Accessibility + AI Support Layer
Claude explains why results appeared, surfaces related procedures,
and adds reasoning notes that keep interpretive openness alive.

Run: streamlit run sop_search.py
Optional: set ANTHROPIC_API_KEY for AI support layer
Falls back to rule-based mode if no API key.

Fixes applied:
- Model updated to claude-sonnet-4-20250514
- Error results no longer cached (allows retry on failure)
- highlight_text() now strips whitespace-only terms for consistency with baseline
"""

import json
import os
import re
from typing import List, Dict, Tuple, Optional

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="SOP Accessibility + AI Support Layer",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# Demo data
# =========================================================
DATA = [
    {
        "doc_id": "SOP-001",
        "title": "Environmental Monitoring Sampling Procedure",
        "section": "Section 4.2 Sampling Sequence",
        "page": 12,
        "keywords": ["environmental monitoring", "sampling", "sequence", "room", "plate", "contact plate"],
        "text": (
            "This procedure describes the sequence of environmental monitoring sampling steps. "
            "Operators must follow room entry order, sampling location order, and plate handling instructions. "
            "Cross-check with gowning and aseptic behavior procedures before execution."
        ),
        "related_docs": ["SOP-004", "SOP-006"],
    },
    {
        "doc_id": "SOP-002",
        "title": "Deviation Handling and Initial Assessment",
        "section": "Section 3.1 Immediate Documentation",
        "page": 5,
        "keywords": ["deviation", "documentation", "initial assessment", "record", "event"],
        "text": (
            "This procedure explains how deviations must be documented immediately after discovery. "
            "Initial assessment should capture event timing, affected process step, preliminary impact, "
            "and whether escalation is needed."
        ),
        "related_docs": ["SOP-003", "SOP-005"],
    },
    {
        "doc_id": "SOP-003",
        "title": "Root Cause Analysis and CAPA Workflow",
        "section": "Section 5.4 Cause Exploration",
        "page": 18,
        "keywords": ["root cause", "rca", "capa", "cause", "investigation", "human error"],
        "text": (
            "This procedure defines investigation steps for root cause analysis and CAPA planning. "
            "Investigators should examine procedural ambiguity, equipment context, training records, "
            "and workflow conditions before attributing the event to operator error."
        ),
        "related_docs": ["SOP-002", "SOP-005"],
    },
    {
        "doc_id": "SOP-004",
        "title": "Aseptic Gowning and Entry Procedure",
        "section": "Section 2.3 Entry Requirements",
        "page": 7,
        "keywords": ["gowning", "entry", "aseptic", "room entry", "operator"],
        "text": (
            "This procedure covers aseptic gowning steps, entry order, and operator behavior "
            "before entering controlled rooms. "
            "Users should review this procedure before any environmental monitoring activity."
        ),
        "related_docs": ["SOP-001"],
    },
    {
        "doc_id": "SOP-005",
        "title": "Change Control and Procedure Revision",
        "section": "Section 6.1 Revision Triggers",
        "page": 22,
        "keywords": ["change control", "revision", "procedure revision", "document update"],
        "text": (
            "This procedure explains when a controlled procedure should be revised, "
            "how change control is initiated, "
            "and how affected records and downstream processes should be identified."
        ),
        "related_docs": ["SOP-002", "SOP-003"],
    },
    {
        "doc_id": "SOP-006",
        "title": "Plate Handling and Incubation Conditions",
        "section": "Section 4.1 Plate Transfer",
        "page": 10,
        "keywords": ["plate", "incubation", "handling", "transfer", "environmental monitoring"],
        "text": (
            "This procedure describes how contact plates and settle plates should be handled, "
            "transported, and incubated after environmental monitoring collection."
        ),
        "related_docs": ["SOP-001"],
    },
]

df = pd.DataFrame(DATA)

# =========================================================
# Rule-based search (always available)
# =========================================================
STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "by",
    "is", "are", "be", "this", "that", "how", "what", "when", "should", "must",
    "after", "before", "into", "from", "through", "about",
}

def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z0-9\-]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

def highlight_text(text: str, query_terms: List[str]) -> str:
    highlighted = text
    for term in sorted(set(query_terms), key=len, reverse=True):
        # FIX 3: strip whitespace-only terms for consistency with baseline
        if not term.strip() or len(term) < 2:
            continue
        pattern = re.compile(rf"({re.escape(term)})", re.IGNORECASE)
        highlighted = pattern.sub(
            r'<mark style="background:#fef08a;color:#111827;border-radius:3px;padding:0 2px;">\1</mark>',
            highlighted,
        )
    return highlighted

def score_row(row: pd.Series, query_terms: List[str]) -> Tuple[int, Dict[str, int]]:
    title_tokens   = tokenize(row["title"])
    section_tokens = tokenize(row["section"])
    keyword_tokens = [k.lower() for k in row["keywords"]]
    text_tokens    = tokenize(row["text"])
    contribution   = {"title": 0, "section": 0, "keywords": 0, "text": 0}
    for term in query_terms:
        contribution["title"]    += title_tokens.count(term)   * 4
        contribution["section"]  += section_tokens.count(term) * 3
        contribution["keywords"] += keyword_tokens.count(term) * 5
        contribution["text"]     += text_tokens.count(term)    * 1
    return sum(contribution.values()), contribution

def search_docs(query: str, data: pd.DataFrame) -> pd.DataFrame:
    query_terms = tokenize(query)
    rows = []
    for _, row in data.iterrows():
        total, contribution = score_row(row, query_terms)
        if total > 0:
            d = row.to_dict()
            d["score"]        = total
            d["contribution"] = contribution
            d["query_terms"]  = query_terms
            rows.append(d)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

def score_label(score: int) -> Tuple[str, str, str]:
    if score >= 15: return "Strong match", "#16a34a", "#f0fdf4"
    elif score >= 7: return "Good match",  "#d97706", "#fefce8"
    else:            return "Weak match",  "#6b7280", "#f9fafb"

def get_related_titles(row: pd.Series, data: pd.DataFrame) -> List[Dict]:
    out = []
    for doc_id in row["related_docs"]:
        matched = data[data["doc_id"] == doc_id]
        if not matched.empty:
            out.append({
                "doc_id": doc_id,
                "title":  matched.iloc[0]["title"],
                "section": matched.iloc[0]["section"],
            })
    return out

# =========================================================
# Rule-based fallbacks (no API)
# =========================================================
def fallback_why(row: pd.Series) -> str:
    parts = []
    c = row["contribution"]
    if c["keywords"] > 0: parts.append("matched indexed procedural keywords")
    if c["title"]    > 0: parts.append("matched document title terms")
    if c["section"]  > 0: parts.append("matched section-level cues")
    if c["text"]     > 0: parts.append("matched supporting body text")
    if not parts:
        return "Retrieved by weak contextual overlap."
    return "Retrieved because it " + ", ".join(parts) + "."

def fallback_reasoning_note(query: str) -> str:
    q = query.lower()
    if "deviation" in q or "rca" in q or "root cause" in q:
        return (
            "Before narrowing the issue to operator error, review whether the event may also involve "
            "procedural ambiguity, missing cross-references, or workflow constraints."
        )
    if "sampling" in q or "environmental monitoring" in q:
        return (
            "This task may require checking adjacent procedures such as entry, gowning, or plate handling "
            "before execution. Retrieval difficulty itself may be part of the work burden."
        )
    if "change control" in q or "revision" in q:
        return (
            "Consider not only the revision trigger itself, but also which downstream documents and "
            "workflows become dependent on this update."
        )
    return (
        "This result may be relevant, but verify neighboring procedures, section context, "
        "and downstream dependencies before acting."
    )

def fallback_related_note(related_title: str) -> str:
    return f"Cross-referenced in document mapping."

# =========================================================
# Claude AI support layer
# =========================================================
def get_api_key() -> Optional[str]:
    return os.getenv("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY", None)

def run_claude_support(
    query: str,
    doc: Dict,
    related_docs: List[Dict],
) -> Dict:
    """
    Ask Claude to explain why this document appeared,
    what related procedures matter and why,
    and what reasoning note to surface.
    Returns dict with keys: why, reasoning_note, related_notes
    """
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=get_api_key())

        related_str = "\n".join(
            f"- {r['doc_id']}: {r['title']} ({r['section']})"
            for r in related_docs
        ) or "None"

        prompt = f"""You are an AI support layer for procedural document retrieval in a regulated pharmaceutical environment.

Your job is NOT to make compliance decisions or give final answers.
Your job is to make the retrieval process more legible — explain why a result appeared,
surface what adjacent procedures matter, and keep interpretive openness alive.

The user searched for: "{query}"

This document was retrieved:
- ID: {doc['doc_id']}
- Title: {doc['title']}
- Section: {doc['section']}
- Keywords: {', '.join(doc['keywords'])}
- Text: {doc['text']}

Related documents mapped to this entry:
{related_str}

Return ONLY valid JSON with this exact shape:
{{
  "why": "2-3 sentences explaining why this document is relevant to the query. Be specific about what in the document connects to what in the query.",
  "reasoning_note": "2-3 sentences that keep interpretive openness alive. Surface what might be ambiguous, what adjacent procedures matter, or what should not be assumed before reading. Do not give a compliance answer.",
  "related_notes": [
    {{"doc_id": "SOP-XXX", "why_relevant": "1 sentence explaining why this related doc matters for this specific query"}}
  ]
}}"""

        message = client.messages.create(
            # FIX 1: updated to current model
            model="claude-sonnet-4-20250514",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = message.content[0].text.strip()
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        return json.loads(raw.strip())

    except Exception as e:
        return {"error": str(e)}

# =========================================================
# HTML helpers
# =========================================================
CARD = ('background:#ffffff;border:1px solid #e5e7eb;border-radius:12px;'
        'padding:22px 24px;margin-bottom:12px;box-shadow:0 1px 3px rgba(0,0,0,.05);')

def render(html: str):
    st.markdown(html, unsafe_allow_html=True)

def overline(t: str) -> str:
    return (f'<div style="font-size:11px;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:.08em;color:#9ca3af;margin-bottom:8px;">{t}</div>')

def heading(t: str, size="17px", mb="10px") -> str:
    return (f'<div style="font-size:{size};font-weight:700;color:#111827;'
            f'line-height:1.35;margin-bottom:{mb};">{t}</div>')

def body(t: str) -> str:
    return f'<div style="font-size:14px;color:#4b5563;line-height:1.7;">{t}</div>'

def slabel(t: str, mt="14px") -> str:
    return (f'<div style="font-size:11px;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:.07em;color:#9ca3af;margin:{mt} 0 6px;">{t}</div>')

# =========================================================
# CSS
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [data-testid="stAppViewContainer"], .stApp {
    background: #f4f5f7 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stHeader"],[data-testid="stDecoration"],
[data-testid="stToolbar"],footer { display:none !important; }

.block-container { max-width:1360px !important; padding:24px 32px 40px !important; }
[data-testid="stVerticalBlock"] > div:empty { display:none !important; }

div[data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="stVerticalBlockBorderWrapper"] > div {
    border:none !important; background:transparent !important;
    box-shadow:none !important; padding:0 !important;
    border-radius:0 !important; margin:0 !important;
}

div[data-testid="stTextInput"] input {
    font-family:'Inter',sans-serif !important; font-size:15px !important;
    color:#111827 !important; background:#ffffff !important;
    border:1.5px solid #d1d5db !important; border-radius:10px !important;
    padding:12px 16px !important; box-shadow:0 1px 3px rgba(0,0,0,.05) !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color:#2563eb !important; outline:none !important;
    box-shadow:0 0 0 3px rgba(37,99,235,.1) !important;
}

div[data-testid="stButton"] > button {
    font-family:'Inter',sans-serif !important;
    font-size:13px !important; font-weight:600 !important;
    color:#374151 !important; background:#ffffff !important;
    border:1.5px solid #d1d5db !important; border-radius:9px !important;
    padding:8px 14px !important; min-height:2.4rem !important;
    box-shadow:0 1px 2px rgba(0,0,0,.05) !important;
    transition:all .12s ease !important; width:100% !important;
}
div[data-testid="stButton"] > button:hover {
    background:#f9fafb !important; border-color:#9ca3af !important; color:#111827 !important;
}

[data-testid="stExpander"] {
    border:1px solid #e5e7eb !important; border-radius:9px !important;
    background:#f9fafb !important;
}
[data-testid="stExpander"] summary {
    font-size:13px !important; font-weight:600 !important; color:#374151 !important;
}
[data-testid="stDataFrame"] { border-radius:8px !important; overflow:hidden; }
p, li { font-size:14px !important; color:#4b5563 !important; line-height:1.7 !important; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# Session state
# =========================================================
if "query" not in st.session_state:
    st.session_state.query = ""
if "ai_cache" not in st.session_state:
    st.session_state.ai_cache = {}  # cache by (query, doc_id)
if "toast_msg" not in st.session_state:
    st.session_state.toast_msg = None

if st.session_state.toast_msg:
    st.toast(st.session_state.toast_msg)
    st.session_state.toast_msg = None

has_api = get_api_key() is not None

# =========================================================
# Header
# =========================================================
ai_badge = (
    '<span style="display:inline-block;padding:3px 10px;background:#f5f3ff;'
    'color:#7c3aed;border:1px solid #ddd6fe;border-radius:5px;'
    'font-size:12px;font-weight:700;margin-left:8px;">AI support active</span>'
    if has_api else
    '<span style="display:inline-block;padding:3px 10px;background:#f9fafb;'
    'color:#9ca3af;border:1px solid #e5e7eb;border-radius:5px;'
    'font-size:12px;font-weight:700;margin-left:8px;">Rule-based mode</span>'
)

render(f"""
<div style="{CARD}margin-bottom:16px;">
  <div style="display:flex;align-items:flex-start;justify-content:space-between;
      flex-wrap:wrap;gap:16px;">
    <div style="max-width:700px;">
      {overline("SOP Accessibility · AI Support Layer Prototype")}
      <div style="font-size:24px;font-weight:700;color:#111827;letter-spacing:-.02em;
          margin-bottom:6px;">
        Procedural Document Search
        {ai_badge}
      </div>
      <div style="font-size:15px;color:#4b5563;line-height:1.7;">
        Retrieval improved access — but access alone does not fully address
        <strong>interpretive burden</strong>. This layer explains why a result appeared,
        surfaces related procedures, and adds reasoning notes that keep multiple
        interpretations visible. It does not replace reading or make compliance decisions.
      </div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;min-width:280px;">
      <div style="padding:12px 14px;background:#eff6ff;border:1px solid #bfdbfe;
          border-radius:9px;text-align:center;">
        <div style="font-size:11px;font-weight:700;text-transform:uppercase;
            letter-spacing:.07em;color:#1d4ed8;">Explains</div>
        <div style="font-size:12px;color:#3b82f6;margin-top:3px;">why result appeared</div>
      </div>
      <div style="padding:12px 14px;background:#f0fdf4;border:1px solid #86efac;
          border-radius:9px;text-align:center;">
        <div style="font-size:11px;font-weight:700;text-transform:uppercase;
            letter-spacing:.07em;color:#16a34a;">Surfaces</div>
        <div style="font-size:12px;color:#15803d;margin-top:3px;">related procedures</div>
      </div>
      <div style="padding:12px 14px;background:#fefce8;border:1px solid #fde68a;
          border-radius:9px;text-align:center;">
        <div style="font-size:11px;font-weight:700;text-transform:uppercase;
            letter-spacing:.07em;color:#d97706;">Keeps</div>
        <div style="font-size:12px;color:#b45309;margin-top:3px;">reasoning visible</div>
      </div>
    </div>
  </div>
</div>
""")

# =========================================================
# Search input
# =========================================================
render(f'<div style="{CARD}margin-bottom:10px;">'
       f'{overline("Search")}'
       f'{heading("Find procedural documents")}'
       f'{body("Enter a query or use a sample below.")}'
       f'</div>')

query_input = st.text_input(
    "Search",
    value=st.session_state.query,
    placeholder="Try: environmental monitoring sampling / root cause deviation / change control",
    label_visibility="collapsed",
)
if query_input != st.session_state.query:
    st.session_state.query = query_input
    st.session_state.ai_cache = {}  # clear cache on new query

render(f'{slabel("Sample queries", mt="8px")}')
sq1, sq2, sq3 = st.columns(3, gap="small")
with sq1:
    if st.button("environmental monitoring sampling", use_container_width=True):
        st.session_state.query = "environmental monitoring sampling"
        st.session_state.ai_cache = {}
        st.session_state.toast_msg = "🔍 Sample query loaded"
        st.rerun()
with sq2:
    if st.button("root cause analysis deviation", use_container_width=True):
        st.session_state.query = "root cause analysis deviation"
        st.session_state.ai_cache = {}
        st.session_state.toast_msg = "🔍 Sample query loaded"
        st.rerun()
with sq3:
    if st.button("change control revision", use_container_width=True):
        st.session_state.query = "change control revision"
        st.session_state.ai_cache = {}
        st.session_state.toast_msg = "🔍 Sample query loaded"
        st.rerun()

query = st.session_state.query

# =========================================================
# Empty state
# =========================================================
if not query.strip():
    render(f"""
    <div style="{CARD}margin-top:8px;background:#fafafa;border-color:#e5e7eb;">
      {overline("How this works")}
      {heading("Enter a query above to begin")}
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px;">
        <div style="padding:14px 16px;background:#eff6ff;border:1px solid #bfdbfe;border-radius:9px;">
          <div style="font-size:13px;font-weight:700;color:#1d4ed8;margin-bottom:4px;">
            Transparent retrieval
          </div>
          <div style="font-size:13px;color:#3b82f6;line-height:1.55;">
            Every result comes with an explanation of why it appeared.
            {"Claude reads the document and query to explain the connection." if has_api else "Rule-based keyword matching."}
          </div>
        </div>
        <div style="padding:14px 16px;background:#f0fdf4;border:1px solid #86efac;border-radius:9px;">
          <div style="font-size:13px;font-weight:700;color:#16a34a;margin-bottom:4px;">
            Related procedures
          </div>
          <div style="font-size:13px;color:#15803d;line-height:1.55;">
            Adjacent documents are surfaced {"with AI-generated explanations of why they matter." if has_api else "from cross-reference mappings."}
          </div>
        </div>
        <div style="padding:14px 16px;background:#fefce8;border:1px solid #fde68a;border-radius:9px;">
          <div style="font-size:13px;font-weight:700;color:#d97706;margin-bottom:4px;">
            Reasoning support
          </div>
          <div style="font-size:13px;color:#b45309;line-height:1.55;">
            {"Claude generates context-aware notes that keep interpretive openness alive." if has_api else "Rule-based notes surface common ambiguities."}
          </div>
        </div>
        <div style="padding:14px 16px;background:#f9fafb;border:1px solid #e5e7eb;border-radius:9px;">
          <div style="font-size:13px;font-weight:700;color:#374151;margin-bottom:4px;">
            Empty results as signal
          </div>
          <div style="font-size:13px;color:#6b7280;line-height:1.55;">
            When nothing is found, that is part of the document usability problem —
            not an individual failure.
          </div>
        </div>
      </div>
    </div>
    """)

# =========================================================
# Results
# =========================================================
else:
    results = search_docs(query, df)

    if results.empty:
        render(f"""
        <div style="background:#fef2f2;border:1px solid #fca5a5;border-left:4px solid #dc2626;
            border-radius:12px;padding:20px 24px;margin-top:8px;">
          {overline("No results found")}
          {heading("Nothing matched — but this is informative")}
          {body("Try broader terms, adjacent workflow words, or a related document name.<br><br>"
                "<strong>Note:</strong> This empty-result state is itself part of the document "
                "usability problem. When workers cannot find what they need, that retrieval gap "
                "is a system-level issue, not an individual failure.")}
        </div>
        """)
    else:
        render(f"""
        <div style="font-size:13px;font-weight:700;text-transform:uppercase;
            letter-spacing:.07em;color:#6b7280;margin:10px 0 10px;">
          {len(results)} result{"s" if len(results) > 1 else ""} for
          <span style="color:#2563eb;">"{query}"</span>
          {"<span style='color:#7c3aed;margin-left:8px;font-size:12px;'>✨ AI support active</span>" if has_api else ""}
        </div>
        """)

        for idx, row in results.iterrows():
            label, score_color, score_bg = score_label(row["score"])
            related_docs = get_related_titles(row, df)

            # AI support — cached per (query, doc_id), errors not cached
            cache_key = f"{query}|{row['doc_id']}"
            ai_data = st.session_state.ai_cache.get(cache_key, None)

            if has_api and ai_data is None:
                with st.spinner(f"AI analysing {row['doc_id']}..."):
                    ai_data = run_claude_support(
                        query=query,
                        doc=row.to_dict(),
                        related_docs=related_docs,
                    )
                    # FIX 2: only cache successful results so errors can be retried
                    if not ai_data.get("error"):
                        st.session_state.ai_cache[cache_key] = ai_data

            # matched terms
            all_tokens = tokenize(
                row["title"] + " " + row["section"] + " " +
                row["text"] + " " + " ".join(row["keywords"])
            )
            matched_terms = [t for t in row["query_terms"] if t in all_tokens]

            # ── Result card ───────────────────────────────
            render(f"""
            <div style="{CARD}border-left:4px solid {score_color};">
              <div style="display:flex;align-items:flex-start;justify-content:space-between;
                  flex-wrap:wrap;gap:12px;margin-bottom:14px;">
                <div>
                  {overline(row['doc_id'])}
                  <div style="font-size:18px;font-weight:700;color:#111827;margin-bottom:4px;">
                    {row['title']}
                  </div>
                  <div style="font-size:13px;color:#6b7280;">
                    {row['section']} · page {row['page']}
                  </div>
                </div>
                <div style="text-align:center;padding:10px 18px;background:{score_bg};
                    border:1px solid {score_color}33;border-radius:9px;flex-shrink:0;">
                  <div style="font-size:20px;font-weight:700;color:{score_color};">{label}</div>
                  <div style="font-size:11px;font-weight:600;color:{score_color};
                      text-transform:uppercase;letter-spacing:.06em;">Match strength</div>
                </div>
              </div>
              <div style="font-size:14px;color:#374151;line-height:1.7;padding:14px 16px;
                  background:#f9fafb;border-radius:8px;margin-bottom:12px;">
                {highlight_text(row['text'], row['query_terms'])}
              </div>
              <div style="display:flex;flex-wrap:wrap;gap:6px;">
                {"".join(
                    f'<span style="display:inline-block;padding:3px 9px;background:#eff6ff;'
                    f'color:#1d4ed8;border:1px solid #bfdbfe;border-radius:5px;'
                    f'font-size:12px;font-weight:600;">{kw}</span>'
                    for kw in row['keywords']
                )}
              </div>
            </div>
            """)

            # ── Support layer expanders ───────────────────
            exp1, exp2, exp3 = st.columns(3, gap="small")

            # Why this result? — first result expanded by default for discoverability
            is_first = (idx == 0)
            with exp1:
                with st.expander("🔍 Why this result?", expanded=is_first):
                    if has_api and ai_data and not ai_data.get("error"):
                        render(f"""
                        <div style="padding:12px 14px;background:#f5f3ff;
                            border:1px solid #ddd6fe;border-radius:8px;margin-bottom:10px;">
                          <div style="font-size:11px;font-weight:700;text-transform:uppercase;
                              letter-spacing:.07em;color:#7c3aed;margin-bottom:6px;">
                            ✨ AI explanation
                          </div>
                          <div style="font-size:14px;color:#374151;line-height:1.65;">
                            {ai_data.get("why", "")}
                          </div>
                        </div>
                        """)
                    else:
                        render(f"""
                        <div style="font-size:14px;color:#374151;line-height:1.65;
                            margin-bottom:12px;">{fallback_why(row)}</div>
                        """)

                    render(slabel("Matched terms", mt="0"))
                    if matched_terms:
                        terms_html = "".join(
                            f'<span style="display:inline-block;padding:3px 8px;'
                            f'background:#fef08a;color:#111827;border-radius:4px;'
                            f'font-size:12px;font-weight:600;margin:2px 3px 2px 0;">{t}</span>'
                            for t in matched_terms
                        )
                        render(f'<div style="margin-bottom:8px;">{terms_html}</div>')
                    else:
                        render('<div style="font-size:13px;color:#9ca3af;">contextual overlap</div>')

                    render(slabel("Score breakdown"))
                    breakdown = pd.DataFrame({
                        "Source":       ["Keyword index", "Title", "Section", "Body text"],
                        "Weight":       ["×5", "×4", "×3", "×1"],
                        "Contribution": [
                            row["contribution"]["keywords"],
                            row["contribution"]["title"],
                            row["contribution"]["section"],
                            row["contribution"]["text"],
                        ],
                    })
                    st.dataframe(breakdown, use_container_width=True, hide_index=True)

            # Related procedures — first result expanded by default
            with exp2:
                with st.expander("📎 Related procedures", expanded=is_first):
                    if related_docs:
                        # build AI notes lookup
                        ai_related = {}
                        if has_api and ai_data and not ai_data.get("error"):
                            for rn in ai_data.get("related_notes", []):
                                ai_related[rn.get("doc_id", "")] = rn.get("why_relevant", "")

                        for rel in related_docs:
                            note = ai_related.get(rel["doc_id"]) or fallback_related_note(rel["title"])
                            ai_tag = "✨ " if rel["doc_id"] in ai_related else ""
                            render(f"""
                            <div style="padding:10px 12px;background:#f9fafb;
                                border:1px solid #e5e7eb;border-radius:8px;margin-bottom:6px;">
                              <div style="font-size:13px;font-weight:700;color:#111827;
                                  margin-bottom:3px;">
                                {ai_tag}{rel['doc_id']} — {rel['title']}
                              </div>
                              <div style="font-size:12px;color:#6b7280;margin-bottom:3px;">
                                {rel['section']}
                              </div>
                              <div style="font-size:13px;color:#374151;line-height:1.55;">
                                {note}
                              </div>
                            </div>
                            """)
                    else:
                        render('<div style="font-size:13px;color:#9ca3af;">No related procedures mapped.</div>')

            # Reasoning support note — first result expanded by default
            with exp3:
                with st.expander("💡 Reasoning support note", expanded=is_first):
                    if has_api and ai_data and not ai_data.get("error"):
                        render(f"""
                        <div style="padding:12px 14px;background:#f5f3ff;
                            border:1px solid #ddd6fe;border-radius:8px;margin-bottom:10px;">
                          <div style="font-size:11px;font-weight:700;text-transform:uppercase;
                              letter-spacing:.07em;color:#7c3aed;margin-bottom:6px;">
                            ✨ AI reasoning note
                          </div>
                          <div style="font-size:14px;color:#374151;line-height:1.65;">
                            {ai_data.get("reasoning_note", "")}
                          </div>
                        </div>
                        """)
                    else:
                        render(f"""
                        <div style="padding:12px 14px;background:#fffbeb;
                            border:1px solid #fde68a;border-radius:8px;margin-bottom:10px;">
                          <div style="font-size:11px;font-weight:700;text-transform:uppercase;
                              letter-spacing:.07em;color:#d97706;margin-bottom:6px;">Note</div>
                          <div style="font-size:14px;color:#374151;line-height:1.65;">
                            {fallback_reasoning_note(query)}
                          </div>
                        </div>
                        """)

                    render(f"""
                    <div style="font-size:12px;color:#9ca3af;line-height:1.55;">
                      This note keeps alternative interpretations visible.
                      It does not provide a final compliance judgment.
                    </div>
                    """)

            render('<div style="height:4px;"></div>')

        # ── Design note ───────────────────────────────────
        render(f"""
        <div style="{CARD}margin-top:8px;background:#f8fafc;border-color:#e2e8f0;">
          {overline("Design note")}
          {heading("Why procedural retrieval is not just search")}
          {body(
              "The original interface improved access — but retrieval alone does not fully address "
              "interpretive burden. In document-heavy systems, people often still need to understand "
              "why a result appeared, what adjacent procedures may matter, and where ambiguity may remain.<br><br>"
              "This support layer was designed to make those reasoning steps more legible, "
              "rather than leaving them implicit. It is support, not autopilot — it does not output "
              "a final compliance answer. It expands context, surfaces connections, and helps users "
              "keep interpretive openness alive while moving through procedural archives."
          )}
        </div>
        """)
