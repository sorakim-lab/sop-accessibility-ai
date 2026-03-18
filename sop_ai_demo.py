import re
from collections import Counter
from typing import List, Dict, Tuple

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="SOP Accessibility + AI Support Layer",
    page_icon="📘",
    layout="wide"
)


# -----------------------------
# Demo data
# -----------------------------
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
        "related_docs": ["SOP-004", "SOP-006"]
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
        "related_docs": ["SOP-003", "SOP-005"]
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
        "related_docs": ["SOP-002", "SOP-005"]
    },
    {
        "doc_id": "SOP-004",
        "title": "Aseptic Gowning and Entry Procedure",
        "section": "Section 2.3 Entry Requirements",
        "page": 7,
        "keywords": ["gowning", "entry", "aseptic", "room entry", "operator"],
        "text": (
            "This procedure covers aseptic gowning steps, entry order, and operator behavior before entering controlled rooms. "
            "Users should review this procedure before any environmental monitoring activity."
        ),
        "related_docs": ["SOP-001"]
    },
    {
        "doc_id": "SOP-005",
        "title": "Change Control and Procedure Revision",
        "section": "Section 6.1 Revision Triggers",
        "page": 22,
        "keywords": ["change control", "revision", "procedure revision", "document update"],
        "text": (
            "This procedure explains when a controlled procedure should be revised, how change control is initiated, "
            "and how affected records and downstream processes should be identified."
        ),
        "related_docs": ["SOP-002", "SOP-003"]
    },
    {
        "doc_id": "SOP-006",
        "title": "Plate Handling and Incubation Conditions",
        "section": "Section 4.1 Plate Transfer",
        "page": 10,
        "keywords": ["plate", "incubation", "handling", "transfer", "environmental monitoring"],
        "text": (
            "This procedure describes how contact plates and settle plates should be handled, transported, and incubated "
            "after environmental monitoring collection."
        ),
        "related_docs": ["SOP-001"]
    },
]

df = pd.DataFrame(DATA)


# -----------------------------
# Helpers
# -----------------------------
STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "by",
    "is", "are", "be", "this", "that", "how", "what", "when", "should", "must",
    "after", "before", "into", "from", "through", "about"
}


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z0-9\-]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def highlight_text(text: str, query_terms: List[str]) -> str:
    highlighted = text
    for term in sorted(set(query_terms), key=len, reverse=True):
        if len(term) < 2:
            continue
        pattern = re.compile(rf"({re.escape(term)})", re.IGNORECASE)
        highlighted = pattern.sub(r"<mark>\1</mark>", highlighted)
    return highlighted


def score_row(row: pd.Series, query_terms: List[str]) -> Tuple[int, Dict[str, int]]:
    title_tokens = tokenize(row["title"])
    section_tokens = tokenize(row["section"])
    keyword_tokens = [k.lower() for k in row["keywords"]]
    text_tokens = tokenize(row["text"])

    contribution = {"title": 0, "section": 0, "keywords": 0, "text": 0}

    for term in query_terms:
        contribution["title"] += title_tokens.count(term) * 4
        contribution["section"] += section_tokens.count(term) * 3
        contribution["keywords"] += keyword_tokens.count(term) * 5
        contribution["text"] += text_tokens.count(term) * 1

    total = sum(contribution.values())
    return total, contribution


def search_docs(query: str, data: pd.DataFrame) -> pd.DataFrame:
    query_terms = tokenize(query)
    rows = []

    for _, row in data.iterrows():
        total, contribution = score_row(row, query_terms)
        if total > 0:
            row_dict = row.to_dict()
            row_dict["score"] = total
            row_dict["contribution"] = contribution
            row_dict["query_terms"] = query_terms
            rows.append(row_dict)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).sort_values(by="score", ascending=False).reset_index(drop=True)
    return result


def make_reason_why(row: pd.Series) -> str:
    parts = []
    contribution = row["contribution"]

    if contribution["keywords"] > 0:
        parts.append("matched indexed procedural keywords")
    if contribution["title"] > 0:
        parts.append("matched document title terms")
    if contribution["section"] > 0:
        parts.append("matched section-level cues")
    if contribution["text"] > 0:
        parts.append("matched supporting body text")

    if not parts:
        return "This result was retrieved by weak contextual overlap."
    return "This result was retrieved because it " + ", ".join(parts) + "."


def make_reasoning_support_note(row: pd.Series, query: str) -> str:
    query_lower = query.lower()

    if "deviation" in query_lower or "rca" in query_lower or "root cause" in query_lower:
        return (
            "Reasoning support note: before narrowing the issue to operator error, review whether the event may "
            "also involve procedural ambiguity, missing cross-references, or workflow constraints."
        )
    if "sampling" in query_lower or "environmental monitoring" in query_lower:
        return (
            "Reasoning support note: this task may require checking adjacent procedures such as entry, gowning, "
            "or plate handling before execution. Retrieval difficulty itself may be part of the work burden."
        )
    if "change control" in query_lower or "revision" in query_lower:
        return (
            "Reasoning support note: consider not only the revision trigger itself, but also which downstream "
            "documents and workflows become dependent on this update."
        )

    return (
        "Reasoning support note: this result may be relevant, but users should still verify neighboring procedures, "
        "section context, and downstream dependencies before acting."
    )


def get_related_titles(row: pd.Series, data: pd.DataFrame) -> List[str]:
    related = []
    for doc_id in row["related_docs"]:
        matched = data[data["doc_id"] == doc_id]
        if not matched.empty:
            related.append(f"{doc_id} — {matched.iloc[0]['title']}")
    return related


def explain_score_breakdown(contribution: Dict[str, int]) -> pd.DataFrame:
    breakdown = pd.DataFrame(
        {
            "Source": ["Keyword index", "Title", "Section", "Body text"],
            "Contribution": [
                contribution["keywords"],
                contribution["title"],
                contribution["section"],
                contribution["text"],
            ],
        }
    )
    return breakdown


# -----------------------------
# UI
# -----------------------------
st.title("📘 SOP Accessibility + Support Layer")
st.caption(
    "Prototype concept: procedural retrieval with explanation, related-document support, "
    "and reasoning-aware assistance. This is not autopilot. It is a visibility layer."
)

with st.sidebar:
    st.subheader("About this prototype")
    st.write(
        "This demo shows a lightweight support layer for regulated document retrieval. "
        "It explains why a result appeared, suggests related procedures, and provides "
        "a reasoning support note without replacing human judgment."
    )
    st.info(
        "Best admission framing: AI-mediated support for navigation and procedural reasoning, "
        "not automated compliance decision-making."
    )

query = st.text_input(
    "Search procedural documents",
    placeholder="Try: environmental monitoring sampling / RCA / change control revision"
)

sample_queries = st.columns(3)
with sample_queries[0]:
    if st.button("environmental monitoring sampling", use_container_width=True):
        query = "environmental monitoring sampling"
with sample_queries[1]:
    if st.button("root cause analysis deviation", use_container_width=True):
        query = "root cause analysis deviation"
with sample_queries[2]:
    if st.button("change control revision", use_container_width=True):
        query = "change control revision"

if query:
    results = search_docs(query, df)

    if results.empty:
        st.warning("No matching procedural results were found.")
        st.write(
            "Try broader terms, adjacent workflow words, or a related document name. "
            "This empty-result state is itself part of the document usability problem."
        )
    else:
        st.success(f"{len(results)} result(s) found")

        for idx, row in results.iterrows():
            with st.container(border=True):
                col1, col2 = st.columns([3.2, 1.3])

                with col1:
                    st.subheader(f"{row['doc_id']} — {row['title']}")
                    st.write(f"{row['section']} · page {row['page']}")

                    snippet = highlight_text(row["text"], row["query_terms"])
                    st.markdown(snippet, unsafe_allow_html=True)

                    with st.expander("Why this result?"):
                        st.write(make_reason_why(row))

                        matched_terms = [t for t in row["query_terms"] if t in tokenize(row["title"] + " " + row["section"] + " " + row["text"] + " " + " ".join(row["keywords"]))]
                        st.write("Matched query terms:")
                        st.code(", ".join(matched_terms) if matched_terms else "contextual overlap")

                        breakdown_df = explain_score_breakdown(row["contribution"])
                        st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

                    with st.expander("Related procedures"):
                        related = get_related_titles(row, df)
                        if related:
                            for item in related:
                                st.write(f"- {item}")
                        else:
                            st.write("No related procedures were mapped for this entry.")

                    with st.expander("Reasoning support note"):
                        st.write(make_reasoning_support_note(row, query))
                        st.caption(
                            "This note is designed to keep alternative interpretations visible. "
                            "It does not provide a final compliance judgment."
                        )

                with col2:
                    st.metric("Retrieval score", row["score"])
                    st.write("Indexed keywords")
                    st.caption(", ".join(row["keywords"]))

                    st.write("Support layer functions")
                    st.caption(
                        "• explains retrieval\n"
                        "• surfaces related docs\n"
                        "• keeps reasoning visible"
                    )

        st.divider()
        st.subheader("Why this matters")
        st.write(
            "This prototype reframes procedural search as more than retrieval. "
            "It makes document relevance visible, surfaces neighboring procedures, "
            "and reduces the chance that hidden documentation burden gets treated as an individual failure."
        )
else:
    st.info("Enter a query above to test the support layer.")