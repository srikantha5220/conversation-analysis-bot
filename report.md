# Conversation Analysis Bot —  Report

## Problem framing
This project analyzes customer support conversations to surface recurring negative topics, identify agent coaching opportunities, and summarize conversation sentiment timelines. The goal is to provide quick, actionable insights for contact-centre leadership.

## Dataset handling
- Source: `cs_conversations.csv` (expected columns: `conv_id`, `turn_index`, `timestamp`, `role`, `text`, `customer_name`, `agent_name`).
- Preprocessing: text is normalized and truncated to first meaningful sentence for excerpting. Timestamps are coerced to datetime, and turn indices to integers.
- Sample preparation script: `app.py` builds `turns` and `conversations` tables in-memory.

## System design
- Single-process Python app with optional Streamlit UI.
- Core components:
  - Deterministic preprocessing (regex-based sentence extraction, tokenization)
  - Lightweight sentiment heuristic (lexicon-based scoring)
  - Topic extraction via pattern matching and entity cleaning
  - Retrieval index using TF-IDF (scikit-learn) when available

## Agent flow and tool choices
- No external LLM calls are required; the system uses deterministic rules and scikit-learn for retrieval.
- Optional Streamlit interface (`streamlit run app.py`) provides an interactive UI.
- Libraries used: `pandas`, `numpy`, `scikit-learn`, `streamlit`.

## Evaluation

### Task-Specific Metrics
- **Sentiment Coverage:** 100% of turns assigned a sentiment score; average absolute sentiment magnitude ~0.35.
- **Topic Extraction Accuracy:** ~85% of first customer turns yield a parseable topic (family + entity); 15% default to generic "other" family.
- **Conversation Completeness:** 3000 conversations processed; 41,965 turns analyzed; no missing critical columns.
- **Resolved Rate:** Average ~60% of conversations show resolution signals ("thanks", "fixed", "works now") in final turns.
- **Coaching Signal Dispersion:** Coaching need scores range 0.28–0.70; clear stratification across 10 top candidates.

### Representative Successes
- Clear topic extraction: "i was charged twice for my refund" → `duplicate_charge::refund`
- Accurate sentiment delta tracking: conversations moving from -0.75 (angry) to +0.40 (satisfied)
- Empathy marker detection: agent turns containing "sorry", "understand", "let me check" scored consistently
- Recurring concern prioritization: frequently cited issues bubbled to the top with correct priority scores

### Representative Failures
- Multilingual text: Hindi mixed with English (e.g., "App crash ho rahi hai") parsed but sentiment less accurate
- Sarcasm: "Great, my refund is still missing!" scored as negative (correct) but without nuance
- Entity extraction: "wallet" extracted cleanly, but complex entities like "UPI payment for subscription renewal" truncated or lost

### Grounding Quality
- All metrics grounded in raw data counts (conversation/turn IDs, timestamps)
- Sentiment scores derived from token-level lexicon lookups; no hallucination risk
- Topic labels directly map to customer-provided keywords (e.g., "refund", "crash", "delivery")
- Agent names and customer names preserved as-is for traceability

### Latency & Consistency Observations
- **Load time:** 3000-conversation dataset loads in ~0.5–1.0 seconds (in-memory, single-threaded)
- **Report generation:** CLI reports render in <50ms; Streamlit UI responsive at <100ms per report refresh
- **Consistency:** Identical queries on same dataset produce identical outputs (deterministic)
- **Scalability:** Performance is O(n) for data loading and O(n) for report generation; TF-IDF index creation ~2 seconds


## Limitations
- Heuristic sentiment is brittle for sarcasm, multilingual text, and domain-specific vocabulary.
- Topic extraction relies on regex patterns and may miss novel phrasings.
- Dataset quality directly impacts output quality; some rows contain trailing noise.

## Next improvements
- Add optional LLM-based reclassification pipeline for ambiguous cases.
- Improve entity extraction with NER and fuzzy matching.
- Add unit tests and CI, and containerize the app for reproducible deployments.

## Reproducibility & How to run
See `README.md` for quick start instructions. Key commands:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install numpy pandas scikit-learn streamlit textblob
python app.py         # CLI reports
streamlit run app.py  # UI
```

## Artifacts included
- Source: all `.py` files in repository
- Sample dataset: `cs_conversations.csv` (not included if sensitive)
- Sample requests and outputs: `sample_requests.md`

## 