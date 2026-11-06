import argparse
import math
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except Exception:
    st = None
    STREAMLIT_AVAILABLE = False


DEFAULT_CSV_PATH = Path(__file__).resolve().parent / "cs_conversations.csv"

NEG_WORDS = {
    "bad", "worst", "angry", "frustrated", "issue", "problem", "broken", "fail", "failed",
    "failing", "error", "not", "can't", "cannot", "wont", "won't", "didn't", "didnt",
    "never", "nothing", "no", "late", "delay", "delayed", "urgent", "escalate",
    "escalated", "complaint", "trouble", "terrible", "unhappy", "sad", "confused", "cancel",
    "cancelled", "missing", "lost", "charged", "crash", "crashing", "stuck", "unable",
    "blocked", "refund", "upset", "slow", "failure",
}

POS_WORDS = {
    "thanks", "thank", "great", "good", "appreciate", "resolved", "fix", "fixed", "working",
    "works", "clear", "understand", "happy", "helpful", "awesome", "perfect", "excellent",
    "done", "completed", "confirmed", "success", "successful", "glad", "solved",
}

EMPATHY_MARKERS = [
    "sorry",
    "i understand",
    "understand",
    "i'm sorry",
    "let me check",
    "investigating",
    "please allow",
    "i can help",
    "i will",
    "i'll",
    "thanks for reaching out",
    "thanks for your patience",
    "appreciate your patience",
    "let me look",
    "let me see",
    "may i have",
    "please",
    "happy to help",
]

CUSTOMER_PATTERNS = [
    (r"(?:hello,?\s*)?my (?P<entity>.+?) is not working as expected", "service_not_working"),
    (r"i was charged twice for my (?P<entity>.+)", "duplicate_charge"),
    (r"coupon isn.?t applying on checkout for the (?P<entity>.+)", "coupon_checkout"),
    (r"coupon is not applying on checkout for the (?P<entity>.+)", "coupon_checkout"),
    (r"the (?P<entity>.+?) hasn.?t arrived yet", "delivery_delay"),
    (r"the (?P<entity>.+?) has not arrived yet", "delivery_delay"),
    (r"mujhe (?P<entity>.+?) ke baare mein help chahiye", "general_help"),
    (r"hi,? i need help with my (?P<entity>.+)", "general_help"),
    (r"i can't log in", "login_access"),
    (r"i cannot log in", "login_access"),
    (r"upi shows successful but i didn't get the service for (?P<entity>.+)", "payment_success_no_service"),
    (r"app crash ho rahi hai while using (?P<entity>.+)", "app_crash"),
    (r"hello,? my (?P<entity>.+?) is not working as expected", "service_not_working"),
    (r"my (?P<entity>.+?) is not working as expected", "service_not_working"),
    (r"i need help with my (?P<entity>.+)", "general_help"),
]

GENERIC_ENTITY_STOPWORDS = {
    "my", "the", "for", "with", "on", "of", "and", "is", "are", "as", "expected",
    "while", "using", "help", "need", "chahiye", "ke", "baare", "mein", "please", "yet",
    "now", "account", "service",
}


def normalize_text(text: object) -> str:
    value = str(text or "")
    value = value.replace("'", "'").replace("'", "'").replace('"', '"').replace('"', '"')
    value = value.lower()
    return re.sub(r"\s+", " ", value).strip()


def tokenize(text: object) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", normalize_text(text))


def first_sentence(text: object) -> str:
    value = normalize_text(text)
    parts = re.split(r"(?<=[.!?])\s+", value, maxsplit=1)
    return parts[0].strip() if parts else value


def safe_excerpt(text: object, max_len: int = 120) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(value) <= max_len:
        return value
    return value[: max_len - 1].rstrip() + "…"


def sentiment_score(text: object) -> float:
    tokens = tokenize(text)
    score = 0.0
    negation_window = 0
    negations = {
        "not", "no", "never", "cannot", "can't", "cant", "won't", "wont",
        "didn't", "didnt", "isn't", "isnt", "aren't", "arent", "wasn't", "wasnt"
    }

    for token in tokens:
        if token in negations:
            negation_window = 3
            continue

        delta = 0
        if token in NEG_WORDS:
            delta = -1
        elif token in POS_WORDS:
            delta = 1

        if delta:
            if negation_window > 0:
                delta *= -1
            score += delta

        if negation_window > 0:
            negation_window -= 1

    return score / max(1.0, math.sqrt(len(tokens)))


def sentiment_label(score: float) -> str:
    if score <= -0.2:
        return "negative"
    if score >= 0.2:
        return "positive"
    return "neutral"


def empathy_score(text: object) -> int:
    value = normalize_text(text)
    return sum(1 for marker in EMPATHY_MARKERS if marker in value)


def clean_entity(entity: Optional[str]) -> Optional[str]:
    if not entity:
        return None

    value = normalize_text(entity)
    value = re.sub(r"[^a-z0-9\s]+", " ", value)
    tokens = [tok for tok in value.split() if tok not in GENERIC_ENTITY_STOPWORDS]

    while tokens and tokens[-1] in {"expected", "yet", "now", "working", "arrived", "applying", "successful", "please"}:
        tokens.pop()

    cleaned = " ".join(tokens).strip()
    return cleaned[:40] if cleaned else None


def extract_issue_family_and_entity(text: object) -> tuple[str, Optional[str]]:
    sample = first_sentence(text)
    for pattern, family in CUSTOMER_PATTERNS:
        match = re.search(pattern, sample)
        if match:
            return family, clean_entity(match.groupdict().get("entity"))

    if "refund" in sample:
        return "refund", None
    if "billing" in sample or "charge" in sample:
        return "billing", None
    if "login" in sample or "log in" in sample or "sso" in sample:
        return "login_access", None
    if "coupon" in sample or "promo" in sample or "voucher" in sample or "discount" in sample:
        return "promo_coupon", None
    if "crash" in sample or "freez" in sample or "hang" in sample:
        return "app_crash", None
    if "arrived" in sample or "delivery" in sample:
        return "delivery_delay", None
    return "other", None


def topic_label(text: object) -> str:
    family, entity = extract_issue_family_and_entity(text)
    return f"{family}::{entity}" if entity else family


class ConversationAnalysisBot:
    def __init__(self, csv_path: str) -> None:
        self.csv_path = csv_path
        self.raw = self.load_csv(csv_path)
        self.turns = self.prepare_turns(self.raw)
        self.conversations = self.build_conversation_table(self.raw)
        self.build_retrieval_index()

    @staticmethod
    def load_csv(path: str) -> pd.DataFrame:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")

        df = pd.read_csv(file_path)
        required = {"conv_id", "turn_index", "timestamp", "role", "text", "customer_name", "agent_name"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["turn_index"] = pd.to_numeric(df["turn_index"], errors="coerce").fillna(0).astype(int)
        return df.sort_values(["conv_id", "turn_index"]).reset_index(drop=True)

    @staticmethod
    def prepare_turns(df: pd.DataFrame) -> pd.DataFrame:
        turns = df.copy()
        turns["clean_text"] = turns["text"].map(first_sentence)
        turns["sentiment_score"] = turns["clean_text"].map(sentiment_score)
        turns["sentiment_label"] = turns["sentiment_score"].map(sentiment_label)
        turns["empathy_score"] = turns["clean_text"].map(empathy_score)
        issue_entity = turns["clean_text"].map(extract_issue_family_and_entity)
        turns[["issue_family", "entity"]] = pd.DataFrame(issue_entity.tolist(), index=turns.index)
        turns["topic"] = turns["clean_text"].map(topic_label)
        return turns

    @staticmethod
    def build_conversation_table(df: pd.DataFrame) -> pd.DataFrame:
        rows = []

        for conv_id, group in df.groupby("conv_id", sort=False):
            group = group.sort_values("turn_index")
            customer_turns = group[group["role"] == "customer"]
            agent_turns = group[group["role"] == "agent"]
            timestamps = group["timestamp"].dropna()

            first_customer = customer_turns["text"].iloc[0] if not customer_turns.empty else ""
            last_customer = customer_turns["text"].iloc[-1] if not customer_turns.empty else ""
            first_clean = first_sentence(first_customer)
            last_clean = first_sentence(last_customer)
            issue_family, entity = extract_issue_family_and_entity(first_clean)
            topic = f"{issue_family}::{entity}" if entity else issue_family

            customer_scores = customer_turns["text"].map(lambda text: sentiment_score(first_sentence(text))).tolist()
            agent_scores = agent_turns["text"].map(lambda text: sentiment_score(first_sentence(text))).tolist()
            agent_empathy = agent_turns["text"].map(lambda text: empathy_score(first_sentence(text))).tolist()

            resolved_like = any(
                any(marker in normalize_text(text) for marker in ["thanks", "thank you", "works now", "resolved", "fixed", "that helps"])
                for text in customer_turns["text"].tail(2).tolist()
            )

            start_ts = timestamps.min() if not timestamps.empty else pd.NaT
            end_ts = timestamps.max() if not timestamps.empty else pd.NaT
            duration_min = (
                (end_ts - start_ts).total_seconds() / 60
                if pd.notna(start_ts) and pd.notna(end_ts)
                else np.nan
            )

            rows.append(
                {
                    "conv_id": conv_id,
                    "customer_name": group["customer_name"].iloc[0],
                    "agent_name": group["agent_name"].iloc[0],
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                    "turn_count": len(group),
                    "customer_turns": len(customer_turns),
                    "agent_turns": len(agent_turns),
                    "duration_min": duration_min,
                    "first_customer_text": first_customer,
                    "last_customer_text": last_customer,
                    "issue_family": issue_family,
                    "entity": entity,
                    "topic": topic,
                    "customer_start_sentiment": customer_scores[0] if customer_scores else 0.0,
                    "customer_end_sentiment": customer_scores[-1] if customer_scores else 0.0,
                    "customer_sentiment_delta": (customer_scores[-1] - customer_scores[0]) if len(customer_scores) >= 2 else 0.0,
                    "negative_customer_turns": sum(1 for score in customer_scores if score < -0.2),
                    "positive_customer_turns": sum(1 for score in customer_scores if score > 0.2),
                    "agent_empathy_total": sum(agent_empathy),
                    "agent_empathy_per_turn": float(np.mean(agent_empathy)) if agent_empathy else 0.0,
                    "agent_positive_total": sum(1 for score in agent_scores if score > 0.2),
                    "agent_negative_total": sum(1 for score in agent_scores if score < -0.2),
                    "resolved_like": bool(resolved_like),
                }
            )

        conversations = pd.DataFrame(rows)
        conversations["start_ts"] = pd.to_datetime(conversations["start_ts"], errors="coerce")
        conversations["end_ts"] = pd.to_datetime(conversations["end_ts"], errors="coerce")
        return conversations.sort_values("start_ts").reset_index(drop=True)

    def build_retrieval_index(self) -> None:
        docs = [
            f"{row.topic} {row.first_customer_text} {row.last_customer_text}"
            for row in self.conversations.itertuples(index=False)
        ]
        self._retrieval_docs = docs
        self._vectorizer = None
        self._tfidf = None

        if SKLEARN_AVAILABLE:
            self._vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_features=8000,
                min_df=1,
            )
            self._tfidf = self._vectorizer.fit_transform(docs)

    def recent_window(self, days: int = 30) -> pd.DataFrame:
        latest = self.conversations["start_ts"].max()
        if pd.isna(latest):
            return self.conversations.copy()
        cutoff = latest - pd.Timedelta(days=days)
        return self.conversations[self.conversations["start_ts"] >= cutoff].copy()

    def top_negative_topics(self, top_n: int = 5, days: int = 30) -> pd.DataFrame:
        recent = self.recent_window(days)
        grouped = (
            recent.groupby("topic")
            .agg(
                conversations=("conv_id", "count"),
                avg_start_sentiment=("customer_start_sentiment", "mean"),
                avg_end_sentiment=("customer_end_sentiment", "mean"),
                avg_delta=("customer_sentiment_delta", "mean"),
                neg_rate=("customer_start_sentiment", lambda s: (s < -0.2).mean()),
            )
            .sort_values(["avg_start_sentiment", "conversations"], ascending=[True, False])
        )
        return grouped.head(top_n).reset_index()

    def recurring_concerns(self, top_n: int = 10, days: int = 30) -> pd.DataFrame:
        recent = self.recent_window(days)
        grouped = (
            recent.groupby("topic")
            .agg(
                conversations=("conv_id", "count"),
                neg_rate=("customer_start_sentiment", lambda s: (s < -0.2).mean()),
                avg_start_sentiment=("customer_start_sentiment", "mean"),
                avg_delta=("customer_sentiment_delta", "mean"),
                resolved_rate=("resolved_like", "mean"),
            )
            .reset_index()
        )
        grouped["priority_score"] = grouped["conversations"] * (1.0 - grouped["avg_start_sentiment"].clip(-1, 1))
        return grouped.sort_values(["priority_score", "conversations"], ascending=[False, False]).head(top_n)

    def coaching_candidates(self, top_n: int = 10, min_conversations: int = 2) -> pd.DataFrame:
        agent_stats = (
            self.conversations.groupby("agent_name")
            .agg(
                conversations=("conv_id", "count"),
                avg_empathy=("agent_empathy_per_turn", "mean"),
                avg_customer_delta=("customer_sentiment_delta", "mean"),
                negative_end_rate=("customer_end_sentiment", lambda s: (s < -0.2).mean()),
                resolved_rate=("resolved_like", "mean"),
                avg_duration=("duration_min", "mean"),
                avg_customer_start=("customer_start_sentiment", "mean"),
            )
            .reset_index()
        )

        filtered = agent_stats[agent_stats["conversations"] >= min_conversations].copy()
        if filtered.empty:
            filtered = agent_stats.copy()

        filtered["coaching_need_score"] = (
            (1.0 - filtered["avg_empathy"].clip(0, 5) / 5.0) * 0.35
            + filtered["negative_end_rate"].clip(0, 1) * 0.35
            + (1.0 - filtered["resolved_rate"].clip(0, 1)) * 0.20
            + (-filtered["avg_customer_delta"]).clip(0, 1) * 0.10
        )
        return filtered.sort_values(["coaching_need_score", "conversations"], ascending=[False, False]).head(top_n)

    def sentiment_timeline(self, conv_id: str) -> pd.DataFrame:
        conversation_turns = self.turns[self.turns["conv_id"] == conv_id].sort_values("turn_index").copy()
        if conversation_turns.empty:
            raise KeyError(f"Conversation not found: {conv_id}")

        timeline = conversation_turns[
            ["conv_id", "turn_index", "role", "clean_text", "sentiment_score", "sentiment_label", "empathy_score"]
        ].copy()
        timeline["text_excerpt"] = timeline["clean_text"].map(lambda text: safe_excerpt(text, 140))
        return timeline

    def conversation_summary(self, conv_id: str) -> dict[str, object]:
        row = self.conversations[self.conversations["conv_id"] == conv_id]
        if row.empty:
            raise KeyError(f"Conversation not found: {conv_id}")
        record = row.iloc[0]
        return {
            "conv_id": record["conv_id"],
            "agent_name": record["agent_name"],
            "customer_name": record["customer_name"],
            "topic": record["topic"],
            "issue_family": record["issue_family"],
            "entity": record["entity"],
            "start_ts": str(record["start_ts"]),
            "end_ts": str(record["end_ts"]),
            "turn_count": int(record["turn_count"]),
            "customer_start_sentiment": float(record["customer_start_sentiment"]),
            "customer_end_sentiment": float(record["customer_end_sentiment"]),
            "customer_sentiment_delta": float(record["customer_sentiment_delta"]),
            "resolved_like": bool(record["resolved_like"]),
        }

    def similar_cases(self, query: str, top_k: int = 5) -> pd.DataFrame:
        if not SKLEARN_AVAILABLE or self._vectorizer is None or self._tfidf is None:
            raise RuntimeError("scikit-learn is required for similarity search.")

        query_vector = self._vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self._tfidf).ravel()
        top_idx = np.argsort(-similarities)[:top_k]

        matches = self.conversations.iloc[top_idx][
            [
                "conv_id",
                "topic",
                "agent_name",
                "customer_name",
                "first_customer_text",
                "last_customer_text",
                "customer_start_sentiment",
                "customer_end_sentiment",
            ]
        ].copy()
        matches["similarity"] = similarities[top_idx]
        return matches

    def _clean_example_text(self, text: object, max_len: int = 110) -> str:
        return safe_excerpt(first_sentence(text), max_len)

    # ------------------------------------------------------------------
    # Conversational phrasing helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _sentiment_phrase(score: float) -> str:
        if score <= -0.6:
            return "very negative"
        if score <= -0.2:
            return "negative"
        if score < 0.2:
            return "neutral"
        if score < 0.6:
            return "positive"
        return "very positive"

    def format_top_negative_topics(self, top_n: int = 5, days: int = 30) -> str:
        report = self.top_negative_topics(top_n=top_n, days=days)
        if report.empty:
            return "I couldn't find any topics in that time window."

        intro = (
            f"Here are the top {len(report)} topics most associated with negative customer "
            f"sentiment over the last {days} days, ranked by how negative customers were when "
            f"they opened the conversation:"
        )
        paragraphs = [intro, ""]
        for i, row in enumerate(report.itertuples(index=False), start=1):
            sample = self.conversations[self.conversations["topic"] == row.topic].sort_values(
                "customer_start_sentiment"
            ).iloc[0]
            example = self._clean_example_text(sample.first_customer_text, 110)
            paragraphs.append(
                f"{i}. **{row.topic}** — {int(row.conversations)} conversation(s), average opening "
                f"sentiment {row.avg_start_sentiment:.2f} ({self._sentiment_phrase(row.avg_start_sentiment)}), "
                f"average shift during the chat {row.avg_delta:+.2f}. "
                f"For example, one customer wrote: \"{example}\""
            )
        paragraphs.append("")
        paragraphs.append(
            "These rankings are driven by the average sentiment score customers expressed in "
            "their opening message for each topic, so a topic with a more negative average score "
            "and a healthy number of conversations is a stronger signal than one based on a single chat."
        )
        return "\n".join(paragraphs)

    def format_coaching_candidates(self, top_n: int = 10, min_conversations: int = 2) -> str:
        report = self.coaching_candidates(top_n=top_n, min_conversations=min_conversations)
        if report.empty:
            return "I couldn't find enough agent data to make a coaching recommendation."

        lead = report.iloc[0]
        lead_sample = self.conversations[self.conversations["agent_name"] == lead.agent_name].sort_values(
            ["customer_end_sentiment", "customer_start_sentiment"]
        ).iloc[0]
        lead_example = self._clean_example_text(lead_sample.first_customer_text, 110)

        paragraphs = []
        paragraphs.append(
            f"Based on empathy language, how often customers were still negative by the end of the "
            f"chat, and resolution rate, **{lead.agent_name}** looks like the strongest coaching "
            f"candidate. Across {int(lead.conversations)} conversation(s), their messages used "
            f"empathy phrasing (e.g. \"I understand\", \"sorry\", \"let me check\") about "
            f"{lead.avg_empathy:.2f} times per reply on average, and {lead.negative_end_rate:.0%} of "
            f"their conversations ended with the customer still sounding negative "
            f"(resolved-looking rate: {lead.resolved_rate:.0%}). One representative case is the "
            f"\"{lead_sample.topic}\" conversation, which opened with: \"{lead_example}\"."
        )

        if len(report) > 1:
            others = report.iloc[1 : min(len(report), 5)]
            other_bits = []
            for row in others.itertuples(index=False):
                other_bits.append(
                    f"{row.agent_name} (avg empathy {row.avg_empathy:.2f}, "
                    f"{row.negative_end_rate:.0%} of chats end negative)"
                )
            paragraphs.append(
                "A few other agents show a similar pattern and are worth a look too: "
                + "; ".join(other_bits) + "."
            )

        paragraphs.append(
            "One caveat: most agents in this dataset only show up in one or two conversations, "
            "so treat this as a starting point for a manual review rather than a final verdict — "
            "a larger sample per agent would make the score more reliable."
        )
        return "\n\n".join(paragraphs)

    def format_sentiment_timeline(self, conv_id: str) -> str:
        timeline = self.sentiment_timeline(conv_id)
        summary = self.conversation_summary(conv_id)

        start_phrase = self._sentiment_phrase(summary["customer_start_sentiment"])
        end_phrase = self._sentiment_phrase(summary["customer_end_sentiment"])
        delta = summary["customer_sentiment_delta"]

        if delta > 0.2:
            trend = "improved noticeably"
        elif delta < -0.2:
            trend = "got noticeably worse"
        else:
            trend = "stayed roughly the same"

        intro = (
            f"In conversation {summary['conv_id']} (topic: {summary['topic']}, handled by "
            f"{summary['agent_name']}), the customer's sentiment {trend} over the course of the "
            f"chat — starting out {start_phrase} ({summary['customer_start_sentiment']:.2f}) and "
            f"ending {end_phrase} ({summary['customer_end_sentiment']:.2f}), a net change of "
            f"{delta:+.2f}. The conversation {'does' if summary['resolved_like'] else 'does not'} "
            f"look resolved based on the closing messages."
        )

        lines = [intro, "", "Turn-by-turn detail:"]
        for row in timeline.itertuples(index=False):
            lines.append(
                f"  Turn {int(row.turn_index)} ({row.role}, {row.sentiment_label}, "
                f"{row.sentiment_score:+.2f}): \"{row.text_excerpt}\""
            )
        return "\n".join(lines)

    def format_recurring_concerns(self, top_n: int = 10, days: int = 30) -> str:
        report = self.recurring_concerns(top_n=top_n, days=days)
        if report.empty:
            return "I couldn't find any recurring concerns in that window."

        intro = (
            f"Looking at the last {days} days, here are the recurring concerns leadership should "
            f"prioritize, ranked by a priority score that combines volume with how negative "
            f"customers were on average:"
        )
        paragraphs = [intro, ""]
        for i, row in enumerate(report.itertuples(index=False), start=1):
            sample = self.conversations[self.conversations["topic"] == row.topic].sort_values(
                "customer_start_sentiment"
            ).iloc[0]
            example = self._clean_example_text(sample.first_customer_text, 110)
            paragraphs.append(
                f"{i}. **{row.topic}** — {int(row.conversations)} conversation(s), priority score "
                f"{row.priority_score:.2f}, {row.neg_rate:.0%} started out negative, and only "
                f"{row.resolved_rate:.0%} look resolved by the end. Example: \"{example}\""
            )
        paragraphs.append("")
        paragraphs.append(
            "Priority score is conversation volume multiplied by how negative the average opening "
            "sentiment was, so it favors issues that are both common and emotionally charged."
        )
        return "\n".join(paragraphs)

    def answer(self, question: str) -> str:
        q = normalize_text(question)
        conv_match = re.search(r"\b(c\d{7})\b", q, flags=re.IGNORECASE)

        if conv_match and any(term in q for term in ["sentiment", "timeline", "conversation", "over the course"]):
            return self.format_sentiment_timeline(conv_match.group(1).upper())

        if any(term in q for term in ["top 5 topics", "top topics", "negative customer sentiment", "negative sentiment"]) and any(
            term in q for term in ["topic", "topics", "concern"]
        ):
            return self.format_top_negative_topics(top_n=5, days=30)

        if any(term in q for term in ["coach", "empathy", "communication", "agent"]):
            return self.format_coaching_candidates(top_n=10, min_conversations=2)

        if any(term in q for term in ["recurring concern", "recurring customer", "leadership should review", "review first"]):
            return self.format_recurring_concerns(top_n=10, days=30)

        if conv_match:
            return self.format_sentiment_timeline(conv_match.group(1).upper())

        if SKLEARN_AVAILABLE:
            matches = self.similar_cases(question, top_k=5)
            bits = []
            for row in matches.itertuples(index=False):
                bits.append(
                    f"- {row.conv_id} ({row.topic}, handled by {row.agent_name}, similarity "
                    f"{row.similarity:.2f}): \"{self._clean_example_text(row.first_customer_text, 100)}\""
                )
            return (
                "I couldn't match this to one of the built-in report types, so here are the "
                "closest matching conversations instead:\n\n" + "\n".join(bits)
            )

        return "I could not route the question with the built-in rules."


if STREAMLIT_AVAILABLE:
    @st.cache_resource(show_spinner=False)
    def load_bot(csv_path: str) -> ConversationAnalysisBot:
        return ConversationAnalysisBot(csv_path)
else:
    def load_bot(csv_path: str) -> ConversationAnalysisBot:
        return ConversationAnalysisBot(csv_path)


def run_streamlit_interface() -> None:
    st.set_page_config(page_title="Conversation Analysis", layout="wide")
    st.title("Conversation Analysis Bot")
    st.markdown("The app automatically loads cs_conversations.csv from the same folder as app.py.")

    csv_path = str(DEFAULT_CSV_PATH)

    if not DEFAULT_CSV_PATH.exists():
        st.error(f"Could not find {DEFAULT_CSV_PATH.name} next to app.py.")
        return

    with st.sidebar:
        st.header("Data")
        st.success(f"✓ Dataset loaded: {DEFAULT_CSV_PATH.name}")
        mode = st.radio(
            "Report type",
            [
                "Top negative topics",
                "Coaching candidates",
                "Recurring concerns",
                "Conversation timeline",
                "Ask a question",
            ],
        )
        top_n = st.number_input("Top N", min_value=1, max_value=20, value=5)
        days = st.slider("Recent window (days)", 7, 90, 30)
        conv_id = st.text_input("Conversation ID", "") if mode == "Conversation timeline" else ""
        question = st.text_area("Question", "What are the top negative topics?") if mode == "Ask a question" else ""

    try:
        bot = load_bot(csv_path)
    except Exception as exc:
        st.error(f"Unable to load CSV: {exc}")
        return

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"Conversations: *{bot.conversations.shape[0]}*  \nTurns: *{bot.turns.shape[0]}*"
    )

    # Dataset preview and quick diagnostics for the Streamlit UI
    st.subheader("Dataset preview")
    st.write(f"Loaded {bot.conversations.shape[0]} conversations and {bot.turns.shape[0]} turns.")
    try:
        st.dataframe(bot.conversations.head(10))
    except Exception as exc:
        st.error(f"Failed to render dataset preview: {exc}")

    if mode == "Top negative topics":
        st.header("Top negative topics")
        st.markdown(bot.format_top_negative_topics(top_n=top_n, days=days))
    elif mode == "Coaching candidates":
        st.header("Coaching candidates")
        st.markdown(bot.format_coaching_candidates(top_n=top_n, min_conversations=2))
    elif mode == "Recurring concerns":
        st.header("Recurring concerns")
        st.markdown(bot.format_recurring_concerns(top_n=top_n, days=days))
    elif mode == "Conversation timeline":
        st.header("Conversation sentiment timeline")
        if not conv_id:
            st.warning("Enter a conversation ID in the sidebar.")
        else:
            try:
                st.markdown(bot.format_sentiment_timeline(conv_id))
            except KeyError as exc:
                st.error(str(exc))
    else:
        st.header("Question answer")
        if not question.strip():
            st.info("Enter a question in the sidebar.")
        else:
            st.markdown(bot.answer(question))


def main() -> None:
    parser = argparse.ArgumentParser(description="Conversation analysis bot for contact-centre insights.")
    parser.add_argument(
        "--csv",
        default=str(DEFAULT_CSV_PATH),
        help="Path to the conversation CSV file",
    )
    parser.add_argument("--question", help="Free-text question to answer")
    parser.add_argument("--topic-report", action="store_true", help="Print top negative topics")
    parser.add_argument("--coach-agents", action="store_true", help="Print coaching candidates")
    parser.add_argument("--sentiment-timeline", help="Print sentiment timeline for a specific conv_id")
    parser.add_argument("--concerns", action="store_true", help="Print recurring concerns report")
    parser.add_argument("--ask", help="Alias for --question")
    args, _ = parser.parse_known_args()

    streamlit_mode = STREAMLIT_AVAILABLE and getattr(st.runtime, "exists", lambda: False)()

    if streamlit_mode:
        run_streamlit_interface()
        return

    csv_path = args.csv or str(DEFAULT_CSV_PATH)
    bot = ConversationAnalysisBot(csv_path)

    if args.topic_report:
        print(bot.format_top_negative_topics())
        return

    if args.coach_agents:
        print(bot.format_coaching_candidates())
        return

    if args.sentiment_timeline:
        print(bot.format_sentiment_timeline(args.sentiment_timeline))
        return

    if args.concerns:
        print(bot.format_recurring_concerns())
        return

    question = args.question or args.ask
    if question:
        print(bot.answer(question))
        return

    print("Conversation Analysis Bot loaded successfully.")
    print(f"Conversations: {bot.conversations.shape[0]} | Turns: {bot.turns.shape[0]}")
    print()
    print(bot.format_top_negative_topics())
    print()
    print(bot.format_coaching_candidates(top_n=5, min_conversations=2))


if __name__ == "__main__":
    main()