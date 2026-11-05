# Sample Requests and Outputs

## 1. Top Negative Topics (CLI)

### Command:
```bash
python app.py --topic-report
```

### Output:
```
Top negative topics

1. duplicate_charge::refund | conversations=2 | avg_start_sentiment=-0.756 | avg_delta=0.402
   Example: i was charged twice for my refund.
2. service_not_working::refund | conversations=5 | avg_start_sentiment=-0.707 | avg_delta=0.540
   Example: hello, my refund is not working as expected.
3. app_crash::refund | conversations=3 | avg_start_sentiment=-0.707 | avg_delta=0.900
   Example: app crash ho rahi hai while using refund.
4. delivery_delay::refund | conversations=2 | avg_start_sentiment=-0.447 | avg_delta=0.447
   Example: the refund hasn't arrived yet.
5. duplicate_charge::wallet | conversations=5 | avg_start_sentiment=-0.378 | avg_delta=0.326
   Example: i was charged twice for my wallet.



## 2. Agent Coaching Candidates (CLI)

### Command:

python app.py --coach-agents

### Output:

Agent coaching candidates

Note: many agents only appear once or twice in this dataset, so the score is a rough signal rather than a verdict.

1. AgentOYYZ | convs=2 | avg_empathy=0.94 | neg_end_rate=100.00% | resolved_rate=100.00% | coaching_need_score=0.699
   Evidence issue: login_access | i can't log in.
2. AgentXLXY | convs=2 | avg_empathy=0.99 | neg_end_rate=50.00% | resolved_rate=50.00% | coaching_need_score=0.557
   Evidence issue: service_not_working::lab test | hello, my lab test is not working as expected.
3. AgentDTWY | convs=2 | avg_empathy=0.93 | neg_end_rate=50.00% | resolved_rate=100.00% | coaching_need_score=0.489
   Evidence issue: delivery_delay::return | the return hasn't arrived yet.
4. AgentMPOW | convs=2 | avg_empathy=1.02 | neg_end_rate=50.00% | resolved_rate=100.00% | coaching_need_score=0.475
   Evidence issue: delivery_delay::roaming | the roaming hasn't arrived yet.
5. AgentOESR | convs=2 | avg_empathy=1.10 | neg_end_rate=50.00% | resolved_rate=100.00% | coaching_need_score=0.458
   Evidence issue: coupon_checkout::mock test | coupon isn't applying on checkout for the mock test.


## 3. Free-Text Question (CLI)

### Command:
```bash
python app.py --question "What are the top negative topics?"
```

### Output:
(Same as Section 1 — returns top negative topics based on the question)

---

## 4. Sentiment Timeline for a Conversation (CLI)

### Command:
```bash
python app.py --sentiment-timeline C0000006
```

### Output:

Conversation C0000006
Topic: service_not_working::audit logs
Agent: AgentLTLP
Customer sentiment delta: 0.612

Turn-by-turn sentiment

00 | customer | negative | -0.816 | hello, my audit logs is not working as expected.
01 | agent    | neutral  | +0.000 | i'm sorry about the trouble.
02 | customer | neutral  | +0.000 | okay, waiting for an update.
03 | agent    | positive | +0.653 | i understand this is frustrating.
04 | customer | positive | +0.707 | it's urgent.


## 5. Recurring Concerns Report (CLI)

### Command:

python app.py --concerns

### Output:

Recurring customer concerns

1. duplicate_charge::refund | conversations=7 | priority_score=5.29 | neg_rate=71.43% | resolved_rate=57.14%
   Example: i was charged twice for my refund.
2. service_not_working::refund | conversations=8 | priority_score=5.66 | neg_rate=75.00% | resolved_rate=62.50%
   Example: hello, my refund is not working as expected.
3. app_crash::refund | conversations=5 | priority_score=3.54 | neg_rate=60.00% | resolved_rate=80.00%
   Example: app crash ho rahi hai while using refund.
4. delivery_delay::refund | conversations=6 | priority_score=3.27 | neg_rate=66.67% | resolved_rate=50.00%
   Example: the refund hasn't arrived yet.
5. duplicate_charge::wallet | conversations=9 | priority_score=3.41 | neg_rate=55.56% | resolved_rate=66.67%
   Example: i was charged twice for my wallet.


## 6. Streamlit Web UI

### Command:

streamlit run app.py

### Access:
Open your browser and navigate to:
- **Local:** `http://localhost:8501`
- **Network:** `http://192.168.2.223:8501`

### Features:
- **Sidebar controls:**
  - Select report type: Top negative topics, Coaching candidates, Recurring concerns, Conversation timeline, Ask a question
  - Adjust "Top N" results (1–20)
  - Set time window (7–90 days)
  - Enter conversation ID for timeline
  - Ask free-text questions

- **Dataset preview:** Shows first 10 conversations in the loaded dataset
- **Real-time updates:** Select report type and parameters to instantly generate reports

### Screenshot example:
The UI displays formatted reports in code blocks for easy copying and analysis.

---

## Summary of Report Types

- **Top Negative Topics**: Identifies conversations with the most negative customer sentiment to prioritize support improvements.
- **Coaching Candidates**: Flags agents needing skill development, empathy training, or communication coaching.
- **Sentiment Timeline**: Shows turn-by-turn sentiment progression within a specific conversation for detailed analysis.
- **Recurring Concerns**: Prioritizes issues by frequency, severity, and impact to guide product/ops roadmap.
- **Free-Text Questions**: Allows natural language queries that intelligently route to the most relevant report type.

---

## Summary of Report Types

- **Top Negative Topics**: Identifies conversations with the most negative customer sentiment
- **Coaching Candidates**: Flags agents needing skill development or empathy coaching
- **Sentiment Timeline**: Shows turn-by-turn sentiment progression within a conversation
- **Recurring Concerns**: Prioritizes issues by frequency and severity
- **Free-Text Questions**: Allows natural language queries (routes to appropriate reports)
