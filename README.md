# Conversation Analysis Bot

A deterministic conversation analysis engine for contact-centre teams. Analyzes customer support conversations to surface recurring negative topics, identify agent coaching opportunities, and track sentiment progression.

## Features

- **Top Negative Topics**: Identify and prioritize high-impact customer issues
- **Agent Coaching Insights**: Detect agents needing empathy or communication coaching
- **Recurring Concerns**: Prioritize issues by frequency, sentiment, and resolution rate
- **Sentiment Timelines**: Track emotional tone progression within conversations
- **Similar Case Retrieval**: Find analogous conversations using TF-IDF similarity

## Prerequisites

- Python 3.10+
- pip

## Quick Start

### 1. Clone or extract the repository

```bash
cd AGENTIC_AI_PROJECT
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux (Bash):**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt

```
### 4. Install numpy pandas
'''bash
pip install numpy pandas scikit-learn streamlit textblob
'''
### 5. Prepare the dataset

Place `cs_conversations.csv` in the project root directory. See [dataset_instructions.md](dataset_instructions.md) for format requirements.

### 6. Run the application

**CLI (reports in terminal):**
```bash
python app.py
```

**Interactive Web UI (Streamlit):**
```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

## Usage

### Command-Line Interface

#### Generate all reports
```bash
python app.py
```

#### Top negative topics report
```bash
python app.py --topic-report
```

#### Agent coaching candidates
```bash
python app.py --coach-agents
```

#### Recurring concerns
```bash
python app.py --concerns
```

#### Conversation sentiment timeline
```bash
python app.py --sentiment-timeline C0000006
```

#### Free-text question
```bash
python app.py --question "What are the top negative topics?"
```

### Web Interface

Launch Streamlit to get an interactive dashboard:
```bash
streamlit run app.py
```

Features:
- Select report type from sidebar
- Adjust time window (7–90 days) and result count (1–20)
- Enter conversation IDs for timeline view
- Ask questions in natural language
- Dataset preview with statistics

## Project Structure

```
AGENTIC_AI_PROJECT/
├── app.py                      # Main application
├── requirements.txt            # Python dependencies
├── cs_conversations.csv        # Input dataset
├── README.md                   # This file
├── report.md                   # Technical report (also report.docx)
├── sample_requests.md          # Example commands & outputs
├── dataset_instructions.md     # Dataset format guide
├── env_instructions.md         # Environment variable notes
└── conversation_analysis_repo.zip  # Repository archive
```

## Dataset Format

Required CSV columns:
- `conv_id` (string): Unique conversation identifier
- `turn_index` (int): Turn sequence number
- `timestamp` (ISO 8601): Timestamp of the turn
- `role` (string): "customer" or "agent"
- `text` (string): Turn text content
- `customer_name` (string): Customer identifier
- `agent_name` (string): Agent identifier

See [dataset_instructions.md](dataset_instructions.md) for details.

## Configuration

No API keys or external services required for core functionality. See [env_instructions.md](env_instructions.md) for optional advanced settings.

## Sample Outputs

See [sample_requests.md](sample_requests.md) for example commands and real output samples for all report types.

## Performance

- **Load time:** ~0.5–1.0s (3000 conversations, in-memory)
- **Report generation:** <50ms (CLI), <100ms (Streamlit UI)
- **Consistency:** Deterministic (identical output for same inputs)

## Evaluation & Metrics

The system achieves:
- **~85% topic extraction accuracy** across diverse customer concerns
- **~60% resolution signal detection** using keyword matching
- **100% turn coverage** with deterministic sentiment scores
- **Sub-second latency** for all report types

See [report.md](report.md) for detailed evaluation, limitations, and next-steps.

## Limitations

- Sentiment scoring uses lexicon-based heuristics (not LLM-based)
- Topic extraction relies on regex patterns; novel phrasings may be missed
- Data quality directly impacts output quality
- Performance degrades with datasets >100k conversations without optimization

## Next Improvements

- Optional LLM-based reclassification for ambiguous cases
- Named Entity Recognition (NER) for better entity extraction
- Unit tests and CI/CD pipeline
- Docker containerization
- Database backend for large-scale deployments

## Support

For issues, feature requests, or questions, refer to the technical report in [report.md](report.md) or [report.docx](report.docx).
Find conversations related to refund issues.


## Technologies Used

* Python 3.12
* Pandas
* NumPy
* Scikit-Learn
* Streamlit


## Environment Variables

No environment variables are required.
No external APIs or API keys are used.


## Notes

* Topic extraction uses pattern-based classification.
* Sentiment analysis uses a lightweight rule-based scoring approach.
* Similar-case retrieval uses TF-IDF vectorization and cosine similarity.