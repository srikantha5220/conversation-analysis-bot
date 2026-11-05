# Dataset Instructions

## Overview
This project requires a CSV file named `cs_conversations.csv` containing customer support conversation data.

## File Location
Place the CSV file in the project root directory (same folder as `app.py`).

## Required Columns

The CSV must contain the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `conv_id` | string | Unique conversation identifier (e.g., C0000001) |
| `turn_index` | int | Turn sequence number within the conversation (0, 1, 2, ...) |
| `timestamp` | ISO 8601 datetime | Timestamp of the turn (e.g., 2025-08-10T16:48:00) |
| `role` | string | "customer" or "agent" |
| `text` | string | Full turn text content |
| `customer_name` | string | Customer identifier or name |
| `agent_name` | string | Agent identifier or name |

## Data Quality Notes

- **Missing values:** Rows with missing `role` or `text` will be skipped or handled gracefully
- **Text encoding:** UTF-8 recommended; other encodings may cause parsing errors
- **Timestamp format:** Must be parseable by pandas `pd.to_datetime()`
- **Noise:** If text contains trailing garbage, the system uses the first sentence for analysis

## Example Rows

```csv
conv_id,turn_index,timestamp,role,text,customer_name,agent_name
C0000001,0,2025-08-10T16:48:00,customer,Hello I was charged twice for my refund.,CustA,AgentX
C0000001,1,2025-08-10T16:50:00,agent,I'm sorry about that. Let me investigate.,CustA,AgentX
C0000002,0,2025-08-11T10:15:00,customer,My app keeps crashing.,CustB,AgentY
C0000002,1,2025-08-11T10:20:00,agent,I understand. Let me help you troubleshoot.,CustB,AgentY
```

## Preparing Your Dataset

1. Ensure the CSV has the required columns
2. Verify timestamps are in ISO 8601 format
3. Confirm `role` values are exactly "customer" or "agent" (case-sensitive)
4. Save as UTF-8 encoded CSV
5. Place in project root as `cs_conversations.csv`

## Running with Custom CSV

To use a different CSV file name or location:

```bash
python app.py --csv /path/to/custom_file.csv
```

## Sample Dataset

A sample `cs_conversations.csv` is included with this project (~23 MB, 3000 conversations).
