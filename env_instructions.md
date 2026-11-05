# Environment & External Services

## Environment Variables

The application does not require any environment variables for basic operation.

### Optional Variables

If planning to integrate with external LLM services:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

(Not used in current deterministic version, but available for future enhancements)

## External Services

**None required.** The application uses only local, deterministic processing:

- No API calls to cloud services
- No database connections
- No authentication/credentials needed

## System Requirements

- **Python Version:** 3.10+
- **OS:** Windows, macOS, Linux
- **RAM:** ≥512 MB (for typical datasets)
- **Disk:** ≥100 MB (for code + small dataset)

## Dependencies

All dependencies are listed in `requirements.txt`:

```
pandas        # Data manipulation
numpy         # Numerical computing
scikit-learn  # TF-IDF similarity and ML utilities
streamlit     # Web UI (optional for CLI-only use)
```

Install with:
```bash
pip install -r requirements.txt
```

## Performance Notes

- **In-memory processing:** Entire dataset loaded into RAM
- **Single-threaded:** No parallelization implemented
- **Typical latency:** <100ms for report generation on 3000-conversation dataset

For larger datasets (>100k conversations), consider:
- Database-backed architecture
- Incremental processing
- Caching strategies

## Troubleshooting

### Module not found errors
Ensure virtual environment is activated and dependencies installed:
```bash
.venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt
```

### CSV not found
Place `cs_conversations.csv` in project root, or specify path:
```bash
python app.py --csv /path/to/file.csv
```

### Streamlit fails to start
Update Streamlit:
```bash
pip install --upgrade streamlit
```

## Security

- No sensitive data is stored or transmitted
- All processing is local
- CSV file remains unchanged during execution
