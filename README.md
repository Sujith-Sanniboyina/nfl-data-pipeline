# NFL Data Pipeline

![Python Version](https://img.shields.io/badge/python-3.8+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-mvp--in--progress-yellow)

An end-to-end data pipeline that extracts NFL play-by-play data, transforms it into analyzable formats, and loads it into a database with automated reporting.

---

## Project Overview

This project demonstrates data engineering practices through a complete ETL (Extract, Transform, Load) pipeline:

- Extract: Pull NFL play-by-play data using nfl_data_py
- Transform: Clean, standardize, and aggregate data with Pandas
- Load: Store processed data in SQLite/PostgreSQL with an optimized schema
- Report: Generate automated weekly team performance summaries

---

## Skills Demonstrated

| Category | Technologies |
|----------|-------------|
| **Languages** | Python 3.8+ |
| **Data Processing** | Pandas, NumPy |
| **ETL Architecture** | Modular pipeline design (extract/transform/load separation) |
| **Database** | SQLite (current), PostgreSQL (planned) |
| **Testing** | pytest |
| **Version Control** | Git, GitHub |
| **Containerization** | Docker (planned) |
| **CI/CD** | GitHub Actions (planned) |

---

## Project Structure

```
nfl-data-pipeline/
├── extract/
│   └── nfl_api_client.py
├── transform/
│   ├── clean.py
│   └── aggregate.py
├── load/
│   ├── schema.sql
│   └── load_to_db.py
├── analytics/
│   └── generate_report.py
├── notebooks/
├── tests/
├── requirements.txt
└── README.md
```

---

## Data Source

This pipeline uses nfl_data_py, a Python library that provides access to NFLverse data, including:

- Play-by-play data since 1999
- Team statistics
- Player information
- Game outcomes

Why this API:
- Free and open-source
- No API key required
- Well-maintained
- Comprehensive dataset

---

## Current Status

- [x] Project structure setup
- [x] Requirements defined
- [x] Basic README documentation
- [ ] Extract module (in progress)
- [ ] Transform module
- [ ] Load module with SQLite
- [ ] Basic reporting
- [ ] Unit tests
- [ ] Complete documentation

---

## Installation

Prerequisites:
- Python 3.8 or higher
- pip

Setup:

```bash
git clone https://github.com/yourusername/nfl-data-pipeline.git
cd nfl-data-pipeline

python -m venv venv

source venv/bin/activate
# Windows:
venv\Scripts\activate

pip install -r requirements.txt
```
---

## Usage

Commands will be available once modules are implemented.

Expected commands:

```bash
# Run full pipeline
python run_pipeline.py

# Run individual modules
python extract/nfl_api_client.py
python transform/aggregate.py
python load/load_to_db.py

# Generate report
python analytics/generate_report.py
```

---

## Sample Output

(To be added once pipeline is functional)

- Top 10 teams by offensive yards
- Weekly performance summaries
- Data quality metrics
- Sample database queries

---

## Roadmap

Phase 1: MVP (Current)
- Complete core ETL pipeline
- Extract 2–3 seasons of play-by-play data
- Generate basic text reports
- Add documentation

Phase 2: Production Ready
- Switch to PostgreSQL
- Add Docker containerization
- Implement unit tests
- Add data validation and error handling
- Create data quality dashboards

Phase 3: Advanced Features
- Automate runs with GitHub Actions
- Build a Streamlit dashboard
- Add ML predictions for game outcomes
- Create REST API endpoints
- Backfill historical data to 1999

---

## Testing

Run tests with:
```bash
pytest tests/
```

Tests are currently in development.

---

## License

MIT License. See LICENSE file for details.

---

## Author

Sujith Sanniboyina
GitHub: https://github.com/Sujith-Sanniboyina
LinkedIn: www.linkedin.com/in/sujith-sanniboyina

---

## Acknowledgments

- nfl_data_py for providing access to NFL data
- NFLverse community for maintaining datasets
- Data engineering community for best practices

---

This project is actively under development.