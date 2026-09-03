# NFL Data Pipeline

![Python Version](https://img.shields.io/badge/python-3.8+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-mvp--complete-brightgreen)

An end-to-end data pipeline that extracts NFL play-by-play data, transforms it into analyzable formats, and loads it into a database with automated reporting.

---

## Project Overview

This project demonstrates data engineering practices through a complete ETL (Extract, Transform, Load) pipeline:

- Extract: Pull NFL play-by-play and weekly player data using nflreadpy
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
├── main.py
├── extract/
│   └── extract.py
├── transform/
│   └── transform.py
├── load/
│   ├── schema.sql
│   └── load.py
├── analytics/
│   └── analytics.py
├── tests/
│   ├── test_extract.py
│   ├── test_transform.py
│   └── test_load.py
├── data/
├── requirements.txt
└── README.md
```

---

## Data Source

This pipeline uses nflreadpy (the actively maintained successor to the now-deprecated nfl_data_py), a Python library that provides access to NFLverse data, including:

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

## Project Status

**COMPLETE** - MVP with working ETL pipeline, database integration, and automated reporting.

- [x] Project structure setup
- [x] Requirements defined
- [x] Basic README documentation
- [x] Extract module
- [x] Transform module
- [x] Load module with SQLite
- [x] Basic reporting
- [x] Unit tests
- [x] Complete documentation

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

Run the complete ETL pipeline:

```bash
# Run full pipeline
python main.py

# Run individual modules
python extract/extract.py
python transform/transform.py
python load/load.py

# Generate report
python analytics/analytics.py

# Run test cases for different modules
python tests/test_extract.py
python tests/test_transform.py
python tests/test_load.py
```

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

Run tests with pytest:
```bash
pytest tests/ -v
# Run with coverage report
pytest tests/ -v --cov=extract --cov=transform --cov=load --cov-report=term-missing
```

---

## License

MIT License. See LICENSE file for details.

---

## Author

**Sujith Sanniboyina**
- GitHub: [@Sujith-Sanniboyina](https://github.com/Sujith-Sanniboyina)
- LinkedIn: [Sujith Sanniboyina](https://www.linkedin.com/in/sujith-sanniboyina)

---

## Acknowledgments

- nflreadpy for providing access to NFL data
- NFLverse community for maintaining datasets
- Data engineering community for best practices