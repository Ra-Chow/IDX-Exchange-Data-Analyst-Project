# IDX Exchange: Data Analyst Internship Project

Welcome to my repository for the **IDX Exchange Data Analyst Internship Program**. This repository houses a progressive data engineering and analytics pipeline designed to transform raw MLS dataset records into polished housing market intelligence.

## Contact & Connect
* **Author**: Rainee Chow (Team Lead)
* **LinkedIn:** [www.linkedin.com/in/rainee-c]
* **Handshake:** [https://uic.joinhandshake.com/profiles/hbfm5d]
* **Tableau Public**: [https://public.tableau.com/app/profile/rainee.chow/vizzes]
---

## Project Summary
This project represents my individual work completed over a structured 12-week curriculum alongside my 6-member cohort team. The objective of the pipeline is to securely ingest raw, confidential monthly MLS listing and transaction datasets, engineer high-value real estate metrics, filter out market anomalies, and deliver interactive business dashboards.

### The Data Pipeline Pipeline Phases
The project is structured as a progressive end-to-end data pipeline

| Phase | Description | Key Tools |
| :--- | :--- | :--- |
| **1. Data Cleaning & Aggregation**  | Load and concatenate raw monthly CSV entries into combined time-series files. | Python, Pandas |
| **2. Market Analytics & Enrichment** | Merge live economic markers (FRED 30-year fixed mortgage rates). | Python (APIs, Datetime) |
| **3. Feature Engineering** | Compute performance metrics like Price per Sq Ft and Escrow durations. | Python (Pandas)|
| **4. Outlier Detection**  | Apply statistical cleaning (Interquartile Range Method) to flag extreme values. | Python, NumPy/Pandas |
| **5. Dashboard Development** | Synthesize clean datasets into interactive, localized market-tracking dashboards. | Tableau Desktop Public |

---

## Deliverables
The ultimate goal of this pipeline is to supply a comprehensive analytics package:
1. **Interactive Tableau Workbooks (`.twbx`)**: Publicly hosted interactive charts tracking localized median close price trends, inventory volume, and top 100 agent performance tiers.
2. **One-Page Market Intelligence Report**: A focused executive brief targeting localized pricing trends, market activity levels, and 3–5 core data-driven takeaways.
3. **Executive Presentation**: A live 5-minute technical walkthrough summarizing critical discoveries for the Summer 2026 Cohort.

---

## How to Run my Work Locally
> **Proprietary Data Notice:** The original source transaction datasets are proprietary to the CoreLogic Trestle API and are protected under strict corporate confidentiality terms. The actual raw files are omitted from this public repository via `.gitignore`. You can execute this pipeline using your own mock data structured to match the layout below.

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/IDX-Exchange-Data-Analyst-Internship-Project.git](https://github.com/your-username/IDX-Exchange-Data-Analyst-Internship-Project.git)
cd IDX-Exchange-Data-Analyst-Internship-Project
```
### 2. Configure Environment
```bash
pip install pandas
```
### 3. Establish Proper Directory Layout
Create a subdirectory named *csv/* inside your project, and drop your structured files inside using the following naming conventions:
* Listing Data: csv/CRMLSListingYYYYMM.csv
* Sold Transaction Data: csv/CRMLSSoldYYYYMM.csv
### 4. Run the Pipeline Execution Scripts
For example, to run automated cleaning, concatenation, and filter routines, execute:
```bash
python week1_aggregation.py
```

## Team & Contributions
This project was developed as a collaborative milestone effort within a 6-member data analyst team.

### Program Leadership & Mentorship
* Aidan Nguyen - Cohort Senior Leadership
* Yoora Choi - Team Coach

### Core Data Analyst Team DA39
* Rainee Chow - Team Lead
* Matthew Baur - Team Member
* Najja King - Team Member
* Palak Lunia - Team Member
* Qianyu (Lisa) Lu - Team Member
* Zewen Yang - Team Member
