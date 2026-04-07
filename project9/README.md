# Water Stress and Agricultural Commodities

## Research Question
Does water stress predict agricultural commodity price movements?

## Methodology
**Language:** Python  
**Methods:** Time series regression, Granger causality

## Data
Yahoo Finance commodity ETFs, drought index data

## Key Findings
Water stress indicators Granger-cause certain commodity returns with 2–3 month lags.

## How to Run
```bash
pip install -r requirements.txt
python code/project9_*.py
```

## Repository Structure
```
├── README.md
├── requirements.txt
├── .gitignore
├── code/          ← Analysis scripts
├── data/          ← Raw and processed data
└── output/
    ├── figures/   ← Charts and visualizations
    └── tables/    ← Summary statistics and regression results
```

## Author
Alfred Bimha

## License
MIT

---
*Part of a 20-project sustainable finance research portfolio.*
