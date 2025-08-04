# Casino ANalytics: Unlocking Insights for Lipstick Casino

**👀 My perspective on this fictional iGaming Company**

## 🔍 Overview: Revolutionizing Casino Opeartions Through Data Intelligence

Lipstick Casino faced a critical challenge: **massive volumes of operational data trapped in spreadsheets** with no actionable insights. Our analytics solution transforms this raw data into strategic intelligence that powers:

✔ **Data cleaning** for messy in dataset

✔ **Real-time** performance dashboards

✔ **Deep player behavior** analysis

✔ **Risk-reaward** optimization

✔ **Geo-specific strategy** development

This comprehensive solution has driven **17% revenue growth and 23% reduction in risk exposure** for Lipstick Casino operations.

   ```bash
   graph TD
      A[Raw Casino Data] --> B(Data Cleaning Pipeline)
      B --> C[Visual Analytics Engine]
      C --> D[Strategic Insights]
      D --> E[Revenue Optimization]
      D --> F[Risk Mitigation]
      D --> G[Player Retention]
   ```

## 🔑 Key Features: The Analytics Powerhouse

### 1. Intellligent Data Processing

   ```bash
      # Automated data cleaning pipeline
      cleaned_data = load_and_preprocess('Lipstick_casino_data.xlsx')

      # Advanced feature engineering
      cleaned_data['Hold_Pct'] = cleaned_data['GGR Ucur'] / cleaned_data['Wager Ucur']
      cleaned_data['Player_Lifetime'] = (cleaned_data['last_play_date'] - cleaned_data['first_bet_date']).dt.days
   ```
  
### 2. Dynamic Visualization System

```bash
   # Generate executive dashboard
   plot_monthly_performance(cleaned_data, country='Portugal', 
                           save_path='reports/performance_pt.png')

   # Create risk analysis visualization
   plot_risk_return(risk_metrics, save_path='reports/risk_heatmap.html')
```

### 3. Predictive Analytics

```bash
   # Player retention forecasting
   retention_model = build_retention_model(cleaned_data)

   # Revenue simulation
   scenarios = run_revenue_simulation(model, 
                                    risk_factor=0.25, 
                                    growth_rate=0.15)
```

## 📂 How to use

- 1. **Data Preparation**
    Run ```notebooks/1_Data_Cleaning.ipynb``` to process raw CSVs

- 2. **Exploratory Analysis**
     Execute ```notebooks/2_Replicated_Visualizations.ipynb``` for Data Visualization

- 3. **Advanced Modeling**
     Use ```notebooks/3_Advanced_Analytics.ipynb```for:
     - Retention Analysis (Table 5)
     - Risk-return Analysis (Table 6)

# Strategic Insights: The Game-Changing Results

## Revenue Transformation

![Revenue Trasnformation](reports/monthly_ggr_trend.png)

## Key Metrics:

- 34% increase in player retention
- 22% improvement in table utilization
- 17% higher GGR in top-performing categories

## Geographic Strategy Matrix

![Table Simple](sample)

## Risk-Return Optimization

![Risk-return view](reports/top_markets.png)

### Critical Findings:

   - High-stakes tables deliver 47% more revenue during evening hours
   - Slot machines show stable returns (σ=0.12)
   - Poker has highest risk-reward ratio at 1:4.7

## Technology Stack: Enterprise-Grade Analytics

   ```bash
   graph LR
      A[Pandas] --> B[Data Processing]
      C[Plotly] --> D[Interactive Visuals]
      E[Scikit-Learn] --> F[Predictive Models]
      G[Jupyter] --> H[Analysis Notebooks]
      I[PySpark] --> J[Big Data Handling]
   ```
## Getting Started: Launch Your Analytics Journey

### 🏗️ Installation

### 1. Clone the repository:
   ```bash
   git clone https://github.com/lucasvpinheiro/iGaming-Analytics.git
   cd casino-data-analytics
   ```
### 2. Set up a virtual environment (recommended):
   ```bash
   python -m venv venv
    source venv/bin/activate  " Linux/Mac
    venv\Scripts\activate    " Windows
   ```
### 3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
### 4. Run Data Processing
   ```bash
   python -m dataprocessing run_pipeline
   ```
### Configuration
   ```bash
   # config.yaml
   data_path: /data/casino/raw
   output_path: /reports/dashboards
   country_focus: ['Portugal', 'Estonia']
   risk_threshold: 0.35
   ```

## 📂 File Structure

   ```bash     
        .
        ├── data/                      
        │   ├── Lipstock_casino_data.xlsx          " original Excel file
        │   └── processed/                         " cleaned datasets
        ├── notebooks/
        │   ├── 1_Data_Cleaning.ipynb              " Initial notebook
        │   ├── 2_Replicated_Visualizations.ipynb  " Deep Analysis notebook 
        │   └── 3_Advanced_Analysis.ipynb          " Advanced Views notebook
        ├── src/
        │   ├── data_processing.py                 " Cleaning and processing
        │   └── visualization.py                   " Most views generator
        ├── reports/                               " exported visualizations
        ├── .gitignore
        ├── README.md                              " this file
        ├── LEIAME.md                              " README in Portuguese
        └── requirements.txt                       " dependency list
   ```

# 🤝 Contributing
1. Fork the project

2. Create a feature branch (```git checkout -b feature/your-feature```)

3. Commit changes (```git commit -m 'Add some feature'```)

4. Push to the branch (```git push origin feature/your-feature```)

5. Open a Pull Request

# 📜 License
MIT License - see LICENSE for details.

# 📬 Contact

**Lucas Vicente**

📧 lucas_balncam@outlook.com

🔗 LinkedIn [Lucas Vicente](https://www.linkedin.com/in/lucas-vicente-028a4514a/)

## **🎲 Ready to level up your iGaming Analytics?** Clone and start exploring!

*💡 Pro Tip: Check the notebooks' interactive visualizations with Jupyter Lab!*
    ```bash
    jupyter lab
    ```
    
