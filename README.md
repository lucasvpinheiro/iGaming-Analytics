<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Casino Data Analytics Project</title>
    <style>
        :root {
            --primary: #2c3e50;
            --secondary: #3498db;
            --accent: #e74c3c;
            --light: #ecf0f1;
            --dark: #2c3e50;
            --success: #27ae60;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f9f9f9;
        }
        
        header {
            background-color: var(--primary);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            text-align: center;
        }
        
        h1, h2, h3 {
            color: var(--primary);
        }
        
        h1 {
            margin-top: 0;
            font-size: 2.5em;
        }
        
        h2 {
            border-bottom: 2px solid var(--secondary);
            padding-bottom: 5px;
            margin-top: 30px;
        }
        
        .badge {
            display: inline-block;
            background-color: var(--secondary);
            color: white;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            margin-right: 5px;
        }
        
        code {
            background-color: #f0f0f0;
            padding: 2px 5px;
            border-radius: 3px;
            font-family: 'Courier New', Courier, monospace;
        }
        
        pre {
            background-color: #2d2d2d;
            color: #f8f8f2;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }
        
        a {
            color: var(--secondary);
            text-decoration: none;
        }
        
        a:hover {
            text-decoration: underline;
        }
        
        .file-structure {
            background-color: #f5f5f5;
            border-left: 4px solid var(--secondary);
            padding: 10px 15px;
            font-family: monospace;
            line-height: 1.8;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .visualization {
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin: 20px 0;
        }
        
        .visualization img {
            max-width: 100%;
            border-radius: 5px;
            display: block;
            margin: 10px auto;
        }
        
        footer {
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            border-top: 1px solid #eee;
        }
        
        @media (max-width: 768px) {
            body {
                padding: 15px;
            }
            
            h1 {
                font-size: 2em;
            }
        }
    </style>
</head>
<body>
    <header>
        <h1>Casino Data Analytics Project</h1>
        <p><strong>📊 Advanced Analysis of iGaming Performance Metrics</strong></p>
    </header>

    <section>
        <h2>📌 Overview</h2>
        <p>This project analyzes casino operational data to extract actionable business insights. It includes:</p>
        <ul>
            <li><span class="badge">✔️</span> <strong>Data cleaning</strong> for messy casino data</li>
            <li><span class="badge">✔️</span> <strong>Market performance</strong> benchmarking</li>
            <li><span class="badge">✔️</span> <strong>Player behavior</strong> analysis (LTV, churn prediction)</li>
            <li><span class="badge">✔️</span> <strong>Time series forecasting</strong> for revenue prediction</li>
            <li><span class="badge">✔️</span> <strong>Bonus effectiveness</strong> evaluation</li>
        </ul>
        <p>Built with Python in Jupyter Notebook, using Pandas, Scikit-learn, and Statsmodels.</p>
    </section>

    <section>
        <h2>🛠️ Installation</h2>
        <ol>
            <li>Clone the repository:
                <pre><code>git clone https://github.com/yourusername/casino-data-analytics.git
cd casino-data-analytics</code></pre>
            </li>
            <li>Set up a virtual environment (recommended):
                <pre><code>python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows</code></pre>
            </li>
            <li>Install dependencies:
                <pre><code>pip install -r requirements.txt</code></pre>
            </li>
        </ol>
    </section>

    <section>
        <h2>📂 File Structure</h2>
        <div class="file-structure">
            .
            ├── data/                      
            │   ├── raw/                  # Original CSV files
            │   └── processed/            # Cleaned datasets
            ├── notebooks/
            │   ├── 1_Data_Cleaning.ipynb # Data preprocessing
            │   ├── 2_EDA.ipynb           # Exploratory analysis 
            │   └── 3_Advanced_Analytics.ipynb # Forecasting & ML
            ├── reports/                  # Exported visualizations
            ├── .gitignore
            ├── README.md                 # This file
            └── requirements.txt          # Dependency list
        </div>
    </section>

    <section>
        <h2>🔍 Key Analyses</h2>
        <div class="grid">
            <div class="card">
                <h3>1. Market Performance</h3>
                <p>📈 Identified top-performing markets (Brazil, Sweden, Finland)</p>
                <p>🔄 Calculated Gross Gaming Revenue (GGR) trends by country</p>
            </div>
            <div class="card">
                <h3>2. Player Analytics</h3>
                <p>🎯 Lifetime Value (LTV) modeling</p>
                <p>⚠️ Churn prediction with 85% accuracy</p>
                <p>📊 RFM segmentation (Recency-Frequency-Monetary)</p>
            </div>
            <div class="card">
                <h3>3. Operational Insights</h3>
                <p>💰 Bonus ROI analysis (Optimal bonus/deposit ratio: 20%)</p>
                <p>🔮 6-month revenue forecasting (ARIMA & Prophet)</p>
            </div>
        </div>
    </section>

    <section>
        <h2>🚀 How to Use</h2>
        <ol>
            <li><strong>Data Preparation</strong><br>
                Run <code>notebooks/1_Data_Cleaning.ipynb</code> to process raw CSVs</li>
            <li><strong>Exploratory Analysis</strong><br>
                Execute <code>notebooks/2_EDA.ipynb</code> for market comparisons</li>
            <li><strong>Advanced Modeling</strong><br>
                Use <code>notebooks/3_Advanced_Analytics.ipynb</code> for:
                <ul>
                    <li>Player churn prediction</li>
                    <li>GGR forecasting</li>
                    <li>Bonus effectiveness tests</li>
                </ul>
            </li>
        </ol>
    </section>

    <section>
        <h2>📊 Sample Outputs</h2>
        <div class="visualization">
            <h3>Market GGR Comparison</h3>
            <img src="reports/market_ggr.png" alt="Top Markets by GGR">
        </div>
        <div class="visualization">
            <h3>Revenue Forecast</h3>
            <img src="reports/ggr_forecast.png" alt="GGR Forecast">
        </div>
        <div class="visualization">
            <h3>Player Segments</h3>
            <img src="reports/player_segments.png" alt="RFM Segmentation">
        </div>
    </section>

    <section>
        <h2>🤝 Contributing</h2>
        <ol>
            <li>Fork the project</li>
            <li>Create a feature branch (<code>git checkout -b feature/your-feature</code>)</li>
            <li>Commit changes (<code>git commit -m 'Add some feature'</code>)</li>
            <li>Push to the branch (<code>git push origin feature/your-feature</code>)</li>
            <li>Open a Pull Request</li>
        </ol>
    </section>

    <section>
        <h2>📜 License</h2>
        <p>MIT License - see <a href="LICENSE">LICENSE</a> for details.</p>
    </section>

    <section>
        <h2>📬 Contact</h2>
        <p><strong>Your Name</strong><br>
        📧 <a href="mailto:your.email@example.com">your.email@example.com</a><br>
        🔗 <a href="https://linkedin.com/in/yourprofile" target="_blank">LinkedIn</a></p>
    </section>

    <footer>
        <p><strong>🎲 Ready to level up your casino analytics?</strong> Clone and start exploring!</p>
        <p><em>💡 Pro Tip: Check the notebooks' interactive visualizations with Jupyter Lab!</em></p>
        <pre><code>jupyter lab</code></pre>
    </footer>
</body>
</html>
