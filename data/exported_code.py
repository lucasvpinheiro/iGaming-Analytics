 1/1: import pandas as pd
 2/1:
import panda as pd

df = pd.read_excel("data/Category_Icons.xlsx","data/Rating_Icon.xlsx", engine="openpylx")
 2/2: import pandas as pd
 2/3:
import pandas as pd

df = pd.read_excel("data/Category_Icons.xlsx","data/Rating_Icon.xlsx", engine="openpylx")
 3/1: pip install openpyxl
 3/2: python.exe -m pip install --upgrade pip
 3/3: pip --upgrade pip
 4/1:
import pandas as pd

df = pd.read_excel("data/Category_Icons.xlsx","data/Rating_Icon.xlsx", engine="openpylx"), engine="openpyxl")
 4/2: import pandas as pd
 4/3:
import pandas as pd

df = pd.read_excel("data/Category_Icons.xlsx","data/Rating_Icon.xlsx", engine="openpylx")
 4/4: pip install openpylx
 4/5: import pandas as pd
 4/6:
import pandas as pd

df = read.pd_excel(data/Category_Icons.xlsx","data/Rating_Icon.xlsx", engine="openpylx")
 4/7: import pandas as pd
 4/8:
import pandas as pd

df = read.pd_excel("data/Category_Icons.xlsx","data/Rating_Icon.xlsx", engine="openpylx")
 4/9:
import pandas as pd

df = pd.read_excel("data/Category_Icons.xlsx","data/Rating_Icon.xlsx", engine="openpylx")
 5/1:
import pandas as pd

df = pd.read_excel("data/Category_Icons.xlsx","data/Rating_Icon.xlsx", engine="openpylx")
 6/1:
import pandas as pd

df.info()      # Check data types and missing values
df.describe()  # Summary statistics
df.isnull().sum()  # Check for missing values
 6/2:
import pandas as pd

df = pd.read.csv('data/blinkit_customer_feedback.csv', 'data/blinkit_customers.csv', 'data/blinkit_delivery_performance.csv', 'data/blinkit_inventory.csv', 'data/blinkit_inventoryNew.csv', 'data/blinkit_marketing_performance.csv', 'data/blinkit_order_items.csv', 'data/blinkit_orders.csv', 'data/blinkit_products.csv', 'data/Category_Icons.csv', 'data/Rating_Icon.csv')

df.head()

df.info()      # Check data types and missing values
df.describe()  # Summary statistics
df.isnull().sum()  # Check for missing values
 6/3:
import pandas as pd

df = pd.read_csv('data/blinkit_customer_feedback.csv', 'data/blinkit_customers.csv', 'data/blinkit_delivery_performance.csv', 'data/blinkit_inventory.csv', 'data/blinkit_inventoryNew.csv', 'data/blinkit_marketing_performance.csv', 'data/blinkit_order_items.csv', 'data/blinkit_orders.csv', 'data/blinkit_products.csv', 'data/Category_Icons.csv', 'data/Rating_Icon.csv')

df.head()

df.info()      # Check data types and missing values
df.describe()  # Summary statistics
df.isnull().sum()  # Check for missing values
 7/1:
import pandas as pd
import os

# Specify the directory where your CSV files are located
directory = 'data'

# List all files in the directory
all_files = os.listdir(directory)

# Filter out only CSV files
csv_files = [file for file in all_files if file.endswith('.csv')]
 7/2:
import pandas as pd
import os

# Specify the directory where your CSV files are located
directory = 'e-commerce-data-analyst/data/'

# List all files in the directory
all_files = os.listdir(directory)

# Filter out only CSV files
csv_files = [file for file in all_files if file.endswith('.csv')]
 7/3:
import pandas as pd
import os

# Specify the directory where your CSV files are located
directory = '/data/'

# List all files in the directory
all_files = os.listdir(directory)

# Filter out only CSV files
csv_files = [file for file in all_files if file.endswith('.csv')]
 7/4:
import pandas as pd
import os

# Specify the directory where your CSV files are located
directory = 'data/'

# List all files in the directory
all_files = os.listdir(directory)

# Filter out only CSV files
csv_files = [file for file in all_files if file.endswith('.csv')]
 8/1:
import pandas as pd
import os

# Specify the directory where your CSV files are located
directory = 'http://localhost:8888/tree/e-commerce-data-analyst/data'

# List all files in the directory
all_files = os.listdir(directory)

# Filter out only CSV files
csv_files = [file for file in all_files if file.endswith('.csv')]
 8/2:
import pandas as pd
import os

# Specify the directory where your CSV files are located
directory = 'tree/e-commerce-data-analyst/data'

# List all files in the directory
all_files = os.listdir(directory)

# Filter out only CSV files
csv_files = [file for file in all_files if file.endswith('.csv')]
 8/3:
import pandas as pd
import os

# Specify the directory where your CSV files are located
directory = './e-commerce-data-analyst/data'

# List all files in the directory
all_files = os.listdir(directory)

# Filter out only CSV files
csv_files = [file for file in all_files if file.endswith('.csv')]
 8/4:
import pandas as pd
import os

# Specify the directory where your CSV files are located
directory = 'C:\Users\lucas\OneDrive\Documentos\Git Project\e-commerce-data-analyst-1\e-commerce-data-analyst\data'

# List all files in the directory
all_files = os.listdir(directory)

# Filter out only CSV files
csv_files = [file for file in all_files if file.endswith('.csv')]
 9/1:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the orders dataset
orders = pd.read_csv('data/blinkit_orders.csv')

# Load the order items dataset
order_items = pd.read_csv('data/blinkit_order_items.csv')

# Load the customers dataset
customers = pd.read_csv('data/blinkit_customers.csv')

# Load the products dataset
products = pd.read_csv('data/blinkit_products.csv')

print(orders.head())
print(order_items.head())
print(customers.head())
print(products.head())
10/1:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
10/2:
# Load orders dataset
orders = pd.read_csv('data/blinkit_orders.csv')

# Load order items dataset
order_items = pd.read_csv('data/blinkit_order_items.csv')

# Load customers dataset
customers = pd.read_csv('data/blinkit_customers.csv')

# Load products dataset
products = pd.read_csv('data/blinkit_products.csv')
10/3:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load orders dataset
orders = pd.read_csv('data/blinkit_orders.csv')

# Load order items dataset
order_items = pd.read_csv('data/blinkit_order_items.csv')

# Load customers dataset
customers = pd.read_csv('data/blinkit_customers.csv')

# Load products dataset
products = pd.read_csv('data/blinkit_products.csv')
11/1:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load orders dataset
orders = pd.read_csv('data/blinkit_orders.csv')

# Load order items dataset
order_items = pd.read_csv('data/blinkit_order_items.csv')

# Load customers dataset
customers = pd.read_csv('data/blinkit_customers.csv')

# Load products dataset
products = pd.read_csv('data/blinkit_products.csv')
11/2:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load orders dataset
orders = pd.read_csv("data/blinkit_orders.csv")

# Load order items dataset
order_items = pd.read_csv("data/blinkit_order_items.csv")

# Load customers dataset
customers = pd.read_csv("data/blinkit_customers.csv")

# Load products dataset
products = pd.read_csv("data/blinkit_products.csv")
12/1:
import os
print(os.getcwd())
13/1:
import os
print(os.getcwd())
14/1:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
14/2:
# Load orders dataset
orders = pd.read_csv('data/blinkit_orders.csv')

# Load order items dataset
order_items = pd.read_csv('data/blinkit_order_items.csv')

# Load customers dataset
customers = pd.read_csv('data/blinkit_customers.csv')

# Load products dataset
products = pd.read_csv('data/blinkit_products.csv')
14/3:
# Display the first few rows of each dataset
print(orders.head())
print(order_items.head())
print(customers.head())
print(products.head())

# Check basic information
print(orders.info())
print(order_items.info())
print(customers.info())
print(products.info())

# Check for missing values
print(orders.isnull().sum())
print(order_items.isnull().sum())
print(customers.isnull().sum())
print(products.isnull().sum())
14/4:
# Drop rows with missing values in the orders dataset
orders_cleaned = orders.dropna()

# Fill missing values in the products dataset (e.g., fill missing product weights with the mean)
products['product_weight_g'].fillna(products['product_weight_g'].mean(), inplace=True)
14/5: import pandas as pd
14/6:
df["order_date"] = pd.to_datetime(df["order_date"])
df["actual_delivery_time"] = pd.to_datetime(df["actual_delivery_time"])
df["promised_delivery_time"] = pd.to_datetime(df["promised_delivery_time"])
14/7:
import pandas as pd

# Load the dataset
df = pd.read_csv("data/blinkit_orders.csv")

# Display the first rows to check the data
df.head()
14/8:
import pandas as pd

# Load the dataset
df = pd.read_csv("data/blinkit_orders.csv")

# Display the first rows to check the data
df.head()
14/9:
df["order_date"] = pd.to_datetime(df["order_date"])
df["actual_delivery_time"] = pd.to_datetime(df["actual_delivery_time"])
df["promised_delivery_time"] = pd.to_datetime(df["promised_delivery_time"])
14/10:
df["year"] = df["order_date"].dt.year
df["month"] = df["order_date"].dt.month
df["day_of_week"] = df["order_date"].dt.day_name()
df["hour"] = df["order_date"].dt.hour
14/11:
df["delivery_time_minutes"] = (df["actual_delivery_time"] - df["order_date"]).dt.total_seconds() / 60
df["on_time"] = df["delivery_status"].apply(lambda x: 1 if x == "On Time" else 0)
14/12: print("tudo certo")
14/13:
import matplotlib.pyplot as plt

df.groupby("month")["order_total"].sum().plot(kind="line", marker="o", figsize=(10, 5))
plt.title("Total Sales Over Time")
plt.xlabel("Month")
plt.ylabel("Total Revenue")
plt.show()
14/14:
on_time_rate = df["on_time"].mean() * 100
print(f"On-Time Delivery Rate: {on_time_rate:.2f}%")
14/15:
import seaborn as sns

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="payment_method", order=df["payment_method"].value_counts().index)
plt.title("Most Used Payment Methods")
plt.show()
15/1:
import pandas as pd

df = pd.read_csv("cassino_data.csv")
df.head()
16/1:
import pandas as pd
df = pd.read_csv("cassino_data.csv")
df.head()
17/1:
import pandas as pd

df = pd.read_csv("cassino_data.csv")
df.head()
17/2:
df.info()
df.describe()
df.isnull().sum()
17/3: df.fillna(method="ffill", inplace=True)
17/4: df.fill()
18/1:
import pandas as pd

df = pd.read_csv("cassino_data.csv")
df.head()
18/2:
df.info()
df.describe()
df.isnull().sum()
18/3: df = df.ffill()
18/4: df['registration_date'] = pd.to_datetime(df['registration_date'])
18/5: df.ffill(inplace=True)
18/6: print(df.columns)
19/1:
import pandas as pd

df = pd.read_csv("cassino_data.csv")
print(df.head())
print(df.info()) 
print(df.describe())
19/2: print(df.columns)
19/3: df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
19/4: df = df.ffill()
19/5: df.ffill(inplace=True)
19/6: df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
19/7: df['year'] = df['year'].astype(str).str.replace(',', '').astype(int)
19/8: df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
19/9: df = df.ffill()
19/10:
import matplotlib.pyplot as plt

df.groupby('date')['registrations'].sum().plot(figsize=(12, 6), marker='o')
plt.title("Total Registrations Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Registrations")
plt.show()
19/11: df['registrations'] = pd.to_numeric(df['registrations'], errors='coerce')
19/12:
df['registrations'].fillna(0, inplace=True)
df['registrations'] = df['registrations'].astype(int)
19/13:
import matplotlib.pyplot as plt

df.groupby('date')['registrations'].sum().plot(figsize=(12, 6), marker='o')
plt.title("Total Registrations Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Registrations")
plt.show()
19/14:
df.groupby('market')['ngr_eur'].sum().sort_values(ascending=False).head(10).plot(kind='bar', figsize=(12,6))
plt.title("Top 10 Markets by Net Gaming Revenue (NGR)")
plt.ylabel("NGR in EUR")
plt.show()
19/15: df['ngr_eur'] = df['ngr_eur'].astype(str).str.replace(',', '').astype(float)
19/16: df['ngr_eur'].fillna(0, inplace=True)
20/1:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better-looking plots
plt.style.use('seaborn')
%matplotlib inline
21/1:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')
%matplotlib inline
   1:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df_raw = pd.read_csv('cassino_data.csv')
df_raw.head()
   2:
def clean_numeric_column(column):
    if column.dtype == 'object':
        column = column.str.replace(' ', '').str.replace('"', '').str.replace(',', '')
        column = column.str.replace('- ', '-')
        column = pd.to_numeric(column, errors='coerce')
    return column

non_numeric_cols = ['year', 'month', 'site_id', 'market']
for col in df_raw.columns:
    if col not in non_numeric_cols:
        df_raw[col] = clean_numeric_column(df_raw[col])

df_raw['year'] = df_raw['year'].astype(str).str.replace(',', '').astype(int)

print(df_raw.dtypes)
   3:
financial_cols = [col for col in df_raw.columns if '_eur' in col]
df_raw[financial_cols] = df_raw[financial_cols].fillna(0)

count_cols = ['registrations', 'ftds', 'active_players', 'deposit_count', 'unique_depositors']
df_raw[count_cols] = df_raw[count_cols].fillna(0).astype(int)

print("Negative registrations:", df_raw[df_raw['registrations'] < 0].shape[0])
print("Negative active players:", df_raw[df_raw['active_players'] < 0].shape[0])
   4:
try:
    df_raw.groupby('market')['ngr_eur'].sum().sort_values(ascending=False).head(10).plot(kind='bar', figsize=(12,6))
    plt.title("Top 10 Markets by Net Gaming Revenue (NGR)")
    plt.ylabel("NGR in EUR")
    plt.show()
except TypeError:
    print("Error: Data still not numeric!")
else:
    print("✅ Success! Data is clean and numeric.")
   5:
inconsistent_rows = df_raw[abs(df_raw['ggr_eur'] - (df_raw['turnover_eur'] - df_raw['winnings_eur'])) > 0.01]
print("Rows with inconsistent GGR calculation:", len(inconsistent_rows))
   6:
df_raw['ggr_eur'] = df_raw['turnover_eur'] - df_raw['winnings_eur']

inconsistent_rows = df_raw[abs(df_raw['ggr_eur'] - (df_raw['turnover_eur'] - df_raw['winnings_eur'])) > 0.01]
print("Remaining inconsistent rows after correction:", len(inconsistent_rows))
   7:
df_raw['ggr_discrepancy'] = df_raw['ggr_eur'] - (df_raw['turnover_eur'] - df_raw['winnings_eur'])

discrepancies = df_raw[abs(df_raw['ggr_discrepancy']) > 0.01].sort_values('ggr_discrepancy', ascending=False)
print(f"Found {len(discrepancies)} rows with GGR discrepancies")
discrepancies[['market', 'year', 'month', 'turnover_eur', 'winnings_eur', 'ggr_eur', 'ggr_discrepancy']].head()
   8:
df_raw['ggr_calculated'] = df_raw['turnover_eur'] - df_raw['winnings_eur']
df_raw['ggr_consistent'] = abs(df_raw['ggr_eur'] - df_raw['ggr_calculated']) <= 0.01

print(f"Percentage of consistent GGR rows: {df_raw['ggr_consistent'].mean()*100:.2f}%")
   9: df_raw.to_csv('cleaned_casino_data.csv', index=False)
  10:
top_markets = df_raw.groupby('market').agg({
    'ggr_eur': 'sum',
    'active_players': 'sum',
    'deposits_eur': 'sum'
}).sort_values('ggr_eur', ascending=False).head(10)

plt.figure(figsize=(12,6))
sns.barplot(x=top_markets.index, y='ggr_eur', data=top_markets)
plt.title('Top 10 Markets by Gross Gaming Revenue (GGR)')
plt.ylabel('Total GGR (EUR)')
plt.xticks(rotation=45)
plt.show()
  11:
df_raw['date'] = pd.to_datetime(df_raw['year'].astype(str) + '-' + df_raw['month'].astype(str))
monthly_trend = df_raw.groupby('date')['ggr_eur'].sum()

plt.figure(figsize=(12,6))
monthly_trend.plot(marker='o')
plt.title('Monthly GGR Trend')
plt.ylabel('GGR (EUR)')
plt.grid(True)
plt.show()
  12:
product_cols = ['sports_turnover_eur', 'casino_turnover_eur', 'live_casino_turnover_eur']
product_mix = df_raw[product_cols].sum()

plt.figure(figsize=(8,8))
plt.pie(product_mix, labels=product_mix.index, autopct='%1.1f%%')
plt.title('Revenue Share by Product Type')
plt.show()
  13:
deposit_analysis = df_raw.groupby('market').agg({
    'deposits_eur': 'sum',
    'unique_depositors': 'sum',
    'deposit_count': 'sum'
})
deposit_analysis['avg_deposit'] = deposit_analysis['deposits_eur'] / deposit_analysis['deposit_count']
deposit_analysis['deposits_per_player'] = deposit_analysis['deposits_eur'] / deposit_analysis['unique_depositors']

deposit_analysis.sort_values('avg_deposit', ascending=False).head(5)
  14:
bonus_impact = df_raw.groupby('market').agg({
    'bonus_issued_eur': 'sum',
    'ggr_eur': 'sum'
})
bonus_impact['roi'] = bonus_impact['ggr_eur'] / bonus_impact['bonus_issued_eur']

plt.figure(figsize=(10,6))
sns.scatterplot(x='bonus_issued_eur', y='ggr_eur', hue=bonus_impact.index, 
                data=bonus_impact, s=100)
plt.title('Bonus ROI by Market')
plt.xlabel('Bonus Issued (EUR)')
plt.ylabel('GGR (EUR)')
plt.show()
  15:
df_raw['house_edge'] = df_raw['ggr_eur'] / df_raw['turnover_eur']
risk_analysis = df_raw.groupby('market')['house_edge'].mean().sort_values()

risk_analysis.plot(kind='barh', figsize=(10,6))
plt.title('Average House Edge by Market')
plt.xlabel('House Edge %')
plt.show()
  16:
player_metrics = df_raw.groupby('market').agg({
    'active_players': 'sum',
    'ggr_eur': 'sum',
    'unique_depositors': 'sum'
})

player_metrics['avg_ggr_per_player'] = player_metrics['ggr_eur'] / player_metrics['unique_depositors']
player_metrics['player_retention_rate'] = player_metrics['unique_depositors'] / player_metrics['active_players']
player_metrics['estimated_ltv'] = (player_metrics['avg_ggr_per_player'] * 
                                 (1 / (1 - player_metrics['player_retention_rate'])))

plt.figure(figsize=(12,6))
sns.barplot(x=player_metrics.index, y='estimated_ltv', data=player_metrics.sort_values('estimated_ltv', ascending=False))
plt.title('Estimated Player Lifetime Value by Market')
plt.ylabel('LTV (EUR)')
plt.xticks(rotation=45)
plt.show()
  17:
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df_raw['last_active_month'] = df_raw.groupby('market')['date'].transform('max')
df_raw['churn'] = (df_raw['date'] < df_raw['last_active_month'] - pd.DateOffset(months=1)).astype(int)

features = ['active_players', 'deposit_count', 'deposits_eur', 
            'withdrawals_eur', 'bonus_issued_eur', 'ngr_eur']
X = df_raw[features]
y = df_raw['churn']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

pd.Series(model.feature_importances_, index=features).sort_values().plot(kind='barh')
plt.title('Churn Prediction Feature Importance')
plt.show()
  18:
campaign_results = df_raw.groupby(['month', 'marketing_campaign']).agg({
    'registrations': 'sum',
    'ftds': 'sum',
    'deposits_eur': 'sum'
}).reset_index()

campaign_results['conversion_rate'] = campaign_results['ftds'] / campaign_results['registrations']
campaign_results['roas'] = campaign_results['deposits_eur'] / campaign_results['marketing_spend']  # Requires spend data

plt.figure(figsize=(12,6))
sns.lineplot(x='month', y='roas', hue='marketing_campaign', 
             data=campaign_results, marker='o')
plt.title('Campaign Return on Ad Spend Over Time')
plt.ylabel('ROAS (EUR)')
plt.show()
  19:
campaign_results = df_raw.groupby(['month', 'marketing_campaign']).agg({
    'registrations': 'sum',
    'ftds': 'sum',
    'deposits_eur': 'sum'
}).reset_index()

campaign_results['conversion_rate'] = campaign_results['ftds'] / campaign_results['registrations']
campaign_results['roas'] = campaign_results['deposits_eur'] / campaign_results['marketing_spend']  

plt.figure(figsize=(12,6))
sns.lineplot(x='month', y='roas', hue='marketing_campaign', 
             data=campaign_results, marker='o')
plt.title('Campaign Return on Ad Spend Over Time')
plt.ylabel('ROAS (EUR)')
plt.show()
  20:
from statsmodels.tsa.arima.model import ARIMA

monthly_ggr = df_raw.groupby('date')['ggr_eur'].sum().reset_index()
monthly_ggr = monthly_ggr.set_index('date').asfreq('MS')

model = ARIMA(monthly_ggr, order=(1,1,1))
results = model.fit()

forecast = results.get_forecast(steps=6)
forecast_index = pd.date_range(monthly_ggr.index[-1] + pd.DateOffset(months=1), periods=6, freq='MS')

plt.figure(figsize=(12,6))
plt.plot(monthly_ggr, label='Historical')
plt.plot(forecast_index, forecast.predicted_mean, label='Forecast', color='red')
plt.fill_between(forecast_index, 
                 forecast.conf_int()['lower ggr_eur'],
                 forecast.conf_int()['upper ggr_eur'],
                 color='pink', alpha=0.3)
plt.title('6-Month GGR Forecast')
plt.ylabel('GGR (EUR)')
plt.legend()
plt.show()
  21: from statsmodels.tsa.arima.model import ARIMA
  22:
import statsmodels
print(statsmodels.__version__)
  23:
from statsmodels.tsa.arima.model import ARIMA

monthly_ggr = df_raw.groupby('date')['ggr_eur'].sum().reset_index()
monthly_ggr = monthly_ggr.set_index('date').asfreq('MS')

model = ARIMA(monthly_ggr, order=(1,1,1))
results = model.fit()

forecast = results.get_forecast(steps=6)
forecast_index = pd.date_range(monthly_ggr.index[-1] + pd.DateOffset(months=1), periods=6, freq='MS')

plt.figure(figsize=(12,6))
plt.plot(monthly_ggr, label='Historical')
plt.plot(forecast_index, forecast.predicted_mean, label='Forecast', color='red')
plt.fill_between(forecast_index, 
                 forecast.conf_int()['lower ggr_eur'],
                 forecast.conf_int()['upper ggr_eur'],
                 color='pink', alpha=0.3)
plt.title('6-Month GGR Forecast')
plt.ylabel('GGR (EUR)')
plt.legend()
plt.show()
  24:
rfm_data = df_raw.groupby('player_id').agg({
    'date': 'max',  
    'deposit_count': 'sum',
    'deposits_eur': 'sum'
})

rfm_data['recency'] = (pd.to_datetime('today') - rfm_data['date']).dt.days
rfm_data = rfm_data.rename(columns={
    'deposit_count': 'frequency',
    'deposits_eur': 'monetary'
})

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_data[['recency','frequency','monetary']])

kmeans = KMeans(n_clusters=4)
rfm_data['segment'] = kmeans.fit_predict(rfm_scaled)

plt.figure(figsize=(10,6))
sns.scatterplot(x='frequency', y='monetary', hue='segment', 
                data=rfm_data, palette='viridis', size='recency')
plt.title('Player RFM Segmentation')
plt.show()
  25:
bonus_analysis = df_raw.groupby('player_id').agg({
    'bonus_issued_eur': 'sum',
    'ggr_eur': 'sum',
    'deposits_eur': 'sum'
})

bonus_analysis['bonus_roi'] = bonus_analysis['ggr_eur'] / bonus_analysis['bonus_issued_eur']
bonus_analysis['bonus_ratio'] = bonus_analysis['bonus_issued_eur'] / bonus_analysis['deposits_eur']

plt.figure(figsize=(10,6))
sns.scatterplot(x='bonus_ratio', y='bonus_roi', data=bonus_analysis)
plt.axvline(x=0.2, color='red', linestyle='--')  
plt.title('Bonus ROI vs. Bonus/Deposit Ratio')
plt.show()
  26: %history -g -f exported_code.py
