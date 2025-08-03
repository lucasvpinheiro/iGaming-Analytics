import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess(file_path):
    """
    Uploads and pre-processes raw casino data (en)
    Carrega e pré-processa os dados brutos do casino (pt-br)
    """
    # Load Excel data | Carregar dados do Excel
    df = pd.read_excel(file_path, sheet_name='Casino Data')
    
    # Convert date column | Converter coluna de data
    df['Month, Year'] = pd.to_datetime(df['Month, Year'])
    
    # Create new temporal features | Criar novas features temporais
    df['Year'] = df['Month, Year'].dt.year
    df['Quarter'] = df['Month, Year'].dt.quarter
    df['Month_Name'] = df['Month, Year'].dt.month_name()
    
    # Normalize categories | Normalizar categorias
    df['Game category'] = df['Game category'].str.title().str.strip()
    df['Country'] = df['Country'].str.title().str.strip()
    
    # Calculate Hold % (GGR / Bet Volume) | Calcular Hold % (GGR / Volume de Apostas)
    df['Hold_Pct'] = df['GGR Ucur'] / df['Wager Ucur']
    df['Hold_Pct'] = df['Hold_Pct'].replace([np.inf, -np.inf], np.nan)
    
    # Handling extreme negative values | Tratar valores negativos extremos
    df.loc[df['GGR Ucur'] < -10000, 'GGR Ucur'] = np.nan
    
    # Consolidate similar table types | Consolidar tipos de mesa similares
    df['Table_Type_Simplified'] = np.where(
        df['Table Type Commercial'].str.contains('High Stakes', case=False),
        'High Stakes',
        'Regular'
    )
    
    return df

def calculate_player_metrics(df):
    """
    Calculates aggregate player metrics by period (en)
    Calcula métricas agregadas de jogadores por período (pt-br)
    """
    # New players (first_bet_date in the current month) | Novos jogadores (first_bet_date no mês atual)
    new_players = df.groupby(['Month, Year', 'Country'])['Player Game Count'].sum().reset_index()
    new_players.rename(columns={'Player Game Count': 'New_Players'}, inplace=True)
    
    # Active players | Jogadores ativos
    active_players = df.groupby(['Month, Year', 'Country'])['Bet Spot Count'].sum().reset_index()
    active_players.rename(columns={'Bet Spot Count': 'Active_Players'}, inplace=True)
    
    # Combine metrics | Combinar métricas
    player_metrics = pd.merge(new_players, active_players, on=['Month, Year', 'Country'])
    return player_metrics

def prepare_blackjack_data(df):
    """
    Prepares specific data for Blackjack analysis (en)
    Prepara dados específicos para análise de Blackjack (pt-br)
    """
    blackjack = df[df['Game category'] == 'Blackjack'].copy()
    
    # Simulate time data | Simular dados de horário
    np.random.seed(42)
    blackjack['Hour'] = np.random.randint(0, 24, size=len(blackjack))
    
    # Classify as morning (00-12h) or afternoon (12-24h) | Classificar como manhã (00-12h) ou tarde (12-24h)
    blackjack['Time_Period'] = np.where(
        blackjack['Hour'] < 12, 
        'Morning (00:00-11:59)', 
        'Evening (12:00-23:59)'
    )
    
    return blackjack[['Table_Type_Simplified', 'Time_Period', 'Wager Ucur']]

def calculate_risk_metrics(df):
    """
    Calculates risk metrics by game category (en)
    Calcula métricas de risco por categoria de jogo (pt-br)
    """
    risk_df = df.groupby('Game category').agg(
        Avg_GGR=('GGR Ucur', 'mean'),
        Std_GGR=('GGR Ucur', 'std'),
        Total_Wager=('Wager Ucur', 'sum'),
        Player_Count=('Player Game Count', 'sum')
    ).reset_index()
    
    risk_df['Risk_Ratio'] = risk_df['Std_GGR'] / risk_df['Avg_GGR'].abs()
    return risk_df.sort_values('Risk_Ratio', ascending=False)