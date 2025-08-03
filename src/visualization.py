import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
from matplotlib.ticker import FuncFormatter

# Style settings | Configurações de estilo
sns.set(style="whitegrid", palette="pastel")
plt.rcParams['font.family'] = 'DejaVu Sans'

def format_currency(x, pos):
    """Function for formatting monetary values (en)
    Função para formatação de valores monetários (pt-br)"""
    return f'€{x/1000:,.0f}K' if abs(x) >= 1000 else f'€{x:,.0f}'

def plot_monthly_performance(df, country=None, save_path=None):
    """
    Plot monthly performance: new players vs GGR (en)

        Args:
        df: DataFrame with processed data
        country: Country to filter (None for all)
        save_path: Path to save the graph

    Plota desempenho mensal: novos jogadores vs GGR (pt-br)
    
    Args:
        df: DataFrame com dados processados
        country: País para filtrar (None para todos)
        save_path: Caminho para salvar o gráfico
    """
    if country:
        df = df[df['Country'] == country]
        title = f'Monthly Performance - {country}'
    else:
        title = 'Global Monthly Performance'
    
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Plot of new players | Plotagem de novos jogadores
    sns.lineplot(data=df, x='Month, Year', y='New_Players', 
                 ax=ax1, color='teal', marker='o', label='New Players')
    ax1.set_ylabel('New Players', color='teal')
    ax1.tick_params(axis='y', labelcolor='teal')
    
    # Plot of GGR | Plot de GGR
    ax2 = ax1.twinx()
    sns.lineplot(data=df, x='Month, Year', y='GGR Ucur', 
                 ax=ax2, color='coral', marker='s', label='GGR')
    ax2.set_ylabel('Gross Gaming Revenue', color='coral')
    ax2.tick_params(axis='y', labelcolor='coral')
    ax2.yaxis.set_major_formatter(FuncFormatter(format_currency))
    
    plt.title(title, fontsize=16)
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_quarterly_change(df, metric='GGR Ucur', save_path=None):
    """
    Plot quarterly variation with positive/negative highlights (en)
    
    Args:
        df: DataFrame with processed data
        metric: Metric for analysis (‘GGR Ucur’ or ‘Wager Ucur’)
        save_path: Path to save the graph

    Plota variação trimestral com destaque positivo/negativo (pt-br)
    
    Args:
        df: DataFrame com dados processados
        metric: Métrica para análise ('GGR Ucur' ou 'Wager Ucur')
        save_path: Caminho para salvar o gráfico
    """
    quarterly = df.groupby(['Year', 'Quarter'])[metric].sum().reset_index()
    quarterly['Pct_Change'] = quarterly[metric].pct_change() * 100
    
    plt.figure(figsize=(12, 6))
    
    # Create colored bars by performance | Criar barras coloridas por desempenho
    colors = ['green' if x > 0 else 'red' for x in quarterly['Pct_Change']]
    bars = plt.bar(
        quarterly['Year'].astype(str) + 'Q' + quarterly['Quarter'].astype(str),
        quarterly['Pct_Change'],
        color=colors
    )
    
    # Add labels | Adicionar rótulos
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., 
                 height + (1 if height > 0 else -3), 
                 f'{height:.1f}%', 
                 ha='center', 
                 va='bottom' if height > 0 else 'top',
                 color='black')
    
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title(f'Quarterly % Change - {metric}', fontsize=14)
    plt.ylabel('Percentage Change (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_blackjack_time_split(blackjack_df, save_path=None):
    """
    Plot Blackjack distribution by time of day (en)
    
    Args:
        blackjack_df: DataFrame prepared for Blackjack
        save_path: Path to save the graph

    Plota distribuição de Blackjack por período do dia (pt-br)
    
    Args:
        blackjack_df: DataFrame preparado para Blackjack
        save_path: Caminho para salvar o gráfico
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    # Prepare data | Preparar dados
    time_split = blackjack_df.groupby(['Time_Period', 'Table_Type_Simplified']).size().unstack()
    
    # Morning plot | Plot de manhã
    time_split.loc['Morning (00:00-11:59)'].plot.pie(
        ax=axes[0], 
        autopct='%1.1f%%', 
        startangle=90,
        colors=['lightblue', 'lightgreen']
    )
    axes[0].set_title('Morning Distribution', fontsize=14)
    axes[0].set_ylabel('')
    
    # Afternoon plot | Plot de tarde
    time_split.loc['Evening (12:00-23:59)'].plot.pie(
        ax=axes[1], 
        autopct='%1.1f%%', 
        startangle=90,
        colors=['lightblue', 'lightgreen']
    )
    axes[1].set_title('Evening Distribution', fontsize=14)
    axes[1].set_ylabel('')
    
    plt.suptitle('Blackjack Table Distribution by Time Period', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_country_category_rankings(df, save_path=None):
    """
    Plot category rankings by country using Plotly. (en)
    
    Args:
        df: DataFrame with processed data.
        save_path: Path to save the graph.

    Plota ranking de categorias por país usando Plotly (pt-br)
    
    Args:
        df: DataFrame com dados processados
        save_path: Caminho para salvar o gráfico
    """
    # Calculate GGR by country and category | Calcular GGR por país e categoria
    country_cat = df.groupby(['Country', 'Game category'])['GGR Ucur'].sum().reset_index()
    
    # Create ranking | Criar ranking
    country_cat['Rank'] = country_cat.groupby('Country')['GGR Ucur'].rank(ascending=False)
    
    # Filter top 3 by country | Filtrar top 3 por país
    top_categories = country_cat[country_cat['Rank'] <= 3]
    
    # Interactive plot | Plot interativo
    fig = px.treemap(
        top_categories,
        path=['Country', 'Game category'],
        values='GGR Ucur',
        color='Rank',
        color_continuous_scale='Blues',
        title='Top Game Categories by Country (GGR Ranking)'
    )
    
    fig.update_traces(
        textinfo="label+value+percent parent",
        hovertemplate="<b>%{label}</b><br>GGR: €%{value:,.0f}<br>Rank: %{color}"
    )
    
    if save_path:
        fig.write_html(save_path)
    fig.show()

def plot_risk_return(risk_df, save_path=None):
    """
    Plot risk-return analysis by game category (en)
    
    Args:
        risk_df: DataFrame with risk metrics
        save_path: Path to save the graph

    Plota análise risco-retorno por categoria de jogo (pt-br)
    
    Args:
        risk_df: DataFrame com métricas de risco
        save_path: Caminho para salvar o gráfico
    """
    plt.figure(figsize=(14, 8))
    
    # Bubble plot | Plot de bolhas
    scatter = sns.scatterplot(
        data=risk_df,
        x='Avg_GGR',
        y='Risk_Ratio',
        size='Total_Wager',
        sizes=(50, 500),
        hue='Game category',
        alpha=0.8,
        palette='viridis'
    )
    
    # Add labels | Adicionar rótulos
    for i, row in risk_df.iterrows():
        plt.text(
            row['Avg_GGR'] * 1.05, 
            row['Risk_Ratio'] * 1.02, 
            row['Game category'],
            fontsize=9
        )
    
    # Medium risk line | Linha de risco médio
    mean_risk = risk_df['Risk_Ratio'].mean()
    plt.axhline(mean_risk, color='red', linestyle='--', alpha=0.7)
    plt.text(
        risk_df['Avg_GGR'].max() * 0.7, 
        mean_risk * 1.05, 
        f'Average Risk: {mean_risk:.2f}',
        color='red'
    )
    
    # Formatting | Formatação
    plt.title('Risk-Return Analysis by Game Category', fontsize=16)
    plt.xlabel('Average Gross Gaming Revenue (€)')
    plt.ylabel('Risk Ratio (Std Dev / Avg GGR)')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend(title='Game Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_hold_performance(df, country_param=None, save_path=None):
    """
    Plot Hold % performance with country parameter (en)
    
    Args:
        df: DataFrame with processed data
        country_param: Country to filter (None for global)
        save_path: Path to save the graph

    Plota desempenho do Hold % com parâmetro de país (pt-br)
    
    Args:
        df: DataFrame com dados processados
        country_param: País para filtrar (None para global)
        save_path: Caminho para salvar o gráfico
    """
    if country_param:
        df = df[df['Country'] == country_param]
        title = f'Hold % Performance - {country_param}'
    else:
        title = 'Global Hold % Performance'
    
    # Calculate monthly averages | Calcular médias mensais
    monthly_hold = df.groupby(['Month, Year', 'Game category'])['Hold_Pct'].mean().reset_index()
    
    plt.figure(figsize=(14, 8))
    
    # Plot lines by category | Plot de linhas por categoria
    sns.lineplot(
        data=monthly_hold,
        x='Month, Year',
        y='Hold_Pct',
        hue='Game category',
        style='Game category',
        markers=True,
        dashes=False,
        markersize=8,
        linewidth=2.5
    )
    
    # Reference line | Linha de referência
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    
    # Formatting | Formatação
    plt.title(title, fontsize=16)
    plt.ylabel('Hold Percentage (%)')
    plt.xlabel('')
    plt.legend(title='Game Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.2)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()