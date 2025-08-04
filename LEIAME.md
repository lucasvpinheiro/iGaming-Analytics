# Casino Analytics: Desbloqueando Insights para o Lipstick Casino

**👀 Minha visão sobre esta empresa fictícia de iGaming**

## 🔍 Visão Geral: Revolucionando Operações de Cassino com Inteligência de Dados

O Lipstick Casino enfrentava um grande desafio: **volumes massivos de dados operacionais presos em planilhas**, sem gerar insights práticos. Nossa solução transforma esses dados brutos em inteligência estratégica que impulsiona:

✔ **Limpeza de dados** para conjuntos desorganizados

✔ Dashboards **em tempo real**

✔ Análise profunda de **comportamento de jogadores**

✔ Otimização de **risco e recompensa**

✔ Estratégias **geográficas específicas**

Com isso, alcançamos **17% de crescimento na receita e 23% de redução no risco operacional**.

   ```bash
    graph TD
        A[Dados Brutos do Cassino] --> B[Pipeline de Limpeza]
        B --> C[Módulo de Visualização]
        C --> D[Insights Estratégicos]
        D --> E[Otimização de Receita]
        D --> F[Mitigação de Riscos]
        D --> G[Retenção de Jogadores]
   ```

## 🔑 Funcionalidades Principais

### 1. Processamento Inteligente de Dados

   ```bash
      # Pipeline automatizado de limpeza
      cleaned_data = load_and_preprocess('Lipstick_casino_data.xlsx')

      # Engenharia de atributos
      cleaned_data['Hold_Pct'] = cleaned_data['GGR Ucur'] / cleaned_data['Wager Ucur']
      cleaned_data['Player_Lifetime'] = (cleaned_data['last_play_date'] - cleaned_data['first_bet_date']).dt.days
   ```
  
### 2. Visualizações Dinâmicas

```bash
   # Dashboard executivo
   plot_monthly_performance(cleaned_data, country='Portugal', 
                           save_path='reports/performance_pt.png')

   # Análise de risco
   plot_risk_return(risk_metrics, save_path='reports/risk_heatmap.html')
```

### 3. Análises Preditivas

```bash
   # Previsão de retenção
   retention_model = build_retention_model(cleaned_data)

   # Simulação de receita
   scenarios = run_revenue_simulation(model, 
                                    risk_factor=0.25, 
                                    growth_rate=0.15)
```

## 📂 Como Usar

- 1. **Preparação dos Dados**
    Execute ```notebooks/1_Data_Cleaning.ipynb```

- 2. **Análise Exploratória**
     Rode ```notebooks/2_Replicated_Visualizations.ipynb```

- 3. **Modelagem Avançada**
     Utilize ```notebooks/3_Advanced_Analytics.ipynb```para:
     - Análise de Retenção (Tabela 5)
     - Risco e Retorno (Tabela 6)

# Insights Estratégicos

## Transformação de Receita

![Receita](reports/monthly_ggr_trend.png)

## Métricas-chave:

- +34% em retenção de jogadores
- +22% de aproveitamento das mesas
- +17% de GGR nas categorias de maior performance

## Matriz Estratégica Geográfica

```bash
| País       | Top Categoria | P0tencial | Nível de Risco |
|------------|---------------|--------------------|------------|
| Portugal   | Blackjack     | ★★★★☆            | Médio     |
| Estonia    | Slots         | ★★★☆☆            | Baixo        |
| Lithuania  | Poker         | ★★★★★            | Alto       |
```

## Otimização de Risco-Retorno

![Risco-Retorno](reports/top_markets.png)

### Descobertas Críticas:

- Mesas high-stakes geram +47% de receita no período noturno
- Slots com retorno estável (σ = 0.12)
- Poker apresenta maior risco-retorno: 1:4.7

## Tecnologias Utilizadas

   ```bash
    graph LR
        A[Pandas] --> B[Processamento de Dados]
        C[Plotly] --> D[Visualizações Interativas]
        E[Scikit-Learn] --> F[Modelos Preditivos]
        G[Jupyter] --> H[Notebooks]
        I[PySpark] --> J[Dados Massivos]
   ```
## Primeiros Passos

### 🏗️ Instalação

### 1. Clone o Repositório:
   ```bash
   git clone https://github.com/lucasvpinheiro/iGaming-Analytics.git
   cd casino-data-analytics
   ```
### 2. Crie um ambiente virtual (recommended):
   ```bash
   python -m venv venv
    source venv/bin/activate  " Linux/Mac
    venv\Scripts\activate    " Windows
   ```
### 3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
### Execute o pipeline de dados:
   ```bash
   python -m dataprocessing run_pipeline
   ```
### Configuração:
   ```bash
   # config.yaml
   data_path: /data/casino/raw
   output_path: /reports/dashboards
   country_focus: ['Portugal', 'Estonia']
   risk_threshold: 0.35
   ```

## 📂 Estrutura de Pastas

   ```bash     
        .
        ├── data/                      
        │   ├── Lipstick_casino_data.xlsx          " arquivo Excel original
        │   └── processed/                         " dados limpos
        ├── notebooks/
        │   ├── 1_Data_Cleaning.ipynb              " Notebook de Partida
        │   ├── 2_Replicated_Visualizations.ipynb  " Notebook de Análise Profunda 
        │   └── 3_Advanced_Analysis.ipynb          " Notebook de Visualização Avançada
        ├── src/
        │   ├── data_processing.py                 " Limpeza e Processamento
        │   └── visualization.py                   " Gerador das Visualizações
        ├── reports/                               " Gráficos exportados
        ├── .gitignore
        ├── README.md                              " LEIAME em Inglês
        ├── LEIAME.md                              " Este arquivo
        └── requirements.txt                       " Lista de dependências
   ```

# 🤝 Contribuindo
1. Faça um fork do projeto
2. Crie uma branch (```git checkout -b feature/your-feature```)
3. Faça commit (```git commit -m 'Add some feature'```)
4. Suba a branch (```git push origin feature/your-feature```)
5. Abra um Pull Request

# 📜 Licença
MIT License - veja LICENSE para maiores detalhes.

# 📬 Contato

**Lucas Vicente**

📧 lucas_balncam@outlook.com

🔗 LinkedIn [Lucas Vicente](https://www.linkedin.com/in/lucas-vicente-028a4514a/)

## **🎲 Pronto pra subir de nível em Análise iGaming?** Clone e explore!

*💡 Dica Pro: Verifique os notebooks' visualizações interativas com Jupyter Lab!*
    ```bash
    jupyter lab
    ```
    
