import matplotlib.pyplot as plt
import seaborn as sns

def explicar_aposta(market, prob_over, prob_under, team_a, team_b, recent_games_a, recent_games_b, h2h_stats_a, h2h_stats_b, home_away_stats_a, home_away_stats_b):
    """
    Retorna uma explicação detalhada sobre a recomendação de aposta considerando fatores como força ofensiva,
    defesa, forma recente, histórico de confrontos diretos (H2H), impacto do local (casa/fora) e visualizações gráficas.
    """
    # Explicação inicial baseada nas probabilidades
    if prob_over > prob_under:
        recomendacao = f"🔍 Recomendamos o mercado *Over {market}* com base na alta probabilidade de {prob_over:.2f}%. "
        recomendacao += f"Isso reflete um jogo com forte tendência ofensiva."
    else:
        recomendacao = f"🔍 Recomendamos o mercado *Under {market}* com base na probabilidade de {prob_under:.2f}%. "
        recomendacao += f"Isso reflete uma tendência de jogo mais defensivo."
    
    # Analisando o desempenho ofensivo e defensivo dos times
    ofensiva_a = team_a.average_goals_scored()
    ofensiva_b = team_b.average_goals_scored()
    defensiva_a = team_a.average_goals_conceded()
    defensiva_b = team_b.average_goals_conceded()
    
    if ofensiva_a > ofensiva_b and defensiva_b < defensiva_a:
        recomendacao += f"\n{team_a.name} tem uma ofensiva mais forte que {team_b.name}, e a defesa de {team_b.name} apresenta vulnerabilidades."
    elif ofensiva_b > ofensiva_a and defensiva_a < defensiva_b:
        recomendacao += f"\n{team_b.name} tem uma ofensiva mais forte que {team_a.name}, e a defesa de {team_a.name} apresenta vulnerabilidades."
    else:
        recomendacao += f"\nAmbos os times apresentam forças ofensivas semelhantes e defesas igualmente vulneráveis."

    # Considerando o desempenho recente dos times
    media_gols_a = sum(recent_games_a) / len(recent_games_a)
    media_gols_b = sum(recent_games_b) / len(recent_games_b)
    
    if media_gols_a > media_gols_b:
        recomendacao += f"\n{team_a.name} está em melhor forma ofensiva recentemente, com uma média de {media_gols_a:.2f} gols nos últimos jogos."
    elif media_gols_b > media_gols_a:
        recomendacao += f"\n{team_b.name} está em melhor forma ofensiva recentemente, com uma média de {media_gols_b:.2f} gols nos últimos jogos."
    else:
        recomendacao += f"\nAmbos os times têm uma forma ofensiva similar nos últimos jogos, com médias de {media_gols_a:.2f} gols para {team_a.name} e {media_gols_b:.2f} gols para {team_b.name}."
    
    # Analisando o histórico de confrontos diretos (H2H)
    h2h_result_a = h2h_stats_a.get('vitorias', 0)  # Exemplo: número de vitórias de A em H2H
    h2h_result_b = h2h_stats_b.get('vitorias', 0)  # Exemplo: número de vitórias de B em H2H
    recomendacao += f"\n🛑 No histórico de confrontos diretos (H2H), {team_a.name} tem {h2h_result_a} vitórias, enquanto {team_b.name} tem {h2h_result_b} vitórias."

    # Analisando o impacto de jogar em casa ou fora
    home_performance_a = home_away_stats_a['home']
    home_performance_b = home_away_stats_b['away']
    
    if home_performance_a > home_performance_b:
        recomendacao += f"\n{team_a.name} tem um desempenho melhor em casa, enquanto {team_b.name} tem um desempenho superior como visitante."
    else:
        recomendacao += f"\n{team_b.name} tem um desempenho melhor em casa ou como visitante, dependendo da partida."

    # Fornecendo mais contexto sobre a competição ou tipo de jogo
    recomendacao += "\nEste é um jogo importante na competição, o que pode influenciar a estratégia das equipes."

    # Determinando o nível de confiança
    diff_prob = abs(prob_over - prob_under)
    if diff_prob < 10:
        recomendacao += "\n🔶 A diferença nas probabilidades não é tão grande, então esta aposta pode ter um grau de risco."
    else:
        recomendacao += "\n✅ A diferença nas probabilidades é considerável, o que indica uma recomendação mais confiável."

    # Gerando visualizações gráficas
    gerar_graficos(team_a, team_b, prob_over, prob_under, recent_games_a, recent_games_b, h2h_stats_a, h2h_stats_b)

    return recomendacao


def gerar_graficos(team_a, team_b, prob_over, prob_under, recent_games_a, recent_games_b, h2h_stats_a, h2h_stats_b):
    """
    Gera gráficos de comparação entre os times, incluindo probabilidades Over/Under, desempenho ofensivo/defensivo e histórico de H2H.
    """
    # Gráfico de probabilidades Over/Under
    plt.figure(figsize=(10, 6))
    sns.barplot(x=["Over", "Under"], y=[prob_over, prob_under], palette="viridis")
    plt.title(f"Probabilidade de Mercado: Over/Under para {team_a.name} vs {team_b.name}")
    plt.ylabel("Probabilidade (%)")
    plt.show()

    # Gráfico de desempenho ofensivo/defensivo
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([team_a.name, team_b.name], [team_a.average_goals_scored(), team_b.average_goals_scored()], color='green', label='Ofensivo')
    ax.barh([team_a.name, team_b.name], [team_a.average_goals_conceded(), team_b.average_goals_conceded()], color='red', label='Defensivo')
    plt.title("Desempenho Ofensivo e Defensivo")
    plt.xlabel("Média de Gols")
    plt.legend()
    plt.show()

    # Gráfico de comparação de H2H
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([team_a.name, team_b.name], [h2h_stats_a.get('vitorias', 0), h2h_stats_b.get('vitorias', 0)], color='blue')
    plt.title(f"Histórico de Confrontos Diretos (H2H): {team_a.name} vs {team_b.name}")
    plt.ylabel("Número de Vitórias")
    plt.show()

    # Gráfico de desempenho recente (últimos jogos)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(recent_games_a, label=f'{team_a.name} - Últimos Jogos', marker='o')
    ax.plot(recent_games_b, label=f'{team_b.name} - Últimos Jogos', marker='o')
    plt.title("Desempenho Recente - Últimos 5 Jogos")
    plt.xlabel("Jogo")
    plt.ylabel("Gols Marcados")
    plt.legend()
    plt.show()