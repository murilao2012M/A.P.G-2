def explicar_aposta(market, prob_over, prob_under, team_a, team_b, recent_games_a, recent_games_b):
    """
    Retorna uma explicação detalhada sobre a recomendação de aposta considerando fatores como força ofensiva,
    defesa, forma recente e a diferença de probabilidade entre Over e Under.
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
    
    # Fornecendo mais contexto sobre a competição ou tipo de jogo
    recomendacao += "\nEste é um jogo importante na competição, o que pode influenciar a estratégia das equipes."

    # Determinando o nível de confiança
    diff_prob = abs(prob_over - prob_under)
    if diff_prob < 10:
        recomendacao += "\n🔶 A diferença nas probabilidades não é tão grande, então esta aposta pode ter um grau de risco."
    else:
        recomendacao += "\n✅ A diferença nas probabilidades é considerável, o que indica uma recomendação mais confiável."

    return recomendacao