def get_best_market(probabilities, avg_goals_a, avg_goals_b, recent_form_a=None, recent_form_b=None, h2h_stats=None, home_away_stats_a=None, home_away_stats_b=None):
    """
    Identifica o melhor mercado (Over/Under) com base nas probabilidades calculadas e nos desempenhos dos times.
    """
    best_market = None
    best_weight = 0
    best_confidence = 0
    confidence_threshold = 0.6  # Limiar de confiança mínima para considerar um mercado viável

    # Condicional para ajustar os mercados a verificar com base no desempenho ofensivo
    if avg_goals_a > 2.5 and avg_goals_b > 2.5:
        markets_to_check = [3.5, 4.5]
    else:
        markets_to_check = [1.5, 2.5]

    # Ajuste para desempenho recente (Se disponível)
    if recent_form_a and recent_form_b:
        avg_recent_a = sum(recent_form_a) / len(recent_form_a)
        avg_recent_b = sum(recent_form_b) / len(recent_form_b)
        if avg_recent_a > avg_goals_a:
            avg_goals_a = avg_recent_a  # Ajustar se o time estiver melhorando
        if avg_recent_b > avg_goals_b:
            avg_goals_b = avg_recent_b  # Ajustar se o time estiver melhorando

    # Ajuste para o impacto de jogar em casa/fora
    if home_away_stats_a and home_away_stats_b:
        home_factor_a = home_away_stats_a.get("home", 1)
        home_factor_b = home_away_stats_b.get("away", 1)
        avg_goals_a *= home_factor_a  # Ajusta a média de gols de A para o fator casa
        avg_goals_b *= home_factor_b  # Ajusta a média de gols de B para o fator fora

    # Ajuste baseado em H2H, se disponível
    if h2h_stats:
        h2h_factor_a = h2h_stats.get("vitorias_a", 0)
        h2h_factor_b = h2h_stats.get("vitorias_b", 0)
        if h2h_factor_a > h2h_factor_b:
            avg_goals_a += 0.3  # Dê uma leve vantagem ao time A baseado no histórico
        elif h2h_factor_b > h2h_factor_a:
            avg_goals_b += 0.3  # Dê uma leve vantagem ao time B baseado no histórico

    # Verificar os mercados com maior probabilidade de sucesso
    for market in markets_to_check:
        over_prob = probabilities[market]["over"]
        under_prob = probabilities[market]["under"]

        # Calcular o critério de confiança, considerando a diferença nas probabilidades
        confidence = abs(over_prob - under_prob)  # A confiança é baseada na diferença entre "over" e "under"

        # Melhorar a escolha do mercado considerando a tendência de cada mercado
        market_prob = max(over_prob, under_prob)

        # Verificar se a confiança no mercado é alta o suficiente
        if market_prob > best_weight and confidence >= confidence_threshold:
            best_weight = market_prob
            best_market = market
            best_confidence = confidence

    # Retornar o melhor mercado, a maior probabilidade de "over" e a probabilidade em %
    if best_market is not None:
        return best_market, best_weight, probabilities[best_market]["over"] * 100, best_confidence
    else:
        return None, None, None, None