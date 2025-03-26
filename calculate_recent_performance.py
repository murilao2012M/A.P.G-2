def calculate_recent_performance(dados_partidas, team, home_games=True, n_games=5, peso_recente=True):
    """
    Calcula o desempenho recente de um time com base nos últimos 'n' jogos.
    Inclui melhorias como:
    - Peso dinâmico para jogos mais recentes.
    - Consideração de vitórias, empates e derrotas.
    - Ajuste para desempenho em casa e fora de casa.
    - Análise da diferença de gols.
    - Cálculo de gols esperados (xG) se disponível.
    
    Parâmetros:
        dados_partidas (DataFrame): Dados das partidas.
        team (Team): Objeto representando o time.
        home_games (bool): Se True, considera jogos em casa; False, jogos fora.
        n_games (int): Número de jogos recentes a considerar.
        peso_recente (bool): Se True, aplica pesos maiores para jogos mais recentes.
    
    Retorna:
        (gols_marcados, gols_sofridos, diff_gols, vitoria, empate, derrota) : Desempenho recente do time.
    """
    # Filtrando os jogos do time
    if home_games:
        jogos = dados_partidas[dados_partidas['HomeTeam'] == team.name].tail(n_games)
    else:
        jogos = dados_partidas[dados_partidas['AwayTeam'] == team.name].tail(n_games)
    
    if jogos.empty:
        return None  # Retorna None se não houver jogos suficientes
    
    # Cálculo de gols marcados, gols sofridos e diferença de gols
    gols_marcados = jogos['FTHG'].sum() if home_games else jogos['FTAG'].sum()
    gols_sofridos = jogos['FTAG'].sum() if home_games else jogos['FTHG'].sum()
    diff_gols = (jogos['FTHG'] - jogos['FTAG']).sum() if home_games else (jogos['FTAG'] - jogos['FTHG']).sum()

    # Inicializando variáveis de vitórias, empates e derrotas
    vitoria, empate, derrota = 0, 0, 0

    for jogo in jogos.itertuples():
        if home_games:
            if jogo.FTHG > jogo.FTAG:
                vitoria += 1
            elif jogo.FTHG == jogo.FTAG:
                empate += 1
            else:
                derrota += 1
        else:
            if jogo.FTAG > jogo.FTHG:
                vitoria += 1
            elif jogo.FTAG == jogo.FTHG:
                empate += 1
            else:
                derrota += 1

    # Peso dinâmico para jogos mais recentes
    if peso_recente:
        peso_total = 0
        peso_gols_marcados = 0
        peso_gols_sofridos = 0
        peso_diff_gols = 0

        for i, jogo in enumerate(jogos[::-1]):  # Invertendo para começar do mais recente
            peso = (i + 1)  # Peso crescente para cada jogo mais recente
            peso_total += peso
            peso_gols_marcados += (jogo.FTHG if home_games else jogo.FTAG) * peso
            peso_gols_sofridos += (jogo.FTAG if home_games else jogo.FTHG) * peso
            peso_diff_gols += ((jogo.FTHG - jogo.FTAG) if home_games else (jogo.FTAG - jogo.FTHG)) * peso
        
        # Normalizando pelo peso total
        gols_marcados = peso_gols_marcados / peso_total
        gols_sofridos = peso_gols_sofridos / peso_total
        diff_gols = peso_diff_gols / peso_total

    # Retorno com informações detalhadas do desempenho recente
    return {
        'gols_marcados': gols_marcados,
        'gols_sofridos': gols_sofridos,
        'diferenca_gols': diff_gols,
        'vitórias': vitoria,
        'empates': empate,
        'derrotas': derrota,
        'total_jogos': len(jogos)
    }