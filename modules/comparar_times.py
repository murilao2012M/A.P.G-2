def calcular_media_ultimos_jogos(gols_marcados, gols_sofridos, n=5):
    """
    Calcula a média de gols marcados e gols sofridos nos últimos 'n' jogos.
    """
    return sum(gols_marcados[-n:]) / min(len(gols_marcados), n), sum(gols_sofridos[-n:]) / min(len(gols_sofridos), n)

def calcular_tendencia(media_gols, media_sofridos, saldo_gols):
    """
    Determina a tendência de um time com base em sua média de gols marcados e gols sofridos, 
    além do saldo de gols.
    """
    if saldo_gols > 0 and media_gols > media_sofridos:
        return "Em Crescimento"
    elif saldo_gols < 0 and media_gols < media_sofridos:
        return "Declínio Acelerado"
    elif saldo_gols == 0:
        return "Estável"
    else:
        return "Oscilante"

def calcular_vitorias_empates_derrotas(gols_marcados, gols_sofridos):
    """
    Calcula o número de vitórias, empates e derrotas de um time com base nos gols marcados e sofridos.
    """
    vitorias = sum(1 for i in range(len(gols_marcados)) if gols_marcados[i] > gols_sofridos[i])
    empates = sum(1 for i in range(len(gols_marcados)) if gols_marcados[i] == gols_sofridos[i])
    derrotas = sum(1 for i in range(len(gols_marcados)) if gols_marcados[i] < gols_sofridos[i])
    return vitorias, empates, derrotas

def comparar_times(team_a, team_b):
    """
    Compara dois times em termos de desempenho ofensivo, defensivo e recente, 
    incluindo forma recente, tendência, saldo de gols, e análises detalhadas.
    """
    print(f"\n🔍 Comparação entre {team_a.name} e {team_b.name}:\n")

    # Desempenho ofensivo e defensivo
    print("⚽ Desempenho Ofensivo:")
    print(f"{team_a.name}: {team_a.average_goals_scored():.2f} gols/jogo")
    print(f"{team_b.name}: {team_b.average_goals_scored():.2f} gols/jogo")

    print("\n🛡️ Desempenho Defensivo:")
    print(f"{team_a.name}: {team_a.average_goals_conceded():.2f} gols sofridos/jogo")
    print(f"{team_b.name}: {team_b.average_goals_conceded():.2f} gols sofridos/jogo")

    # Forma recente (últimos 5 jogos)
    media_gols_a, media_sofridos_a = calcular_media_ultimos_jogos(team_a.golsMarcados, team_a.golsSofridos, 5)
    media_gols_b, media_sofridos_b = calcular_media_ultimos_jogos(team_b.golsMarcados, team_b.golsSofridos, 5)

    print("\n📊 Forma Recente (Últimos 5 Jogos):")
    print(f"{team_a.name}: {media_gols_a:.2f} gols marcados, {media_sofridos_a:.2f} gols sofridos")
    print(f"{team_b.name}: {media_gols_b:.2f} gols marcados, {media_sofridos_b:.2f} gols sofridos")

    # Saldo de gols nos últimos jogos
    saldo_gols_a = sum(team_a.golsMarcados[-5:]) - sum(team_a.golsSofridos[-5:])
    saldo_gols_b = sum(team_b.golsMarcados[-5:]) - sum(team_b.golsSofridos[-5:])
    
    print("\n💥 Saldo de Gols (Últimos 5 Jogos):")
    print(f"{team_a.name}: {saldo_gols_a}")
    print(f"{team_b.name}: {saldo_gols_b}")

    # Tendência (analisando média de gols e saldo de gols)
    tendencia_a = calcular_tendencia(media_gols_a, media_sofridos_a, saldo_gols_a)
    tendencia_b = calcular_tendencia(media_gols_b, media_sofridos_b, saldo_gols_b)

    print("\n📈 Tendência:")
    print(f"{team_a.name}: {tendencia_a}")
    print(f"{team_b.name}: {tendencia_b}")

    # Performance recente (últimos 5 jogos)
    print("\n🏆 Performance Recente (Últimos 5 Jogos):")
    wins_a, draws_a, losses_a = calcular_vitorias_empates_derrotas(team_a.golsMarcados[-5:], team_a.golsSofridos[-5:])
    wins_b, draws_b, losses_b = calcular_vitorias_empates_derrotas(team_b.golsMarcados[-5:], team_b.golsSofridos[-5:])

    print(f"{team_a.name}: {wins_a} vitórias, {draws_a} empates, {losses_a} derrotas")
    print(f"{team_b.name}: {wins_b} vitórias, {draws_b} empates, {losses_b} derrotas")

    # Análise final: Melhor desempenho recente
    if wins_a > wins_b:
        print(f"\n🏅 {team_a.name} teve melhor desempenho recente!")
    elif wins_b > wins_a:
        print(f"\n🏅 {team_b.name} teve melhor desempenho recente!")
    else:
        print(f"\n🏅 Ambos os times tiveram desempenho semelhante nos últimos jogos.")

    # Classificação final dos times com base no desempenho recente
    if wins_a + draws_a > wins_b + draws_b:
        print(f"\n🥇 {team_a.name} é o time em melhor forma atualmente!")
    elif wins_b + draws_b > wins_a + draws_a:
        print(f"\n🥇 {team_b.name} é o time em melhor forma atualmente!")
    else:
        print("\n🥇 Ambos os times estão igualmente em boa forma atualmente!")