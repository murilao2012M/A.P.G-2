from modules.calcular_xg import calcular_xg

def exibir_xg(team_a, team_b, recent_games=5):
    """
    Mostra os xG, xGA e análise comparativa entre dois times, considerando também o desempenho recente,
    a comparação de vitórias recentes, e a análise de conversão e defesa.
    """
    try:
        # Verificar se a partida é importante entre os times
        if team_a.is_important_match(team_b):  # Passa o adversário (team_b) corretamente
            print(f"\n⚡ A partida {team_a.name} x {team_b.name} é uma partida importante!")
        else:
            print(f"\n🧑‍🤝‍🧑 A partida {team_a.name} x {team_b.name} não é considerada importante.")
        
        # Calcular o xG e xGA para os dois times
        xg_a = calcular_xg(team_a)
        xg_b = calcular_xg(team_b)
        
        if xg_a is None or xg_b is None:
            raise ValueError("Erro no cálculo de xG para um dos times.")

        print(f"\n🔢 Estatísticas Avançadas (xG e xGA) para os Times {team_a.name} e {team_b.name}:")
        
        print(f"{team_a.name} - xG: {xg_a['xG']:.2f}, xGA: {xg_a['xGA']:.2f}")
        print(f"{team_b.name} - xG: {xg_b['xG']:.2f}, xGA: {xg_b['xGA']:.2f}")

        # Comparar os times
        diff_xg = xg_a['xG'] - xg_b['xG']
        diff_xga = xg_a['xGA'] - xg_b['xGA']
        
        if diff_xg > 0:
            print(f"\n{team_a.name} tem uma expectativa de gols maior (+{diff_xg:.2f}) do que {team_b.name}.")
        elif diff_xg < 0:
            print(f"\n{team_b.name} tem uma expectativa de gols maior (+{abs(diff_xg):.2f}) do que {team_a.name}.")
        else:
            print(f"\nAmbos os times têm a mesma expectativa de gols ({xg_a['xG']:.2f}).")
        
        if diff_xga > 0:
            print(f"{team_a.name} tem uma expectativa de gols sofridos maior (+{diff_xga:.2f}) do que {team_b.name}.")
        elif diff_xga < 0:
            print(f"{team_b.name} tem uma expectativa de gols sofridos maior (+{abs(diff_xga):.2f}) do que {team_a.name}.")
        else:
            print(f"Ambos os times têm a mesma expectativa de gols sofridos ({xg_a['xGA']:.2f}).")

        # Eficiência de Conversão: Comparar xG vs Gols Marcados
        gols_realizados_a = sum(team_a.golsMarcados[-recent_games:])
        gols_realizados_b = sum(team_b.golsMarcados[-recent_games:])
        
        eficiencia_a = gols_realizados_a / xg_a['xG'] if xg_a['xG'] > 0 else 0
        eficiencia_b = gols_realizados_b / xg_b['xG'] if xg_b['xG'] > 0 else 0

        print(f"\n⚽ Eficiência de Conversão (Gols Marcados / xG):")
        print(f"{team_a.name}: {eficiencia_a:.2f} (gols reais: {gols_realizados_a}, xG: {xg_a['xG']:.2f})")
        print(f"{team_b.name}: {eficiencia_b:.2f} (gols reais: {gols_realizados_b}, xG: {xg_b['xG']:.2f})")

        # Comparação de Defesa: xGA vs Gols Sofridos
        gols_sofridos_a = sum(team_a.golsSofridos[-recent_games:])
        gols_sofridos_b = sum(team_b.golsSofridos[-recent_games:])
        
        eficiencia_defesa_a = gols_sofridos_a / xg_a['xGA'] if xg_a['xGA'] > 0 else 0
        eficiencia_defesa_b = gols_sofridos_b / xg_b['xGA'] if xg_b['xGA'] > 0 else 0

        print(f"\n🛡️ Eficiência Defensiva (Gols Sofridos / xGA):")
        print(f"{team_a.name}: {eficiencia_defesa_a:.2f} (gols sofridos: {gols_sofridos_a}, xGA: {xg_a['xGA']:.2f})")
        print(f"{team_b.name}: {eficiencia_defesa_b:.2f} (gols sofridos: {gols_sofridos_b}, xGA: {xg_b['xGA']:.2f})")

        # Verificar vitórias recentes
        wins_a = team_a.recent_wins(recent_games)
        wins_b = team_b.recent_wins(recent_games)
        
        # Verificar empates recentes
        draws_a = team_a.recent_draws(recent_games)
        draws_b = team_b.recent_draws(recent_games)

        print(f"\n🏆 Desempenho nos últimos {recent_games} jogos:")
        print(f"{team_a.name}: {wins_a} vitórias, {draws_a} empates")
        print(f"{team_b.name}: {wins_b} vitórias, {draws_b} empates")

        if wins_a > wins_b:
            print(f"\n{team_a.name} tem mais vitórias recentes ({wins_a} contra {wins_b}).")
        elif wins_a < wins_b:
            print(f"\n{team_b.name} tem mais vitórias recentes ({wins_b} contra {wins_a}).")
        else:
            print(f"\nAmbos os times têm o mesmo número de vitórias nos últimos {recent_games} jogos ({wins_a}).")
        
        if draws_a > draws_b:
            print(f"\n{team_a.name} tem mais empates recentes ({draws_a} contra {draws_b}).")
        elif draws_a < draws_b:
            print(f"\n{team_b.name} tem mais empates recentes ({draws_b} contra {draws_a}).")
        else:
            print(f"\nAmbos os times têm o mesmo número de empates nos últimos {recent_games} jogos ({draws_a}).")

        # Análise de qualidade de adversários recentes (strong, medium, weak)
        def classificar_adversarios(vitorias, adversarios):
            strong = sum(1 for a in adversarios if a.strength == 'strong')
            medium = sum(1 for a in adversarios if a.strength == 'medium')
            weak = sum(1 for a in adversarios if a.strength == 'weak')
            return strong, medium, weak

        adversarios_a = team_a.get_recent_opponents(recent_games)
        adversarios_b = team_b.get_recent_opponents(recent_games)
        
        strong_a, medium_a, weak_a = classificar_adversarios(wins_a, adversarios_a)
        strong_b, medium_b, weak_b = classificar_adversarios(wins_b, adversarios_b)

        print(f"\n📊 Análise de Qualidade de Adversários Recentes:")
        print(f"{team_a.name} enfrentou {strong_a} adversários fortes, {medium_a} médios, {weak_a} fracos.")
        print(f"{team_b.name} enfrentou {strong_b} adversários fortes, {medium_b} médios, {weak_b} fracos.")
        
    except Exception as e:
        print(f"⚠️ Ocorreu um erro ao calcular ou exibir as estatísticas de xG: {e}")