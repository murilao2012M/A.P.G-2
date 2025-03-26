import pandas as pd

def calculate_h2h_statistics(team_a, team_b, dados_partidas=None, ultimos=5, peso_recente=1, recent_games_count=3):
    try:
        jogos_h2h = []

        # Adicionar confrontos manuais entre os times
        if input("\nDeseja adicionar confrontos manuais entre os dois times? (s/n): ").strip().lower() == "s":
            num_jogos = int(input("Quantos confrontos diretos deseja adicionar? "))
            for i in range(num_jogos):
                gols_a = int(input(f"Gols marcados por {team_a} no jogo {i + 1}: "))
                gols_b = int(input(f"Gols marcados por {team_b} no jogo {i + 1}: "))
                jogos_h2h.append({
                    "HomeTeam": team_a,
                    "AwayTeam": team_b,
                    "FTHG": gols_a,
                    "FTAG": gols_b
                })

        # Adicionar dados de confrontos anteriores a partir de 'dados_partidas'
        if dados_partidas is not None:
            jogos_h2h.extend(dados_partidas[
                ((dados_partidas['HomeTeam'] == team_a) & (dados_partidas['AwayTeam'] == team_b)) |
                ((dados_partidas['HomeTeam'] == team_b) & (dados_partidas['AwayTeam'] == team_a))
            ][['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].to_dict(orient="records"))

        # Filtrando os últimos 'ultimos' jogos
        jogos_h2h = jogos_h2h[-ultimos:]

        if not jogos_h2h:
            return None

            # Função para calcular vitórias como mandante ou visitante
        def calcular_vitorias(jogos, team, is_mandante):
            return sum(1 for jogo in jogos if 
                    "FTHG" in jogo and "FTAG" in jogo and 
                    ((jogo["HomeTeam"] == team and jogo["FTHG"] > jogo["FTAG"]) if is_mandante else 
                        (jogo["AwayTeam"] == team and jogo["FTAG"] > jogo["FTHG"])))

        def calcular_gols(jogos, team, is_mandante):
                return sum(jogo["FTHG"] if is_mandante else jogo["FTAG"] for jogo in jogos if 
                        (jogo["HomeTeam"] == team and is_mandante) or (jogo["AwayTeam"] == team and not is_mandante))

        def calcular_media_diferenca_gols(jogos, team):
                gols_marcados = sum(jogo["FTHG"] if jogo["HomeTeam"] == team else jogo["FTAG"] for jogo in jogos)
                gols_sofridos = sum(jogo["FTAG"] if jogo["HomeTeam"] == team else jogo["FTHG"] for jogo in jogos)
                return gols_marcados / len(jogos), gols_sofridos / len(jogos), gols_marcados - gols_sofridos

        # Calculando as estatísticas de vitórias e gols
        vit_a_mandante = calcular_vitorias(jogos_h2h, team_a, is_mandante=True)
        vit_a_visitante = calcular_vitorias(jogos_h2h, team_a, is_mandante=False)
        vit_b_mandante = calcular_vitorias(jogos_h2h, team_b, is_mandante=True)
        vit_b_visitante = calcular_vitorias(jogos_h2h, team_b, is_mandante=False)

        empates = sum(1 for jogo in jogos_h2h if "FTHG" in jogo and "FTAG" in jogo and jogo["FTHG"] == jogo["FTAG"])

        # Média de gols e diferença de gols para cada time
        media_gols_a, media_gols_a_sofridos, diff_gols_a = calcular_media_diferenca_gols(jogos_h2h, team_a)
        media_gols_b, media_gols_b_sofridos, diff_gols_b = calcular_media_diferenca_gols(jogos_h2h, team_b)

        # Estatísticas gerais
        gols_a = sum(jogo["FTHG"] for jogo in jogos_h2h if "FTHG" in jogo and (jogo["HomeTeam"] == team_a or jogo["AwayTeam"] == team_a))
        gols_b = sum(jogo["FTAG"] for jogo in jogos_h2h if "FTAG" in jogo and (jogo["HomeTeam"] == team_b or jogo["AwayTeam"] == team_b))

        # Calculando o impacto dos últimos jogos
        recent_games = jogos_h2h[-recent_games_count:]
        recent_victories_a = sum(1 for jogo in recent_games if jogo["HomeTeam"] == team_a and jogo["FTHG"] > jogo["FTAG"])
        recent_victories_b = sum(1 for jogo in recent_games if jogo["AwayTeam"] == team_b and jogo["FTAG"] > jogo["FTHG"])

        # Retorno das estatísticas
        return {
            'vitorias_a_mandante': vit_a_mandante,
            'vitorias_a_visitante': vit_a_visitante,
            'vitorias_b_mandante': vit_b_mandante,
            'vitorias_b_visitante': vit_b_visitante,
            'empates': empates,
            'gols_a': gols_a,
            'gols_b': gols_b,
            'media_gols_a': media_gols_a,
            'media_gols_b': media_gols_b,
            'media_gols_a_sofridos': media_gols_a_sofridos,
            'media_gols_b_sofridos': media_gols_b_sofridos,
            'diferenca_gols_a': diff_gols_a,
            'diferenca_gols_b': diff_gols_b,
            'recent_victories_a': recent_victories_a,
            'recent_victories_b': recent_victories_b,
            'total_jogos': len(jogos_h2h)
        }

    except Exception as e:
        raise ValueError(f"Erro ao calcular as estatísticas de H2H: {e}, Dados: {jogos_h2h}")
