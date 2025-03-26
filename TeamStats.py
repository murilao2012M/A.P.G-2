import math
import numpy as np
from random import random
from scipy.stats import poisson
from modules.calcular_xg import calcular_xg

class TeamStats:
    def __init__(self, name, golsMarcados, golsSofridos, tendencia_gols, recent_games=None, estilo_ataque=1.0, estilo_defesa=1.0, jogos=None, ranking=None):
        self.name = name
        self.golsMarcados = golsMarcados  # Lista de gols marcados por jogo
        self.golsSofridos = golsSofridos  # Lista de gols sofridos por jogo
        self.tendencia_gols = tendencia_gols  # Pode ser uma lista ou outro tipo de dado
        self.recent_games = recent_games if recent_games else []  # Lista de jogos recentes
        self.estilo_ataque = estilo_ataque  # Fator de estilo de ataque (por padrão 1.0)
        self.estilo_defesa = estilo_defesa  # Fator de estilo de defesa (por padrão 1.0)
        self.jogos = jogos if jogos else []  # Lista de jogos
        self.ranking = ranking  # Pode ser o ranking atual do time no campeonato

    def style_of_play(self, tipo):
        """
        Retorna o fator multiplicador de acordo com o tipo de estilo de jogo ('ataque' ou 'defesa').
        """
        if tipo == "ataque":
            return self.estilo_ataque  # Retorna o estilo de ataque
        elif tipo == "defesa":
            return self.estilo_defesa  # Retorna o estilo de defesa
        else:
            raise ValueError(f"Tipo de estilo de jogo desconhecido: {tipo}")

    def playing_home(self):
        """
        Retorna True se o time está jogando em casa, baseado nos jogos registrados.
        Verifica se o time está como mandante em algum dos jogos.
        """
        for jogo in self.jogos:
            if jogo['mandante'] == self.name:
                return True  # Se o time é mandante em algum jogo, retorna True (jogando em casa)
        return False  # Caso contrário, assume que está jogando fora

    def get_recent_games(self, n=5):
        """
        Retorna os últimos N jogos (gols marcados) do time.
        """
        return self.recent_games[-n:]  # Retorna os últimos n jogos da lista recent_games
    
    def get_recent_opponents(self, n=5):
        opponents = []
        for jogo in self.jogos[-n:]:
            if jogo['mandante'] == self.name:
                opponents.append(jogo['visitante'])
            elif jogo['visitante'] == self.name:
                opponents.append(jogo['mandante'])
        return opponents

    def recent_wins(self, n=5):
        """
        Retorna o número de vitórias nos últimos N jogos.
        """
        recent_results = zip(self.golsMarcados[-n:], self.golsSofridos[-n:])
        wins = sum(1 for g_marcados, g_sofridos in recent_results if g_marcados > g_sofridos)
        return wins

    def recent_draws(self, n=5):
        """
        Conta os empates nos últimos 'recent_games' jogos.
        """
        draws = 0
        for g_marcados, g_sofridos in zip(self.golsMarcados[-n:], self.golsSofridos[-n:]):
            if g_marcados == g_sofridos:
                draws += 1
        return draws

    def recent_losses(self, n=5):
        """
        Conta as derrotas nos últimos 'recent_games' jogos.
        """
        losses = 0
        for g_marcados, g_sofridos in zip(self.golsMarcados[-n:], self.golsSofridos[-n:]):
            if g_marcados < g_sofridos:
                losses += 1
        return losses

    def get_home_away_stats(self, dados):
        """
        Retorna as estatísticas de jogos em casa e fora, com base nos dados fornecidos.
        """
        home_games = []
        away_games = []

        # Separando jogos em casa e fora com base em 'dados'
        for jogo in dados:
            if isinstance(jogo, dict):  # Verifique se é um dicionário
                if 'mandante' in jogo and 'visitante' in jogo:
                    if jogo['mandante'] == self.name:
                        home_games.append(jogo)
                    elif jogo['visitante'] == self.name:
                        away_games.append(jogo)

        # Retornando um dicionário com as estatísticas
        return {
            'home': home_games,  # Jogos em casa
            'away': away_games   # Jogos fora de casa
        }

    def average_goals_scored(self):
        """
        Retorna a média de gols marcados por jogo do time.
        """
        return np.mean(self.golsMarcados) if self.golsMarcados else 0

    def average_goals_conceded(self):
        """
        Retorna a média de gols sofridos por jogo do time.
        """
        return np.mean(self.golsSofridos) if self.golsSofridos else 0

    def average_goals_scored_last(self, n=5):
        """
        Retorna a média de gols marcados nos últimos N jogos.
        """
        return np.mean(self.golsMarcados[-n:]) if len(self.golsMarcados) >= n else 0

    def average_goals_conceded_last(self, n=5):
        """
        Retorna a média de gols sofridos nos últimos N jogos.
        """
        return np.mean(self.golsSofridos[-n:]) if len(self.golsSofridos) >= n else 0

    def get_strength(self):
        """
        Retorna a força do time baseada em seu desempenho geral.
        """
        return self.ranking if self.ranking else 0

    def is_important_match(self, opponent):
        """
        Determina se a partida contra o adversário é importante. 
        A lógica de definição de partida importante pode variar.
        Aqui, exemplo básico que compara força do time com o adversário.
        """
        return self.get_strength() > opponent.get_strength()


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
        
        print(f"\n🟢 Eficiência de Conversão nos Últimos {recent_games} Jogos:")
        print(f"{team_a.name}: {eficiencia_a:.2f} | {team_b.name}: {eficiencia_b:.2f}")

        # Defesa: Comparar xGA vs Gols Sofridos
        gols_sofridos_a = sum(team_a.golsSofridos[-recent_games:])
        gols_sofridos_b = sum(team_b.golsSofridos[-recent_games:])
        
        eficiencia_defensiva_a = gols_sofridos_a / xg_a['xGA'] if xg_a['xGA'] > 0 else 0
        eficiencia_defensiva_b = gols_sofridos_b / xg_b['xGA'] if xg_b['xGA'] > 0 else 0
        
        print(f"\n🟢 Eficiência Defensiva nos Últimos {recent_games} Jogos:")
        print(f"{team_a.name}: {eficiencia_defensiva_a:.2f} | {team_b.name}: {eficiencia_defensiva_b:.2f}")
    
    except Exception as e:
        print(f"\nErro ao exibir a análise da partida: {e}")