import numpy as np
from modules.calcular_fatores import ajustar_fatores

def calcular_xg(team, adversario=None, fator_ataque=1.2, fator_defesa=1.1, ajuste_recente=True, 
                ajustar_por_adversario=True, estilo_jogo=True, impacto_clima=True, 
                lesoes_jogadores=True, influencia_arbitragem=True, fator_casa=1.1, fator_fora=0.9):
    """
    Calcula o xG (Expected Goals) e xGA (Expected Goals Against) para um time, considerando fatores como
    desempenho recente, estilo de jogo, adversário, jogo em casa/fora, lesões, clima e influência da arbitragem.
    """
    
    # Se não houver adversário, usamos o time contra si mesmo (ou podemos omitir o ajuste relacionado ao adversário)
    if adversario is None:
        adversario = team  # Faz o cálculo como se estivesse jogando contra si mesmo.

    # Ajuste dinâmico dos fatores de ataque e defesa com base no desempenho recente
    if ajuste_recente:
        fator_ataque = ajustar_fatores(team, tipo="ataque")
        fator_defesa = ajustar_fatores(team, tipo="defesa")
    
    # Ajuste com base no adversário
    if ajustar_por_adversario:
        # Ajuste da defesa do time considerando a média de gols sofridos pelo adversário
        fator_defesa = ajustar_fatores(adversario, tipo="defesa", intensidade=False)
    
    # Ajuste baseado no estilo de jogo do time
    if estilo_jogo:
        estilo_ataque = team.style_of_play("ataque")  # Fator de ataque do time
        estilo_defesa = adversario.style_of_play("defesa")  # Fator de defesa do adversário
    else:
        estilo_ataque, estilo_defesa = 1, 1  # Estilo de jogo neutro, sem impacto

    # Ajuste dinâmico da média de gols marcados e concedidos
    media_gols = team.average_goals_scored()
    media_gols_concedidos = team.average_goals_conceded()

    # Calculando o xG (Expected Goals) e xGA (Expected Goals Against)
    xg = (media_gols * estilo_ataque) * fator_ataque
    xga = (media_gols_concedidos * estilo_defesa) * fator_defesa

    # Ajuste para a condição de casa/fora
    if team.playing_home():
        xg *= fator_casa  # Vantagem em casa
        xga *= fator_fora  # Desvantagem defensiva em casa
    else:
        xg *= fator_fora  # Desvantagem fora de casa
        xga *= fator_casa  # Vantagem defensiva fora de casa
    
    # Considerando impacto de jogos decisivos ou contextos específicos
    if team.is_important_match(adversario):
        xg *= 1.2  # Aumenta o xG em jogos decisivos

    # Limitar o valor de xG e xGA a um máximo de 5 (para evitar superestimação)
    xg = np.clip(xg, 0, 5)
    xga = np.clip(xga, 0, 5)
    
    # Retornando os valores ajustados de xG e xGA
    return {
        "xG": xg,
        "xGA": xga,
        "adjustments": {
            "fator_ataque": fator_ataque,
            "fator_defesa": fator_defesa,
            "estilo_ataque": estilo_ataque,
            "estilo_defesa": estilo_defesa,
        }
    }