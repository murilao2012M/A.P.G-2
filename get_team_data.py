from modules.TeamStats import TeamStats


def get_team_data(name, dados_partidas=None):
    """
    Busca os dados de um time, permitindo que o usuário insira manualmente os últimos jogos.
    Oferece a opção de usar os últimos 5 ou 10 jogos, e pode calcular o desempenho com base na média e tendência recente.
    """
    try:
        golsMarcados = []
        golsSofridos = []

        # Pedir ao usuário para inserir os dados dos últimos jogos
        print(f"\nInsira os dados dos últimos jogos do time '{name}':")

        # Solicita a quantidade de jogos a serem inseridos (usando input robusto)
        while True:
            try:
                jogos_manuais = int(input("Quantos jogos deseja inserir manualmente? (mínimo 3, máximo 10): ").strip())
                if jogos_manuais < 3 or jogos_manuais > 10:
                    raise ValueError("O número de jogos deve ser entre 3 e 10.")
                break
            except ValueError as e:
                print(f"Erro: {e}. Por favor, insira um número válido entre 3 e 10.")
        
        for i in range(1, jogos_manuais + 1):
            # Pede ao usuário para inserir os gols marcados e gols sofridos de cada jogo
            while True:
                try:
                    marcados = int(input(f"Gols marcados pelo {name} no jogo {i}: "))
                    sofridos = int(input(f"Gols sofridos pelo {name} no jogo {i}: "))
                    if marcados < 0 or sofridos < 0:
                        raise ValueError("Os valores de gols não podem ser negativos.")
                    break
                except ValueError as e:
                    print(f"Erro: {e}. Por favor, insira números válidos para gols.")
            
            golsMarcados.append(marcados)
            golsSofridos.append(sofridos)

        # Calcular a média de gols marcados e sofridos
        media_gols = sum(golsMarcados) / len(golsMarcados) if golsMarcados else 0
        print(f"\nMédia de gols marcados: {media_gols:.2f}")
        media_sofridos = sum(golsSofridos) / len(golsSofridos) if golsSofridos else 0
        print(f"Média de gols sofridos: {media_sofridos:.2f}")

        # Determinar o peso com base na média de gols
        peso = 1.0  # Peso padrão
        if media_gols > 2.5:  
            peso = 1.2  # Pesos maiores para jogos de alta média de gols
        elif media_gols < 1.5:  
            peso = 0.8  # Pesos menores para jogos de baixa média de gols
        print(f"Peso aplicado para o campeonato: {peso}")

        # Aplicar o peso aos gols marcados e gols sofridos
        golsMarcadosPesados = [gols * peso for gols in golsMarcados]
        golsSofridosPesados = [gols * peso for gols in golsSofridos]

        # Análise de tendência (aumento ou queda de desempenho)
        tendencia_gols = "neutra"
        if len(golsMarcados) > 1:  # Verificar se há pelo menos 2 jogos para análise de tendência
            if golsMarcados[-1] > golsMarcados[-2]:  # Compara o último jogo com o penúltimo
                tendencia_gols = "em alta"
            elif golsMarcados[-1] < golsMarcados[-2]:
                tendencia_gols = "em queda"

        print(f"Tendência de gols: {tendencia_gols}")

        # Retornar os dados do time com as estatísticas calculadas
        return TeamStats(name, golsMarcadosPesados, golsSofridosPesados, tendencia_gols)

    except ValueError as e:
        print(f"Erro na entrada de dados: {e}")
        return None
    except Exception as e:
        print(f"Erro ao processar os dados do time '{name}': {e}")
        return None