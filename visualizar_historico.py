import json
from datetime import datetime


# Lista global para armazenar simulações
historico_simulacoes = []

# Função para carregar o histórico de simulações a partir de um arquivo JSON
def carregar_historico():
    try:
        with open("historico_simulacoes.json", "r") as f:
            global historico_simulacoes
            historico_simulacoes = json.load(f)
    except FileNotFoundError:
        # Arquivo não encontrado, significa que não há histórico salvo
        print("📂 Nenhum histórico de simulação encontrado. Iniciando um novo histórico.")

# Função para salvar o histórico de simulações em um arquivo JSON
def salvar_historico():
    with open("historico_simulacoes.json", "w") as f:
        json.dump(historico_simulacoes, f, indent=4)
    print("✅ Histórico de simulações salvo com sucesso!")

# Função para salvar o resultado de uma simulação no histórico
def salvar_simulacao(historico_simulacoes, team_a, team_b, resultado, gols_a, gols_b):
    # Aqui você deve salvar os dados, por exemplo, em um arquivo ou em uma lista
    # Exemplo básico de como poderia ser feita a gravação:
    simulacao = {
        'time_a': team_a.name,
        'time_b': team_b.name,
        'resultado': resultado,
        'gols_a': gols_a,
        'gols_b': gols_b,
    }
    historico_simulacoes.append(simulacao)
    print("✅ Simulação salva com sucesso!")
    
# Função para visualizar o histórico de simulações com a opção de ordenar
def visualizar_historico(ordem="desc"):
    """
    Exibe o histórico de simulações, ordenando por data, se desejado.
    """
    if not historico_simulacoes:
        print("\n📜 Nenhum histórico de simulação disponível.")
        return

    # Ordena o histórico por data (decrescente ou crescente)
    historico_simulacoes_sorted = sorted(historico_simulacoes, key=lambda x: x["Data"], reverse=(ordem=="desc"))

    print("\n📜 Histórico de Simulações:")
    print(f"{'Data':<20} {'Time A':<20} {'FTHG':<10} {'FTAG':<20} {'FTAG':<10} {'Resultado':<15}")
    for sim in historico_simulacoes_sorted:
        print(f"{sim['Data']:<20} {sim['Time A']:<20} {sim['FTHG']:<10} {sim['FTAG']:<20} {sim[' FTAG ']:<10} {sim['Resultado']:<15}")

# Função para remover uma simulação específica
def remover_simulacao(index):
    """
    Remove uma simulação do histórico com base no índice.
    """
    try:
        simulacao_removida = historico_simulacoes.pop(index - 1)  # Ajuste para index 1-based
        salvar_historico()  # Salva a lista atualizada
        print(f"✅ Simulação '{simulacao_removida['Time A']} vs {simulacao_removida['Time B']}' removida.")
    except IndexError:
        print("❌ Índice inválido. Não foi possível remover a simulação.")

# Função para editar uma simulação existente
def editar_simulacao(index, novos_resultados):
    """
    Edita uma simulação já salva no histórico.
    """
    try:
        simulacao = historico_simulacoes[index - 1]  # Ajuste para index 1-based
        simulacao['FTHG'] = novos_resultados['FTHG']
        simulacao['FTAG'] = novos_resultados['FTAG']
        simulacao['Resultado'] = novos_resultados['Resultado']
        salvar_historico()  # Salva as alterações no arquivo
        print(f"✅ Simulação '{simulacao['Time A']} vs {simulacao['Time B']}' editada com sucesso.")
    except IndexError:
        print("❌ Índice inválido. Não foi possível editar a simulação.")

# Carregar histórico ao iniciar
carregar_historico()