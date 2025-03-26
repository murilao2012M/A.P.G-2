from modules.simulate_match import simulate_match


def simular_partida_monte_carlo(self, time_adversario, n_simulacoes=1000):
        vitorias, empates, derrotas = 0, 0, 0

        # Pegamos as médias dos últimos 5 jogos para ambos os times
        media_gols_time, media_gols_time_adversario = self.last_n_game_performance(5)
        media_gols_adversario, media_gols_adversario_adversario = time_adversario.last_n_game_performance(5)
        
        print(f"Simulando {n_simulacoes} jogos entre {self.name} e {time_adversario.name}...")
        
        # Simulando os jogos
        for _ in range(n_simulacoes):
            gols_a, gols_b = simulate_match(self, time_adversario)  # Usando 'self' para o time A e 'time_adversario' para o time B

            if gols_a > gols_b:
                vitorias += 1
            elif gols_b > gols_a:
                derrotas += 1
            else:
                empates += 1

        # Calculando as probabilidades
        probabilidade_vitoria = (vitorias / n_simulacoes) * 100
        probabilidade_empate = (empates / n_simulacoes) * 100
        probabilidade_derrota = (derrotas / n_simulacoes) * 100
        
        # Exibindo o resultado final
        print(f"Simulação finalizada após {n_simulacoes} simulações.")
        print(f"{self.name} venceu {probabilidade_vitoria:.2f}% das simulações.")
        print(f"{self.name} empatou {probabilidade_empate:.2f}% das simulações.")
        print(f"{self.name} perdeu {probabilidade_derrota:.2f}% das simulações.")
        
        # Exibindo uma mensagem indicando a probabilidade
        if probabilidade_vitoria > probabilidade_derrota:
            print(f"Com base nas simulações, {self.name} tem mais chances de vencer o jogo.")
        elif probabilidade_vitoria < probabilidade_derrota:
            print(f"Com base nas simulações, {time_adversario.name} tem mais chances de vencer o jogo.")
        else:
            print("As simulações indicam uma probabilidade equilibrada entre os times.")

        return probabilidade_vitoria, probabilidade_empate, probabilidade_derrota