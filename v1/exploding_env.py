# exploding_env.py - V1 (Simple version)
# Entorno simplificado de Exploding Kittens + helpers de estado/acciones.

import numpy as np
import random


class ExplodingKittensEnv:
    """Entorno de 2 jugadores (agente vs oponente heurístico).

    Jugador 0: agente (DQN).
    Jugador 1: oponente heurístico o humano (según el modo).
    """

    def __init__(self,
                 deck_size=30,
                 num_bombs=4,
                 num_defuse_in_deck=2,
                 max_turns=200):
        self.deck_size_init = deck_size
        self.num_bombs_init = num_bombs
        self.num_defuse_in_deck = num_defuse_in_deck
        self.max_turns = max_turns

        # Parámetros de mano inicial
        self.start_defuse = 1
        self.start_skip = 1
        self.start_attack = 1

        self.reset()

    # ---------- API pública ----------

    def reset(self):
        """Reinicia el juego y devuelve la observación inicial."""
        num_bombs = self.num_bombs_init
        num_defuse_deck = self.num_defuse_in_deck
        remaining = self.deck_size_init - num_bombs - num_defuse_deck

        num_skip_deck = max(0, remaining // 8)
        num_attack_deck = max(0, remaining // 8)
        num_safe_deck = remaining - num_skip_deck - num_attack_deck

        deck = ['Bomb'] * num_bombs
        deck += ['Defuse'] * num_defuse_deck
        deck += ['Skip'] * num_skip_deck
        deck += ['Attack'] * num_attack_deck
        deck += ['Safe'] * num_safe_deck

        random.shuffle(deck)
        self.deck = deck

        # Manos iniciales
        self.hands = [
            {'Defuse': self.start_defuse, 'Skip': self.start_skip, 'Attack': self.start_attack, 'Safe': 0},
            {'Defuse': self.start_defuse, 'Skip': self.start_skip, 'Attack': self.start_attack, 'Safe': 0},
        ]

        # Obligación de robo
        self.pending_draws = [1, 1]

        # Estado global
        self.current_player = 0
        self.done = False
        self.winner = None
        self.turn_count = 0

        # Fase del turno del agente
        self.phase = 'action'  # 'action' o 'defuse'
        self.defuse_pending_for_agent = False

        # Última acción del oponente vista por el agente:
        # 0 = solo robar, 1 = Skip, 2 = Attack, 3 = usar Defuse
        self.last_opp_action = 0

        # Última carta robada por cada jugador (para UI / logs)
        self.last_drawn = {0: None, 1: None}

        return self._get_obs()

    def step(self, action):
        """Paso del agente (jugador 0) contra oponente heurístico.

        - Si phase == 'defuse': acciones 3/4/5 -> posición de la bomba.
        - Si phase == 'action': acciones 0/1/2 (otras se ignoran).
        """
        assert not self.done, "No se puede llamar step() en un episodio terminado"
        assert self.current_player == 0, "step() solo se usa en turno del agente"

        reward = 0.0
        info = {}

        # Fase de defuse del agente
        if self.phase == 'defuse':
            pos_choice = self._map_defuse_action_to_position(action)
            self._reinsert_bomb_for_agent(pos_choice)
            self.defuse_pending_for_agent = False
            self.phase = 'action'

            if not self.done:
                self.current_player = 1
                self._opponent_turn()

            if self.done:
                reward = 1.0 if self.winner == 0 else -1.0
                info['winner'] = self.winner
                return self._get_obs(), reward, True, info

            self.current_player = 0
            self.turn_count += 1
            if self.turn_count >= self.max_turns:
                self.done = True
                self.winner = None
                info['winner'] = None
                info['reason'] = 'max_turns'
                return self._get_obs(), 0.0, True, info

            return self._get_obs(), 0.0, False, info

        # Fase normal del agente
        self._play_agent_turn(action)

        if self.done:
            reward = 1.0 if self.winner == 0 else -1.0
            info['winner'] = self.winner
            return self._get_obs(), reward, True, info

        # Si entró en defuse del agente, espera la próxima acción
        if self.phase == 'defuse':
            info['phase'] = 'defuse'
            return self._get_obs(), 0.0, False, info

        # Turno del oponente heurístico
        self.current_player = 1
        self._opponent_turn()

        if self.done:
            reward = 1.0 if self.winner == 0 else -1.0
            info['winner'] = self.winner
            return self._get_obs(), reward, True, info

        self.current_player = 0
        self.turn_count += 1
        if self.turn_count >= self.max_turns:
            self.done = True
            self.winner = None
            info['winner'] = None
            info['reason'] = 'max_turns'
            return self._get_obs(), 0.0, True, info

        return self._get_obs(), 0.0, False, info

    def render(self):
        print(f"\n=== Turno {self.turn_count}, Jugador {self.current_player} ===")
        print(f"Cartas en deck: {len(self.deck)}")
        for i, h in enumerate(self.hands):
            print(f"Mano jugador {i}: {h}")
        print(f"Pending draws: {self.pending_draws}")
        print(f"Done: {self.done}, Winner: {self.winner}")

    def _get_obs(self):
        """Observación para el agente.
        
        Features:
        - 0: Número de cartas en el deck
        - 1-4: Cartas del agente (Defuse, Skip, Attack, Safe)
        - 5: Total de cartas del oponente (normalizado)
        - 6-7: Última acción del oponente (Skip, Attack)
        - 8: Número de bombas restantes en el deck
        - 9: Obligación de robo (pending_draws[0])
        - 10: Fase (0=action, 1=defuse)
        """
        deck_size = len(self.deck)
        deck_size_norm = deck_size / max(1, self.deck_size_init)
        
        bombs_in_deck = self._count_bombs_in_deck()
        bombs_norm = bombs_in_deck / max(1, self.num_bombs_init)

        h_agent = self.hands[0]
        h_opp = self.hands[1]

        opp_cards_norm = sum(h_opp.values()) / 15.0

        last_opp_skip = 1.0 if self.last_opp_action == 1 else 0.0
        last_opp_attack = 1.0 if self.last_opp_action == 2 else 0.0

        phase_defuse = 1.0 if self.phase == 'defuse' else 0.0

        obs = np.array([
            deck_size_norm,
            float(h_agent['Defuse']),
            float(h_agent['Skip']),
            float(h_agent['Attack']),
            float(h_agent['Safe']),
            float(self.pending_draws[0]),
            opp_cards_norm,
            last_opp_skip,
            last_opp_attack,
            bombs_norm,
            phase_defuse,
        ], dtype=np.float32)
        return obs

    # ---------- Utilidades internas ----------

    def _count_bombs_in_deck(self):
        return sum(1 for c in self.deck if c == 'Bomb')

    def _map_defuse_action_to_position(self, action):
        if action == 3:
            return 'top'
        if action == 4:
            return 'middle'
        if action == 5:
            return 'bottom'
        return 'middle'

    def _reinsert_bomb_for_agent(self, position):
        if position == 'top' or len(self.deck) == 0:
            self.deck.append('Bomb')
        elif position == 'bottom':
            self.deck.insert(0, 'Bomb')
        else:  # middle
            idx = len(self.deck) // 2
            self.deck.insert(idx, 'Bomb')

    def _draw_card(self, player, is_agent):
        if len(self.deck) == 0 or self.done:
            return

        card = self.deck.pop()
        self.last_drawn[player] = card

        if card == 'Bomb':
            if self.hands[player]['Defuse'] > 0:
                self.hands[player]['Defuse'] -= 1
                if is_agent:
                    # agente elige posición después
                    self.phase = 'defuse'
                    self.defuse_pending_for_agent = True
                else:
                    # oponente o humano: se usa política de reinserción automática
                    self.last_opp_action = 3  # usar Defuse
                    pos = self._opponent_defuse_position_choice()
                    self._reinsert_bomb_for_agent(pos)
            else:
                self.done = True
                self.winner = 1 - player
                return
        elif card == 'Defuse':
            self.hands[player]['Defuse'] += 1
        elif card == 'Skip':
            self.hands[player]['Skip'] += 1
        elif card == 'Attack':
            self.hands[player]['Attack'] += 1
        else:
            self.hands[player]['Safe'] += 1

    def _play_agent_turn(self, action):
        assert self.phase == 'action'

        draws = self.pending_draws[0]

        # Skip - reduce draws by 1
        if action == 1 and self.hands[0]['Skip'] > 0:
            self.hands[0]['Skip'] -= 1
            draws = max(0, draws - 1)
        # Attack - opponent must draw 2, agent draws 0
        elif action == 2 and self.hands[0]['Attack'] > 0:
            self.hands[0]['Attack'] -= 1
            self.pending_draws[1] = 2
            draws = 0

        # Reset pending_draws for next turn BEFORE drawing
        self.pending_draws[0] = 1

        # Draw the required number of cards
        for _ in range(draws):
            if self.done:
                break
            self._draw_card(player=0, is_agent=True)
            if self.phase == 'defuse':
                break

    def _opponent_defuse_position_choice(self):
        deck_size = len(self.deck)
        bombs = self._count_bombs_in_deck()
        bomb_prob = (bombs / deck_size) if deck_size > 0 else 0.0

        if deck_size > 10 and bomb_prob < 0.3:
            return 'top'
        if deck_size <= 10 and bomb_prob > 0.3:
            return 'bottom'
        return 'middle'

    def _opponent_turn(self):
        if self.done:
            return

        draws = self.pending_draws[1]
        self.last_opp_action = 0

        action = self._opponent_policy()
        if action == 1 and self.hands[1]['Skip'] > 0:
            self.hands[1]['Skip'] -= 1
            draws = max(0, draws - 1)
            self.last_opp_action = 1
        elif action == 2 and self.hands[1]['Attack'] > 0:
            self.hands[1]['Attack'] -= 1
            self.pending_draws[0] = 2
            draws = 0
            self.last_opp_action = 2

        # Draw the required number of cards
        for _ in range(draws):
            if self.done:
                break
            self._draw_card(player=1, is_agent=False)
        
        # Reset pending_draws for next turn
        self.pending_draws[1] = 1

    def _opponent_policy(self):
        """V1: SIMPLE HEURISTIC"""
        deck_size = len(self.deck)
        bombs = self._count_bombs_in_deck()
        bomb_prob = (bombs / deck_size) if deck_size > 0 else 0.0
        h = self.hands[1]

        # Simple policy
        if deck_size <= 10 and bomb_prob > 0.1 and h['Skip'] > 0:
            return 1  # Skip
        if bomb_prob > 0.15 and h['Attack'] > 0 and random.random() < 0.7:
            return 2  # Attack
        return 0  # Draw


# ---------- Helpers de espacio de acciones ----------

def valid_actions_from_state(state):
    """Devuelve las acciones válidas dado el estado (fase normal o defuse)."""
    phase_defuse = state[10] > 0.5
    if phase_defuse:
        return [3, 4, 5]
    else:
        return [0, 1, 2]


def action_name_from_state(action, state):
    """Nombre legible de la acción dado el estado (fase)."""
    phase_defuse = state[10] > 0.5

    if not phase_defuse:
        mapping = {
            0: 'No jugar / Robar',
            1: 'Jugar Skip',
            2: 'Jugar Attack',
            3: 'No-op (fase normal)',
            4: 'No-op (fase normal)',
            5: 'No-op (fase normal)',
        }
    else:
        mapping = {
            3: 'Defuse -> Bomb TOP',
            4: 'Defuse -> Bomb MIDDLE',
            5: 'Defuse -> Bomb BOTTOM',
            0: 'Acción inválida (ignorada)',
            1: 'Acción inválida (ignorada)',
            2: 'Acción inválida (ignorada)',
        }

    return mapping.get(action, f'Desconocida({action})')


if __name__ == "__main__":
    # Pequeña prueba rápida
    env = ExplodingKittensEnv()
    obs = env.reset()
    print("Dimensión de la observación:", obs.shape)
    env.render()
