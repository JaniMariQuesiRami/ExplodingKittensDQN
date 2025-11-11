# play_ascii.py
# Jugar contra el agente DQN en consola (tablero ASCII + input de texto)

import torch

from exploding_env import ExplodingKittensEnv
from exploding_env import valid_actions_from_state, action_name_from_state
from dqn_training import QNetwork, device


def last_opp_action_name(code):
    mapping = {
        0: "Robar (sin carta especial)",
        1: "Jugar Skip",
        2: "Jugar Attack",
        3: "Usar Defuse",
    }
    return mapping.get(code, f"Desconocida({code})")


def print_board(env, last_msg=""):
    deck_size = len(env.deck)
    bombs = env._count_bombs_in_deck()
    h0 = env.hands[0]
    h1 = env.hands[1]

    print("\n" + "=" * 40)
    print("        EXPLODING KITTENS RL")
    print("=" * 40)

    # Mazo
    print("+-----------------------------+")
    print("|            MAZO            |")
    print(f"|  Cartas: {deck_size:3d}           Bombas: {bombs:2d} |")
    print("+-----------------------------+\n")

    # Agente
    print("AGENTE (J0)")
    print(f"  Defuse: {h0['Defuse']}  Skip: {h0['Skip']}  Attack: {h0['Attack']}  Safe: {h0['Safe']}")
    print(f"  Obligación de robo: {env.pending_draws[0]}")
    print("-" * 40)

    # Humano
    print("HUMANO (J1)")
    print(f"  Defuse: {h1['Defuse']}  Skip: {h1['Skip']}  Attack: {h1['Attack']}  Safe: {h1['Safe']}")
    print(f"  Obligación de robo: {env.pending_draws[1]}")
    print("-" * 40)

    if last_msg:
        print(f">> {last_msg}")


def agent_turn(env, q_net):
    """Ejecuta un turno completo del agente (incluye fase de defuse si ocurre)."""
    logs = []
    done = False

    env.current_player = 0

    while True:
        if env.done:
            done = True
            break

        state = env._get_obs()
        valid_actions = valid_actions_from_state(state)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            q_values = q_net(s).cpu().numpy().flatten()
        action = max(valid_actions, key=lambda a: q_values[a])
        a_name = action_name_from_state(action, state)

        if env.phase == "defuse":
            pos_choice = env._map_defuse_action_to_position(action)
            env._reinsert_bomb_for_agent(pos_choice)
            env.defuse_pending_for_agent = False
            env.phase = "action"
            logs.append(f"Agente decide {a_name}")
            env.current_player = 1
            break
        else:
            logs.append(f"Agente elige {a_name}")
            env._play_agent_turn(action)
            if env.done:
                done = True
                break
            if env.phase == "defuse" and env.defuse_pending_for_agent:
                # vuelve a iterar para decidir posición de bomba
                continue
            else:
                env.current_player = 1
                break

    return " | ".join(logs), done


def play_interactive_vs_agent(q_net, max_steps=200):
    env = ExplodingKittensEnv()
    env.reset()
    done = False

    print("\n=== MODO INTERACTIVO: AGENTE (J0) vs HUMANO (J1) ===")
    print("Acciones HUMANO: 0=No jugar/robar, 1=Jugar Skip, 2=Jugar Attack, q=salir")

    step_count = 0
    turn = "agent"
    last_msg = ""

    while not done and step_count < max_steps:
        if turn == "agent":
            msg, done = agent_turn(env, q_net)
            last_msg = msg
            print_board(env, last_msg)
            if done:
                break
            turn = "human"
            continue

        # Turno humano
        print_board(env, last_msg)
        print(f"Última acción del 'oponente' (agente ve humano como opp): {last_opp_action_name(env.last_opp_action)}")

        # Menú humano
        while True:
            try:
                human_in = input("Elige acción HUMANO (0=No jugar, 1=Skip, 2=Attack, q=salir): ").strip()
            except EOFError:
                human_in = "0"
            if human_in.lower() == "q":
                print("Saliendo de la partida interactiva.")
                return
            if human_in in ["0", "1", "2"]:
                human_action = int(human_in)
                break
            print("Entrada inválida, intenta de nuevo.")

        # Aplicar acción del humano
        draws = env.pending_draws[1]
        env.pending_draws[1] = 1
        env.last_opp_action = 0  # para el agente, el humano es 'opp'

        msg_parts = []

        if human_action == 1 and env.hands[1]["Skip"] > 0:
            env.hands[1]["Skip"] -= 1
            draws = 0
            env.last_opp_action = 1
            msg_parts.append("Humano juega SKIP")
        elif human_action == 2 and env.hands[1]["Attack"] > 0:
            env.hands[1]["Attack"] -= 1
            env.pending_draws[0] = 2
            draws = 0
            env.last_opp_action = 2
            msg_parts.append("Humano juega ATTACK")
        else:
            if human_action in [1, 2]:
                msg_parts.append("No tienes esa carta; se considera 'no jugar'")

        # Robos del humano
        for _ in range(draws):
            if env.done:
                break
            env._draw_card(player=1, is_agent=False)
            if env.last_drawn[1] is not None:
                msg_parts.append(f"Humano roba: {env.last_drawn[1]}")

        if env.done:
            done = True
            last_msg = " | ".join(msg_parts)
            print_board(env, last_msg)
            break

        env.current_player = 0
        env.turn_count += 1
        if env.turn_count >= env.max_turns:
            env.done = True
            env.winner = None
            done = True
            last_msg = "Se alcanzó el número máximo de turnos, empate."
            print_board(env, last_msg)
            break

        last_msg = " | ".join(msg_parts)
        turn = "agent"
        step_count += 1

    print("\n=== FIN DE LA PARTIDA INTERACTIVA ===")
    if env.winner is None:
        print("Empate.")
    elif env.winner == 0:
        print("Gana el AGENTE (jugador 0).")
    else:
        print("Gana el HUMANO (jugador 1).")


if __name__ == "__main__":
    # Cargar modelo entrenado
    env_tmp = ExplodingKittensEnv()
    state_dim = len(env_tmp._get_obs())
    action_dim = 6

    q_net = QNetwork(state_dim, action_dim).to(device)
    q_net.load_state_dict(torch.load("dqn_exploding_kittens.pth", map_location=device))
    q_net.eval()
    print("Modelo cargado. Empezamos el juego.\n")

    play_interactive_vs_agent(q_net)

