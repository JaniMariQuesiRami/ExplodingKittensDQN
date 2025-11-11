# play_pygame.py
# Jugar contra el agente DQN con una UI mejorada en Pygame

import pygame
import torch
import math
import random

from exploding_env import ExplodingKittensEnv, valid_actions_from_state, action_name_from_state
from dqn_training import QNetwork, device


WIDTH, HEIGHT = 1200, 800
FPS = 60

# Exploding Kittens color palette
BG_COLOR = (255, 240, 220)  # Warm beige
CARD_BACK = (255, 100, 150)  # Hot pink
BOMB_COLOR = (255, 50, 50)  # Red
DEFUSE_COLOR = (100, 200, 255)  # Blue
SKIP_COLOR = (255, 200, 50)  # Yellow
ATTACK_COLOR = (255, 100, 50)  # Orange
SAFE_COLOR = (150, 255, 150)  # Green
DECK_COLOR = (200, 100, 200)  # Purple
TEXT_COLOR = (50, 50, 50)  # Dark gray
TITLE_COLOR = (255, 50, 100)  # Pink

# Animation states
class Animation:
    def __init__(self):
        self.drawing_card = False
        self.card_pos = [0, 0]
        self.card_target = [0, 0]
        self.card_progress = 0
        self.card_type = None
        self.explosion = False
        self.explosion_frame = 0
        self.message_queue = []
        self.current_message = ""
        self.message_timer = 0
        self.delay_timer = 0
        # New: card play animation
        self.playing_card = False
        self.play_card_pos = [0, 0]
        self.play_card_target = [0, 0]
        self.play_card_progress = 0
        self.play_card_type = None
        self.play_card_from = "agent"  # "agent" or "human"
        
class GameLog:
    def __init__(self):
        self.entries = []
        self.max_entries = 20
        
    def add(self, message):
        self.entries.append(message)
        if len(self.entries) > self.max_entries:
            self.entries.pop(0)


def load_agent_model():
    env_tmp = ExplodingKittensEnv()
    state_dim = len(env_tmp._get_obs())
    action_dim = 6

    q_net = QNetwork(state_dim, action_dim).to(device)
    q_net.load_state_dict(torch.load("dqn_exploding_kittens.pth", map_location=device))
    q_net.eval()
    return q_net


def agent_turn(env, q_net, anim, game_log):
    """Un turno completo del agente con animaciones y log."""
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
            msg = "[AGENT] Used Defuse and reinserted bomb"
            logs.append(msg)
            game_log.add(msg)
            env.current_player = 1
            break
        else:
            msg = f"[AGENT] Action: {a_name}"
            logs.append(msg)
            game_log.add(msg)
            
            # Get current state before the action
            draws_before = env.pending_draws[0]
            cards_before = sum(env.hands[0].values())
            
            # Trigger card play animation for Skip/Attack
            if action == 1 and env.hands[0]['Skip'] > 0:  # Skip
                anim.playing_card = True
                anim.play_card_type = 'Skip'
                anim.play_card_from = 'agent'
                anim.play_card_pos = [200, 150]
                anim.play_card_target = [WIDTH // 2 - 50, HEIGHT // 2 - 75]
                anim.play_card_progress = 0
                anim.delay_timer = 0.8  # Time for card play animation
            elif action == 2 and env.hands[0]['Attack'] > 0:  # Attack
                anim.playing_card = True
                anim.play_card_type = 'Attack'
                anim.play_card_from = 'agent'
                anim.play_card_pos = [200, 150]
                anim.play_card_target = [WIDTH // 2 - 50, HEIGHT // 2 - 75]
                anim.play_card_progress = 0
                anim.delay_timer = 0.8  # Time for card play animation
            
            # Use the environment's method (now fixed)
            env._play_agent_turn(action)
            
            # Log what happened based on the action
            if action == 1:  # Skip was played
                log_msg = f"[AGENT] Played SKIP - draws reduced by 1"
                logs.append(log_msg)
                game_log.add(log_msg)
            elif action == 2:  # Attack was played
                log_msg = "[AGENT] Played ATTACK - Human must draw 2!"
                logs.append(log_msg)
                game_log.add(log_msg)
            
            # Calculate how many cards were drawn
            cards_after = sum(env.hands[0].values())
            cards_drawn = cards_after - cards_before
            
            # Adjust for action cards played (Skip/Attack reduce hand count)
            if action == 1:  # Skip used
                cards_drawn += 1  # We removed a Skip card
            elif action == 2:  # Attack used
                cards_drawn += 1  # We removed an Attack card
            
            # Log all draws
            if cards_drawn > 0:
                anim.drawing_card = True
                anim.card_pos = [WIDTH // 2 - 50, 250]
                anim.card_target = [200, 150]
                anim.card_progress = 0
                
                if env.last_drawn[0] is not None:
                    anim.card_type = env.last_drawn[0]
                    drawn_msg = f"[AGENT] Drew {cards_drawn} card(s) - last: {env.last_drawn[0]}"
                    logs.append(drawn_msg)
                    game_log.add(drawn_msg)
                    
                    if env.last_drawn[0] == "Bomb":
                        # CRITICAL: Trigger explosion animation with longer duration
                        anim.explosion = True
                        anim.explosion_frame = 0
                        anim.delay_timer = max(anim.delay_timer, 2.0)  # At least 2 seconds to see explosion
                        
                        if env.hands[0]["Defuse"] > 0 or env.phase == "defuse":
                            bomb_msg = "[AGENT] 💣 HIT A BOMB! Using Defuse..."
                            game_log.add(bomb_msg)
                            logs.append(bomb_msg)
                        else:
                            bomb_msg = "[AGENT] 💣 HIT A BOMB! NO DEFUSE - EXPLODED!"
                            game_log.add(bomb_msg)
                            logs.append(bomb_msg)
                    else:
                        # Normal card draw
                        anim.delay_timer = 1.0 * cards_drawn
            
            if env.done:
                done = True
                break
            if env.phase == "defuse" and env.defuse_pending_for_agent:
                continue
            else:
                env.current_player = 1
                break

    return " | ".join(logs), done




def draw_card_icon(surface, x, y, width, height, card_type):
    """Dibuja el icono de una carta usando formas geométricas."""
    center_x = x + width // 2
    center_y = y + height // 3
    
    if card_type == 'Bomb':
        # Bomba: círculo con mecha
        pygame.draw.circle(surface, (0, 0, 0), (center_x, center_y + 5), 20)
        pygame.draw.circle(surface, (50, 50, 50), (center_x, center_y + 5), 18)
        # Mecha
        pygame.draw.line(surface, (100, 50, 0), (center_x, center_y - 15), (center_x + 8, center_y - 30), 3)
        # Chispa
        pygame.draw.circle(surface, (255, 200, 0), (center_x + 8, center_y - 30), 4)
        
    elif card_type == 'Defuse':
        # Escudo
        points = [
            (center_x, center_y - 20),
            (center_x + 15, center_y - 15),
            (center_x + 15, center_y + 5),
            (center_x + 10, center_y + 15),
            (center_x, center_y + 20),
            (center_x - 10, center_y + 15),
            (center_x - 15, center_y + 5),
            (center_x - 15, center_y - 15),
        ]
        pygame.draw.polygon(surface, (0, 100, 200), points)
        pygame.draw.polygon(surface, (0, 0, 0), points, 2)
        # Cruz en el escudo
        pygame.draw.line(surface, (255, 255, 255), (center_x, center_y - 10), (center_x, center_y + 10), 3)
        pygame.draw.line(surface, (255, 255, 255), (center_x - 8, center_y), (center_x + 8, center_y), 3)
        
    elif card_type == 'Skip':
        # Flechas de saltar
        # Flecha 1
        points1 = [(center_x - 10, center_y), (center_x - 5, center_y - 8), (center_x - 5, center_y + 8)]
        pygame.draw.polygon(surface, (0, 0, 0), points1)
        pygame.draw.rect(surface, (0, 0, 0), (center_x - 18, center_y - 5, 8, 10))
        # Flecha 2
        points2 = [(center_x + 10, center_y), (center_x + 5, center_y - 8), (center_x + 5, center_y + 8)]
        pygame.draw.polygon(surface, (0, 0, 0), points2)
        pygame.draw.rect(surface, (0, 0, 0), (center_x + 2, center_y - 5, 8, 10))
        
    elif card_type == 'Attack':
        # Espada
        # Hoja
        pygame.draw.polygon(surface, (0, 0, 0), [
            (center_x, center_y - 20),
            (center_x + 6, center_y + 10),
            (center_x - 6, center_y + 10)
        ])
        # Guardia
        pygame.draw.rect(surface, (150, 150, 150), (center_x - 12, center_y + 8, 24, 4))
        # Empuñadura
        pygame.draw.rect(surface, (100, 50, 0), (center_x - 3, center_y + 12, 6, 12))
        
    else:  # Safe - Gato
        # Cabeza del gato
        pygame.draw.circle(surface, (255, 200, 150), (center_x, center_y), 15)
        pygame.draw.circle(surface, (0, 0, 0), (center_x, center_y), 15, 2)
        # Orejas
        points_left = [(center_x - 12, center_y - 8), (center_x - 8, center_y - 18), (center_x - 4, center_y - 8)]
        points_right = [(center_x + 12, center_y - 8), (center_x + 8, center_y - 18), (center_x + 4, center_y - 8)]
        pygame.draw.polygon(surface, (255, 200, 150), points_left)
        pygame.draw.polygon(surface, (0, 0, 0), points_left, 2)
        pygame.draw.polygon(surface, (255, 200, 150), points_right)
        pygame.draw.polygon(surface, (0, 0, 0), points_right, 2)
        # Ojos
        pygame.draw.circle(surface, (0, 0, 0), (center_x - 5, center_y - 2), 2)
        pygame.draw.circle(surface, (0, 0, 0), (center_x + 5, center_y - 2), 2)
        # Nariz
        pygame.draw.circle(surface, (255, 150, 150), (center_x, center_y + 3), 2)
        # Bigotes
        pygame.draw.line(surface, (0, 0, 0), (center_x - 15, center_y + 2), (center_x - 5, center_y + 2), 1)
        pygame.draw.line(surface, (0, 0, 0), (center_x + 15, center_y + 2), (center_x + 5, center_y + 2), 1)


def draw_card_visual(surface, x, y, width, height, card_type, count=0, is_back=False):
    """Dibuja una carta visualmente atractiva."""
    if is_back:
        # Carta boca abajo
        pygame.draw.rect(surface, CARD_BACK, (x, y, width, height), border_radius=10)
        pygame.draw.rect(surface, (200, 50, 100), (x, y, width, height), 3, border_radius=10)
        # Patrón de rayas
        for i in range(0, height, 15):
            pygame.draw.line(surface, (230, 80, 130), (x, y + i), (x + width, y + i), 2)
    else:
        # Carta boca arriba según tipo
        if card_type == 'Bomb':
            color = BOMB_COLOR
        elif card_type == 'Defuse':
            color = DEFUSE_COLOR
        elif card_type == 'Skip':
            color = SKIP_COLOR
        elif card_type == 'Attack':
            color = ATTACK_COLOR
        else:  # Safe
            color = SAFE_COLOR
        
        pygame.draw.rect(surface, color, (x, y, width, height), border_radius=10)
        pygame.draw.rect(surface, (0, 0, 0), (x, y, width, height), 3, border_radius=10)
        
        # Dibujar icono
        draw_card_icon(surface, x, y, width, height, card_type)
        
        # Nombre
        font_small = pygame.font.SysFont(None, 20, bold=True)
        name_surf = font_small.render(card_type, True, (0, 0, 0))
        name_rect = name_surf.get_rect(center=(x + width // 2, y + 2 * height // 3))
        surface.blit(name_surf, name_rect)
        
        # Contador si hay múltiples
        if count > 1:
            font_count = pygame.font.SysFont(None, 30, bold=True)
            count_surf = font_count.render(f"x{count}", True, (255, 255, 255))
            count_bg = pygame.Rect(x + width - 35, y + 5, 30, 30)
            pygame.draw.circle(surface, (0, 0, 0), count_bg.center, 18)
            count_rect = count_surf.get_rect(center=count_bg.center)
            surface.blit(count_surf, count_rect)



def draw_hand(surface, x, y, hand_dict, spacing=20):
    """Dibuja la mano de un jugador con cartas individuales."""
    card_width, card_height = 80, 110
    current_x = x
    
    card_types = ['Defuse', 'Skip', 'Attack', 'Safe']
    for card_type in card_types:
        count = hand_dict.get(card_type, 0)
        if count > 0:
            draw_card_visual(surface, current_x, y, card_width, card_height, card_type, count)
            current_x += card_width + spacing


def draw_deck(surface, x, y, deck_size, bombs_count):
    """Dibuja el mazo con estilo."""
    card_width, card_height = 100, 140
    
    # Sombra del mazo
    for i in range(min(5, deck_size)):
        offset = i * 3
        pygame.draw.rect(surface, (150, 150, 150), 
                        (x + offset, y + offset, card_width, card_height), 
                        border_radius=10)
    
    # Carta superior
    draw_card_visual(surface, x, y, card_width, card_height, None, is_back=True)
    
    # Info del mazo
    font = pygame.font.SysFont(None, 28, bold=True)
    
    # Contador de cartas
    count_surf = font.render(str(deck_size), True, (255, 255, 255))
    count_bg = pygame.Rect(x + card_width - 40, y - 10, 50, 40)
    pygame.draw.ellipse(surface, DECK_COLOR, count_bg)
    pygame.draw.ellipse(surface, (0, 0, 0), count_bg, 3)
    count_rect = count_surf.get_rect(center=count_bg.center)
    surface.blit(count_surf, count_rect)
    
    # Contador de bombas con icono
    bomb_text = f"Bombs: {bombs_count}"
    font_small = pygame.font.SysFont(None, 24, bold=True)
    bomb_surf = font_small.render(bomb_text, True, (255, 255, 255))
    bomb_bg = pygame.Rect(x - 15, y + card_height + 10, 130, 35)
    pygame.draw.rect(surface, BOMB_COLOR, bomb_bg, border_radius=8)
    pygame.draw.rect(surface, (0, 0, 0), bomb_bg, 3, border_radius=8)
    
    # Dibujar mini icono de bomba
    bomb_icon_x = x - 5
    bomb_icon_y = y + card_height + 27
    pygame.draw.circle(surface, (0, 0, 0), (bomb_icon_x, bomb_icon_y), 6)
    pygame.draw.circle(surface, (50, 50, 50), (bomb_icon_x, bomb_icon_y), 5)
    pygame.draw.line(surface, (255, 200, 0), (bomb_icon_x, bomb_icon_y - 6), (bomb_icon_x + 3, bomb_icon_y - 10), 2)
    
    bomb_rect = bomb_surf.get_rect(center=(x + 50, y + card_height + 27))
    surface.blit(bomb_surf, bomb_rect)


def draw_explosion(surface, x, y, frame):
    """Dibuja una animación de explosión."""
    max_radius = 150
    num_circles = 5
    
    for i in range(num_circles):
        progress = (frame + i * 3) % 30 / 30
        radius = int(max_radius * progress)
        alpha = int(255 * (1 - progress))
        
        colors = [(255, 100, 0), (255, 200, 0), (255, 50, 50)]
        color = colors[i % len(colors)]
        
        # Crear superficie con alpha
        circle_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(circle_surf, (*color, alpha), (radius, radius), radius)
        surface.blit(circle_surf, (x - radius, y - radius))
    
    # Texto de explosión
    font_exp = pygame.font.SysFont(None, int(80 + 20 * math.sin(frame * 0.5)), bold=True)
    exp_surf = font_exp.render("BOOM!", True, (255, 255, 255))
    exp_rect = exp_surf.get_rect(center=(x, y))
    
    # Sombra
    shadow_surf = font_exp.render("BOOM!", True, (0, 0, 0))
    shadow_rect = shadow_surf.get_rect(center=(x + 3, y + 3))
    surface.blit(shadow_surf, shadow_rect)
    surface.blit(exp_surf, exp_rect)


def draw_animated_card(surface, anim):
    """Dibuja una carta en movimiento."""
    if anim.drawing_card:
        x = int(anim.card_pos[0])
        y = int(anim.card_pos[1])
        
        # Efecto de rotación durante el movimiento
        rotation = math.sin(anim.card_progress * math.pi) * 20
        
        draw_card_visual(surface, x - 40, y - 55, 80, 110, anim.card_type)


def draw_played_card(surface, anim):
    """Dibuja una carta siendo jugada (Skip/Attack) con animación."""
    if anim.playing_card:
        x = int(anim.play_card_pos[0])
        y = int(anim.play_card_pos[1])
        
        # Efecto de escala durante el movimiento (crece hacia el centro)
        scale = 1.0 + anim.play_card_progress * 0.3
        width = int(80 * scale)
        height = int(110 * scale)
        
        # Centrar la carta escalada
        x_centered = x - width // 2 + 40
        y_centered = y - height // 2 + 55
        
        draw_card_visual(surface, x_centered, y_centered, width, height, anim.play_card_type)
        
        # Texto flotante indicando la acción
        font = pygame.font.SysFont(None, int(32 + 8 * anim.play_card_progress), bold=True)
        action_text = f"{anim.play_card_from.upper()} plays {anim.play_card_type}!"
        text_surf = font.render(action_text, True, ATTACK_COLOR if anim.play_card_type == 'Attack' else SKIP_COLOR)
        text_rect = text_surf.get_rect(center=(x + 40, y - 40))
        
        # Sombra del texto
        shadow_surf = font.render(action_text, True, (50, 50, 50))
        shadow_rect = shadow_surf.get_rect(center=(x + 42, y - 38))
        surface.blit(shadow_surf, shadow_rect)
        surface.blit(text_surf, text_rect)


def draw_message_box(surface, message, y_pos):
    """Dibuja un cuadro de mensaje estilizado."""
    if not message:
        return
    
    font = pygame.font.SysFont(None, 32, bold=True)
    text_surf = font.render(message, True, TEXT_COLOR)
    text_rect = text_surf.get_rect(center=(WIDTH // 2, y_pos))
    
    # Fondo del mensaje
    padding = 20
    bg_rect = pygame.Rect(text_rect.x - padding, text_rect.y - padding,
                          text_rect.width + 2 * padding, text_rect.height + 2 * padding)
    pygame.draw.rect(surface, (255, 255, 200), bg_rect, border_radius=15)
    pygame.draw.rect(surface, TITLE_COLOR, bg_rect, 4, border_radius=15)
    
    surface.blit(text_surf, text_rect)


def draw_player_info(surface, x, y, name, hand, pending_draws, is_active):
    """Dibuja información del jugador con estilo."""
    font_name = pygame.font.SysFont(None, 36, bold=True)
    
    # Fondo
    bg_rect = pygame.Rect(x - 10, y - 10, 480, 180)
    border_color = TITLE_COLOR if is_active else (150, 150, 150)
    border_width = 5 if is_active else 2
    
    pygame.draw.rect(surface, (255, 255, 255), bg_rect, border_radius=15)
    pygame.draw.rect(surface, border_color, bg_rect, border_width, border_radius=15)
    
    # Nombre
    name_surf = font_name.render(name, True, TITLE_COLOR if is_active else TEXT_COLOR)
    surface.blit(name_surf, (x, y))
    
    # Cartas
    draw_hand(surface, x, y + 50, hand, spacing=15)
    
    # Indicador de turnos pendientes
    if pending_draws > 0:
        font_draws = pygame.font.SysFont(None, 28, bold=True)
        draws_text = f"Must draw: {pending_draws}" if pending_draws > 0 else "Turn complete"
        draws_surf = font_draws.render(draws_text, True, ATTACK_COLOR if pending_draws > 1 else TEXT_COLOR)
        surface.blit(draws_surf, (x + 300, y + 5))


def draw_button(surface, rect, text, color, hover=False):
    """Dibuja un botón estilizado."""
    button_color = tuple(min(c + 30, 255) for c in color) if hover else color
    
    # Sombra
    shadow_rect = rect.copy()
    shadow_rect.y += 5
    pygame.draw.rect(surface, (100, 100, 100), shadow_rect, border_radius=12)
    
    # Botón
    pygame.draw.rect(surface, button_color, rect, border_radius=12)
    pygame.draw.rect(surface, (0, 0, 0), rect, 4, border_radius=12)
    
    # Texto
    font = pygame.font.SysFont(None, 32, bold=True)
    text_surf = font.render(text, True, (255, 255, 255))
    text_rect = text_surf.get_rect(center=rect.center)
    surface.blit(text_surf, text_rect)


def draw_defuse_panel(surface, x, y, width, height):
    """Dibuja el panel para elegir dónde poner la bomba después de defuse."""
    # Fondo oscuro
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 200))
    surface.blit(overlay, (0, 0))
    
    # Panel principal
    panel_rect = pygame.Rect(x, y, width, height)
    pygame.draw.rect(surface, (255, 255, 255), panel_rect, border_radius=20)
    pygame.draw.rect(surface, BOMB_COLOR, panel_rect, 5, border_radius=20)
    
    # Título
    font_title = pygame.font.SysFont(None, 48, bold=True)
    title_surf = font_title.render("DEFUSE! Choose bomb position:", True, BOMB_COLOR)
    title_rect = title_surf.get_rect(center=(x + width // 2, y + 40))
    surface.blit(title_surf, title_rect)
    
    # Instrucciones
    font_inst = pygame.font.SysFont(None, 28)
    inst_surf = font_inst.render("Where do you want to place the bomb back in the deck?", True, TEXT_COLOR)
    inst_rect = inst_surf.get_rect(center=(x + width // 2, y + 90))
    surface.blit(inst_surf, inst_rect)



def draw_log_panel(surface, game_log, x, y, width, height):
    """Dibuja el panel de log del juego."""
    # Fondo
    pygame.draw.rect(surface, (255, 255, 255), (x, y, width, height), border_radius=10)
    pygame.draw.rect(surface, (100, 100, 100), (x, y, width, height), 3, border_radius=10)
    
    # Título
    font_title = pygame.font.SysFont(None, 32, bold=True)
    title_surf = font_title.render("GAME LOG", True, TITLE_COLOR)
    surface.blit(title_surf, (x + 10, y + 10))
    
    # Entradas del log
    font_log = pygame.font.SysFont(None, 20)
    log_y = y + 50
    
    # Mostrar las últimas entradas (desde la más reciente)
    visible_entries = game_log.entries[-15:]  # Últimas 15 entradas
    for entry in reversed(visible_entries):
        if log_y + 25 > y + height:
            break
        
        # Dividir entrada larga en múltiples líneas si es necesario
        if len(entry) > 45:
            words = entry.split()
            lines = []
            current_line = ""
            for word in words:
                if len(current_line + word) < 45:
                    current_line += word + " "
                else:
                    lines.append(current_line)
                    current_line = word + " "
            if current_line:
                lines.append(current_line)
            
            for line in lines:
                log_surf = font_log.render(line.strip(), True, TEXT_COLOR)
                surface.blit(log_surf, (x + 10, log_y))
                log_y += 20
        else:
            log_surf = font_log.render(entry, True, TEXT_COLOR)
            surface.blit(log_surf, (x + 10, log_y))
            log_y += 25


def draw_log_button(surface, rect, hover=False):
    """Dibuja el botón circular del log."""
    color = (150, 100, 200) if not hover else (180, 130, 230)
    
    # Sombra
    pygame.draw.circle(surface, (100, 100, 100), (rect.centerx + 2, rect.centery + 2), rect.width // 2)
    
    # Círculo principal
    pygame.draw.circle(surface, color, rect.center, rect.width // 2)
    pygame.draw.circle(surface, (0, 0, 0), rect.center, rect.width // 2, 3)
    
    # Icono de libro/log
    font_icon = pygame.font.SysFont(None, 40, bold=True)
    icon_surf = font_icon.render("LOG", True, (255, 255, 255))
    icon_rect = icon_surf.get_rect(center=rect.center)
    surface.blit(icon_surf, icon_rect)




def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Exploding Kittens RL")
    clock = pygame.time.Clock()

    q_net = load_agent_model()
    env = ExplodingKittensEnv()
    env.reset()

    running = True
    done = False
    turn = "human"  # Start with human player
    msg = "Game started! Your turn."
    defuse_mode = False  # Whether human is choosing bomb position
    
    # Animaciones y log
    anim = Animation()
    game_log = GameLog()
    game_log.add("=== GAME START ===")
    game_log.add("Human player goes first")
    
    # Estado de la UI
    waiting_for_animation = False
    show_log = False

    # Botones para humano (más grandes y centrados)
    btn_draw = pygame.Rect(WIDTH // 2 - 350, 650, 200, 60)
    btn_skip = pygame.Rect(WIDTH // 2 - 100, 650, 200, 60)
    btn_attack = pygame.Rect(WIDTH // 2 + 150, 650, 200, 60)
    btn_log = pygame.Rect(20, 20, 80, 80)  # Botón del log
    # Restart button (used on end screen)
    btn_restart = pygame.Rect(WIDTH // 2 - 100, HEIGHT // 2 + 120, 200, 60)
    
    # Botones para defuse
    btn_defuse_top = pygame.Rect(WIDTH // 2 - 250, 350, 150, 80)
    btn_defuse_middle = pygame.Rect(WIDTH // 2 - 75, 350, 150, 80)
    btn_defuse_bottom = pygame.Rect(WIDTH // 2 + 100, 350, 150, 80)
    
    # Botón de restart
    btn_restart = pygame.Rect(WIDTH // 2 - 100, HEIGHT // 2 + 100, 200, 60)
    
    # Mouse hover
    mouse_pos = (0, 0)

    while running:
        dt = clock.tick(FPS) / 1000.0  # Delta time en segundos
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.MOUSEMOTION:
                mouse_pos = event.pos
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                
                # Restart button on game over screen
                # Restart button on game over screen
                if done and btn_restart.collidepoint(mx, my):
                    # Reset everything
                    env.reset()
                    done = False
                    turn = "human"
                    msg = "Game restarted! Your turn."
                    defuse_mode = False
                    waiting_for_animation = False
                    anim = Animation()
                    game_log = GameLog()
                    game_log.add("=== GAME RESTARTED ===")
                    game_log.add("Human player goes first")
                    continue
                # Toggle log
                if btn_log.collidepoint(mx, my):
                    show_log = not show_log
                    continue
                
                # Defuse mode - choosing bomb position
                if defuse_mode:
                    position = None
                    if btn_defuse_top.collidepoint(mx, my):
                        position = 'top'
                    elif btn_defuse_middle.collidepoint(mx, my):
                        position = 'middle'
                    elif btn_defuse_bottom.collidepoint(mx, my):
                        position = 'bottom'
                    
                    if position:
                        # Reinsert bomb at chosen position
                        if position == 'top' or len(env.deck) == 0:
                            env.deck.append('Bomb')
                        elif position == 'bottom':
                            env.deck.insert(0, 'Bomb')
                        else:  # middle
                            idx = len(env.deck) // 2
                            env.deck.insert(idx, 'Bomb')
                        
                        game_log.add(f"[HUMAN] Placed bomb at {position} of deck")
                        msg = f"Bomb placed at {position} of deck! Agent's turn next."
                        defuse_mode = False
                        
                        # Human turn is complete after defusing, agent goes next
                        if not env.done:
                            env.current_player = 0
                            env.turn_count += 1
                            waiting_for_animation = True
                            anim.delay_timer = 0.8  # Short delay before agent turn
                        continue

                if not done and not waiting_for_animation and turn == "human":
                    human_action = None
                    
                    if btn_draw.collidepoint(mx, my):
                        human_action = 0
                    elif btn_skip.collidepoint(mx, my):
                        human_action = 1
                    elif btn_attack.collidepoint(mx, my):
                        human_action = 2

                    if human_action is not None:
                        # Set current player to human
                        env.current_player = 1
                        
                        # Aplicar acción humana
                        draws = env.pending_draws[1]
                        env.last_opp_action = 0

                        parts = []

                        if human_action == 1 and env.hands[1]["Skip"] > 0:
                            env.hands[1]["Skip"] -= 1
                            draws = max(0, draws - 1)  # Skip reduces draws by 1
                            env.last_opp_action = 1
                            msg_action = f"[HUMAN] Played SKIP - draws reduced to {draws}"
                            parts.append(msg_action)
                            game_log.add(msg_action)
                        elif human_action == 2 and env.hands[1]["Attack"] > 0:
                            env.hands[1]["Attack"] -= 1
                            env.pending_draws[0] = 2  # Agent must draw 2
                            draws = 0
                            env.last_opp_action = 2
                            msg_action = "[HUMAN] Played ATTACK - Agent must draw 2!"
                            parts.append(msg_action)
                            game_log.add(msg_action)
                        else:
                            if human_action in [1, 2]:
                                msg_error = "[HUMAN] Card not available"
                                parts.append(msg_error)
                                game_log.add(msg_error)

                        # Animación de robar cartas
                        if draws > 0:
                            waiting_for_animation = True
                            anim.delay_timer = 1.0 * draws  # Total time for all cards
                            
                        for i in range(draws):
                            if env.done:
                                break
                            
                            # Iniciar animación
                            anim.drawing_card = True
                            anim.card_pos = [WIDTH // 2 - 50, 250]
                            anim.card_target = [WIDTH - 200, 550]
                            anim.card_progress = 0
                            
                            # Draw card manually for human
                            if len(env.deck) > 0:
                                card = env.deck.pop()
                                env.last_drawn[1] = card
                                
                                anim.card_type = card
                                msg_draw = f"[HUMAN] Drew: {card}"
                                parts.append(msg_draw)
                                game_log.add(msg_draw)
                                
                                if card == "Bomb":
                                    anim.explosion = True
                                    anim.explosion_frame = 0
                                    
                                    if env.hands[1]["Defuse"] > 0:
                                        env.hands[1]["Defuse"] -= 1
                                        game_log.add("[HUMAN] Used Defuse! Choose where to place bomb.")
                                        defuse_mode = True
                                        msg = "You drew a BOMB! Used Defuse. Choose position."
                                        waiting_for_animation = False  # Allow clicking immediately
                                        break  # Stop drawing more cards
                                    else:
                                        env.done = True
                                        env.winner = 0
                                        game_log.add("[HUMAN] EXPLODED! Game Over.")
                                        break
                                elif card == 'Defuse':
                                    env.hands[1]['Defuse'] += 1
                                elif card == 'Skip':
                                    env.hands[1]['Skip'] += 1
                                elif card == 'Attack':
                                    env.hands[1]['Attack'] += 1
                                else:
                                    env.hands[1]['Safe'] += 1

                        # Reset pending_draws AFTER drawing all cards
                        env.pending_draws[1] = 1

                        if env.done:
                            done = True
                            msg = " | ".join(parts)
                        else:
                            env.current_player = 0  # Next is agent
                            env.turn_count += 1
                            if env.turn_count >= env.max_turns:
                                env.done = True
                                env.winner = None
                                done = True
                                parts.append("Time limit - Draw")
                                msg = " | ".join(parts)
                                game_log.add("=== GAME OVER: DRAW ===")
                            else:
                                msg = " | ".join(parts) if parts else "Turn complete"
                                # Human turn complete - agent goes next after animation
                                if not waiting_for_animation:
                                    # No animation, switch immediately
                                    turn = "agent"
                                # else: animation timer will switch turn when done

        # Actualizar animaciones
        
        # Card play animation (Skip/Attack being played)
        if anim.playing_card:
            anim.play_card_progress += dt * 2.5  # Fast animation
            if anim.play_card_progress >= 1:
                anim.play_card_progress = 1
                anim.playing_card = False
            
            # Interpolación suave (ease-in-out)
            t = anim.play_card_progress
            t = t * t * (3 - 2 * t)
            anim.play_card_pos[0] = anim.play_card_pos[0] * (1 - t) + anim.play_card_target[0] * t
            anim.play_card_pos[1] = anim.play_card_pos[1] * (1 - t) + anim.play_card_target[1] * t
        
        # Card draw animation
        if anim.drawing_card:
            anim.card_progress += dt * 2.0  # Velocidad de animación más rápida
            if anim.card_progress >= 1:
                anim.card_progress = 1
                anim.drawing_card = False
            
            # Interpolación suave (ease-in-out)
            t = anim.card_progress
            t = t * t * (3 - 2 * t)
            start_x = WIDTH // 2 - 50
            start_y = 250
            anim.card_pos[0] = start_x * (1 - t) + anim.card_target[0] * t
            anim.card_pos[1] = start_y * (1 - t) + anim.card_target[1] * t
        
        if anim.explosion:
            anim.explosion_frame += 1
            if anim.explosion_frame > 60:  # 1 segundo de explosión
                anim.explosion = False
                anim.explosion_frame = 0
        
        # Timer de delay
        if anim.delay_timer > 0:
            anim.delay_timer -= dt
            if anim.delay_timer <= 0:
                anim.delay_timer = 0
                if waiting_for_animation:
                    waiting_for_animation = False
                    # Animation complete - switch turn based on whose turn should be next
                    if not done:
                        # env.current_player was set when the action completed
                        turn = "agent" if env.current_player == 0 else "human"
                        msg = "Agent's turn..." if turn == "agent" else "Your turn!"

        # Turno del agente con delay
        if not done and not waiting_for_animation and turn == "agent" and anim.delay_timer <= 0:
            # Pequeño delay antes de que el agente juegue
            if not hasattr(anim, 'agent_delay_started') or not anim.agent_delay_started:
                anim.agent_delay_started = True
                anim.delay_timer = 0.5  # Medio segundo de delay antes del turno del agente
                msg = "Agent is thinking..."
            else:
                # Agent takes their turn
                env.current_player = 0
                msg_agent, done_agent = agent_turn(env, q_net, anim, game_log)
                msg = msg_agent
                anim.agent_delay_started = False
                
                # Decide whether we need to wait for animations triggered by agent
                if anim.delay_timer > 0 or anim.drawing_card:
                    waiting_for_animation = True
                
                if done_agent:
                    done = True
                    if env.winner == 0:
                        game_log.add("=== GAME OVER: AGENT WINS ===")
                    elif env.winner == 1:
                        game_log.add("=== GAME OVER: HUMAN WINS ===")
                else:
                    # Agent turn complete - set next player to human
                    env.current_player = 1
                    # If no animation, switch turn immediately
                    if not waiting_for_animation:
                        turn = "human"
                        msg = "Your turn!"
                    # else: animation timer will switch turn when done

        # DIBUJAR TODO
        screen.fill(BG_COLOR)
        
        # Título con estilo
        font_title = pygame.font.SysFont(None, 72, bold=True)
        title_surf = font_title.render("EXPLODING KITTENS", True, TITLE_COLOR)
        title_rect = title_surf.get_rect(center=(WIDTH // 2, 40))
        
        # Sombra del título
        shadow_surf = font_title.render("EXPLODING KITTENS", True, (200, 200, 200))
        shadow_rect = shadow_surf.get_rect(center=(WIDTH // 2 + 3, 43))
        screen.blit(shadow_surf, shadow_rect)
        screen.blit(title_surf, title_rect)
        
        # Mazo central
        deck_size = len(env.deck)
        bombs = env._count_bombs_in_deck()
        draw_deck(screen, WIDTH // 2 - 50, 200, deck_size, bombs)
        
        # Información de jugadores
        h0 = env.hands[0]
        h1 = env.hands[1]
        
        draw_player_info(screen, 50, 100, "AGENT", h0, env.pending_draws[0], 
                        turn == "agent" and not done)
        draw_player_info(screen, 670, 480, "YOU (Human)", h1, env.pending_draws[1], 
                        turn == "human" and not done)
        
        # Mensaje de estado
        draw_message_box(screen, msg, 400)
        
        # Botones del humano
        if not done and turn == "human" and not waiting_for_animation:
            hover_draw = btn_draw.collidepoint(mouse_pos)
            hover_skip = btn_skip.collidepoint(mouse_pos)
            hover_attack = btn_attack.collidepoint(mouse_pos)
            
            draw_button(screen, btn_draw, "Draw Card", SAFE_COLOR, hover_draw)
            
            skip_enabled = env.hands[1]["Skip"] > 0
            skip_color = SKIP_COLOR if skip_enabled else (150, 150, 150)
            draw_button(screen, btn_skip, "Skip", skip_color, hover_skip and skip_enabled)
            
            attack_enabled = env.hands[1]["Attack"] > 0
            attack_color = ATTACK_COLOR if attack_enabled else (150, 150, 150)
            draw_button(screen, btn_attack, "Attack", attack_color, hover_attack and attack_enabled)
        
        # Botón de log
        hover_log = btn_log.collidepoint(mouse_pos)
        draw_log_button(screen, btn_log, hover_log)
        
        # Panel de log
        if show_log:
            draw_log_panel(screen, game_log, WIDTH - 420, 100, 400, 550)
        
        # Animación de carta siendo jugada (Skip/Attack)
        if anim.playing_card:
            draw_played_card(screen, anim)
        
        # Animación de carta moviéndose (draw)
        if anim.drawing_card:
            draw_animated_card(screen, anim)
        
        # Animación de explosión
        if anim.explosion:
            explosion_x = int(anim.card_pos[0]) if anim.drawing_card else WIDTH // 2
            explosion_y = int(anim.card_pos[1]) if anim.drawing_card else HEIGHT // 2
            draw_explosion(screen, explosion_x, explosion_y, anim.explosion_frame)
        
        # Panel de defuse (si está activo)
        if defuse_mode:
            draw_defuse_panel(screen, WIDTH // 2 - 300, HEIGHT // 2 - 200, 600, 400)
            
            # Botones de posición
            hover_top = btn_defuse_top.collidepoint(mouse_pos)
            hover_middle = btn_defuse_middle.collidepoint(mouse_pos)
            hover_bottom = btn_defuse_bottom.collidepoint(mouse_pos)
            
            draw_button(screen, btn_defuse_top, "Top", DEFUSE_COLOR, hover_top)
            draw_button(screen, btn_defuse_middle, "Middle", SKIP_COLOR, hover_middle)
            draw_button(screen, btn_defuse_bottom, "Bottom", SAFE_COLOR, hover_bottom)
        
        # Mensaje de fin de juego
        if done:
            if env.winner is None:
                end_msg = "DRAW"
                end_color = (150, 150, 150)
            elif env.winner == 0:
                end_msg = "AGENT WINS"
                end_color = (255, 100, 100)
            else:
                end_msg = "YOU WIN!"
                end_color = (100, 255, 100)
            
            # Overlay oscuro
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            screen.blit(overlay, (0, 0))
            
            # Mensaje de victoria/derrota
            font_end = pygame.font.SysFont(None, 96, bold=True)
            end_surf = font_end.render(end_msg, True, end_color)
            end_rect = end_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 50))
            
            # Fondo del mensaje
            bg_rect = pygame.Rect(end_rect.x - 30, end_rect.y - 20,
                                 end_rect.width + 60, end_rect.height + 40)
            pygame.draw.rect(screen, (50, 50, 50), bg_rect, border_radius=20)
            pygame.draw.rect(screen, end_color, bg_rect, 6, border_radius=20)
            
            screen.blit(end_surf, end_rect)
            # Draw restart button
            hover_restart = btn_restart.collidepoint(mouse_pos)
            draw_button(screen, btn_restart, "Restart", (80, 180, 120), hover_restart)
            
            # Botón de restart
            hover_restart = btn_restart.collidepoint(mouse_pos)
            draw_button(screen, btn_restart, "Play Again", (100, 200, 100), hover_restart)


        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
