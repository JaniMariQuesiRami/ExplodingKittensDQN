# play_pygame.py - V2 (Extended version with new cards)
# Jugar contra el agente DQN con UI mejorada y nuevas cartas

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
SEE_FUTURE_COLOR = (180, 100, 255)  # Purple
DRAW_BOTTOM_COLOR = (100, 255, 200)  # Cyan
SHUFFLE_COLOR = (255, 150, 100)  # Coral
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
        self.see_future_cards = None  # For See the Future visualization
        self.see_future_interactive = False  # True si el humano debe elegir una carta
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
    action_dim = 9  # V2: Extended action space

    q_net = QNetwork(state_dim, action_dim).to(device)
    try:
        # Intentar cargar desde output/ primero
        q_net.load_state_dict(torch.load("output/dqn_exploding_kittens_v2.pth", map_location=device))
        print("Loaded V2 model from output/")
    except:
        try:
            # Intentar desde raíz como fallback
            q_net.load_state_dict(torch.load("dqn_exploding_kittens_v2.pth", map_location=device))
            print("Loaded V2 model from root")
        except:
            print("Warning: V2 model not found, using random initialization")
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
        
        # VALIDACIÓN: Verificar que el agente tenga la carta antes de usarla
        hand = env.hands[0]
        card_requirements = {
            1: ('Skip', hand.get('Skip', 0) > 0),
            2: ('Attack', hand.get('Attack', 0) > 0),
            6: ('SeeFuture', hand.get('SeeFuture', 0) > 0),
            7: ('DrawBottom', hand.get('DrawBottom', 0) > 0),
            8: ('Shuffle', hand.get('Shuffle', 0) > 0),
        }
        
        if action in card_requirements:
            card_name, has_card = card_requirements[action]
            if not has_card:
                # Si no tiene la carta, forzar Draw (acción 0)
                action = 0
        
        a_name = action_name_from_state(action, state)

        if env.phase == "defuse":
            pos_choice = env._map_defuse_action_to_position(action)
            env._reinsert_bomb_for_agent(pos_choice)
            env.defuse_pending_for_agent = False
            env.phase = "action"
            msg = "[AGENT] Used Defuse and reinserted bomb"
            logs.append(msg)
            game_log.add(msg)
            
            # Animación de usar Defuse
            anim.playing_card = True
            anim.play_card_type = 'Defuse'
            anim.play_card_from = 'agent'
            anim.play_card_pos = [200, 150]
            anim.play_card_target = [WIDTH // 2 - 50, HEIGHT // 2 - 75]
            anim.play_card_progress = 0
            anim.delay_timer = 1.5  # Mostrar por 1.5 segundos
            
            env.current_player = 1
            break
        else:
            msg = f"[AGENT] Action: {a_name}"
            logs.append(msg)
            game_log.add(msg)
            
            # Get current state before the action
            cards_before = sum(env.hands[0].values())
            
            # Trigger card play animation for Skip/Attack/New cards
            if action == 1 and env.hands[0]['Skip'] > 0:  # Skip
                anim.playing_card = True
                anim.play_card_type = 'Skip'
                anim.play_card_from = 'agent'
                anim.play_card_pos = [200, 150]
                anim.play_card_target = [WIDTH // 2 - 50, HEIGHT // 2 - 75]
                anim.play_card_progress = 0
                anim.delay_timer = 0.8
            elif action == 2 and env.hands[0]['Attack'] > 0:  # Attack
                anim.playing_card = True
                anim.play_card_type = 'Attack'
                anim.play_card_from = 'agent'
                anim.play_card_pos = [200, 150]
                anim.play_card_target = [WIDTH // 2 - 50, HEIGHT // 2 - 75]
                anim.play_card_progress = 0
                anim.delay_timer = 0.8
            elif action == 6 and env.hands[0].get('SeeFuture', 0) > 0:  # See Future
                anim.playing_card = True
                anim.play_card_type = 'SeeFuture'
                anim.play_card_from = 'agent'
                anim.play_card_pos = [200, 150]
                anim.play_card_target = [WIDTH // 2 - 50, HEIGHT // 2 - 75]
                anim.play_card_progress = 0
                anim.delay_timer = 0.8
            elif action == 7 and env.hands[0].get('DrawBottom', 0) > 0:  # Draw Bottom
                anim.playing_card = True
                anim.play_card_type = 'DrawBottom'
                anim.play_card_from = 'agent'
                anim.play_card_pos = [200, 150]
                anim.play_card_target = [WIDTH // 2 - 50, HEIGHT // 2 - 75]
                anim.play_card_progress = 0
                anim.delay_timer = 0.8
            elif action == 8 and env.hands[0].get('Shuffle', 0) > 0:  # Shuffle
                anim.playing_card = True
                anim.play_card_type = 'Shuffle'
                anim.play_card_from = 'agent'
                anim.play_card_pos = [200, 150]
                anim.play_card_target = [WIDTH // 2 - 50, HEIGHT // 2 - 75]
                anim.play_card_progress = 0
                anim.delay_timer = 0.8
            
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
            elif action == 6:  # See the Future
                log_msg = "[AGENT] Played SEE FUTURE - Peeking at deck..."
                logs.append(log_msg)
                game_log.add(log_msg)
                # Show top 3 cards to agent (visual effect, auto-close)
                if len(env.deck) >= 3:
                    anim.see_future_cards = env.deck[-3:]
                    anim.see_future_interactive = False  # Solo visual, no interactivo
                    anim.delay_timer = max(anim.delay_timer, 2.0)
            elif action == 7:  # Draw from Bottom
                log_msg = "[AGENT] Played DRAW FROM BOTTOM"
                logs.append(log_msg)
                game_log.add(log_msg)
            elif action == 8:  # Shuffle
                log_msg = "[AGENT] Played SHUFFLE - Deck shuffled!"
                logs.append(log_msg)
                game_log.add(log_msg)
            
            # Calculate how many cards were drawn
            cards_after = sum(env.hands[0].values())
            cards_drawn = cards_after - cards_before
            
            # Adjust for action cards played
            if action in [1, 2, 6, 7, 8]:  # Action cards used
                cards_drawn += 1
            
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
                        
                        if env.hands[0]["Defuse"] > 0 or env.phase == "defuse":
                            bomb_msg = "[AGENT] 💣 HIT A BOMB! Using Defuse..."
                            game_log.add(bomb_msg)
                            logs.append(bomb_msg)
                            anim.delay_timer = max(anim.delay_timer, 2.5)  # 2.5 segundos para ver explosión con defuse
                        else:
                            bomb_msg = "[AGENT] 💣 HIT A BOMB! NO DEFUSE - EXPLODED!"
                            game_log.add(bomb_msg)
                            logs.append(bomb_msg)
                            anim.delay_timer = max(anim.delay_timer, 3.5)  # 3.5 segundos para ver explosión completa
                    else:
                        # Normal card draw
                        anim.delay_timer = 1.0 * cards_drawn
            
            if env.done:
                done = True
                break
            # IMPORTANTE: Si tiene defuse, esperar a que termine la animación de explosión
            if env.phase == "defuse" and env.defuse_pending_for_agent:
                # Si la explosión está activa, esperar más antes de continuar
                if anim.explosion and anim.explosion_frame < 10:
                    # Todavía mostrando explosión, seguir en el loop
                    continue
                else:
                    # Ya terminó explosión, ahora sí mostrar defuse
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
        pygame.draw.line(surface, (100, 50, 0), (center_x, center_y - 15), (center_x + 8, center_y - 30), 3)
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
        pygame.draw.line(surface, (255, 255, 255), (center_x, center_y - 12), (center_x, center_y + 12), 3)
        pygame.draw.line(surface, (255, 255, 255), (center_x - 8, center_y), (center_x + 8, center_y), 3)
        
    elif card_type == 'Skip':
        # Flechas de skip (>>)
        arrow_size = 12
        for offset in [-8, 8]:
            points = [
                (center_x + offset - 5, center_y - arrow_size),
                (center_x + offset + 5, center_y),
                (center_x + offset - 5, center_y + arrow_size),
            ]
            pygame.draw.polygon(surface, (200, 150, 0), points)
            pygame.draw.polygon(surface, (0, 0, 0), points, 2)
    
    elif card_type == 'Attack':
        # Espada
        pygame.draw.rect(surface, (150, 0, 0), (center_x - 3, center_y - 20, 6, 30))
        pygame.draw.circle(surface, (100, 100, 100), (center_x, center_y + 12), 6)
        pygame.draw.rect(surface, (100, 50, 0), (center_x - 10, center_y + 10, 20, 4))
        pygame.draw.polygon(surface, (180, 180, 180), [
            (center_x, center_y - 25),
            (center_x - 8, center_y - 15),
            (center_x + 8, center_y - 15)
        ])
        
    elif card_type == 'SeeFuture':
        # Ojo mágico
        pygame.draw.ellipse(surface, (100, 50, 150), (center_x - 18, center_y - 10, 36, 20))
        pygame.draw.circle(surface, (50, 0, 100), (center_x, center_y), 8)
        pygame.draw.circle(surface, (255, 255, 255), (center_x + 2, center_y - 2), 3)
        # Rayos de luz
        for angle in range(0, 360, 60):
            rad = math.radians(angle)
            x1 = center_x + math.cos(rad) * 12
            y1 = center_y + math.sin(rad) * 8
            x2 = center_x + math.cos(rad) * 20
            y2 = center_y + math.sin(rad) * 15
            pygame.draw.line(surface, (150, 100, 200), (int(x1), int(y1)), (int(x2), int(y2)), 2)
    
    elif card_type == 'DrawBottom':
        # Flecha hacia abajo desde caja
        pygame.draw.rect(surface, (50, 150, 150), (center_x - 15, center_y - 20, 30, 20))
        pygame.draw.rect(surface, (0, 0, 0), (center_x - 15, center_y - 20, 30, 20), 2)
        # Flecha
        pygame.draw.line(surface, (0, 150, 150), (center_x, center_y + 2), (center_x, center_y + 18), 3)
        pygame.draw.polygon(surface, (0, 150, 150), [
            (center_x, center_y + 22),
            (center_x - 6, center_y + 14),
            (center_x + 6, center_y + 14)
        ])
    
    elif card_type == 'Shuffle':
        # Símbolo de mezclar (flechas circulares)
        pygame.draw.arc(surface, (200, 100, 50), (center_x - 15, center_y - 15, 30, 30), 0, math.pi, 3)
        pygame.draw.arc(surface, (200, 100, 50), (center_x - 15, center_y - 15, 30, 30), math.pi, 2*math.pi, 3)
        # Flechas
        pygame.draw.polygon(surface, (200, 100, 50), [
            (center_x + 15, center_y - 3),
            (center_x + 12, center_y - 8),
            (center_x + 18, center_y - 8)
        ])
        pygame.draw.polygon(surface, (200, 100, 50), [
            (center_x - 15, center_y + 3),
            (center_x - 12, center_y + 8),
            (center_x - 18, center_y + 8)
        ])
        
    else:  # Safe / Cat
        # Cara de gato simple
        pygame.draw.circle(surface, (255, 200, 150), (center_x, center_y), 18)
        pygame.draw.circle(surface, (0, 0, 0), (center_x - 6, center_y - 4), 3)
        pygame.draw.circle(surface, (0, 0, 0), (center_x + 6, center_y - 4), 3)
        pygame.draw.arc(surface, (0, 0, 0), (center_x - 6, center_y, 12, 8), 0, math.pi, 2)
        # Orejas
        pygame.draw.polygon(surface, (255, 200, 150), [
            (center_x - 15, center_y - 15),
            (center_x - 10, center_y - 22),
            (center_x - 8, center_y - 15)
        ])
        pygame.draw.polygon(surface, (255, 200, 150), [
            (center_x + 15, center_y - 15),
            (center_x + 10, center_y - 22),
            (center_x + 8, center_y - 15)
        ])


def draw_card_visual(surface, x, y, width, height, card_type, show_back=False):
    """Dibuja una carta visual completa."""
    # Fondo de la carta
    if show_back:
        color = CARD_BACK
    else:
        color_map = {
            'Bomb': BOMB_COLOR,
            'Defuse': DEFUSE_COLOR,
            'Skip': SKIP_COLOR,
            'Attack': ATTACK_COLOR,
            'SeeFuture': SEE_FUTURE_COLOR,
            'DrawBottom': DRAW_BOTTOM_COLOR,
            'Shuffle': SHUFFLE_COLOR,
            'Safe': SAFE_COLOR,
        }
        color = color_map.get(card_type, SAFE_COLOR)
    
    pygame.draw.rect(surface, color, (x, y, width, height), border_radius=8)
    pygame.draw.rect(surface, (0, 0, 0), (x, y, width, height), 2, border_radius=8)
    
    if not show_back:
        draw_card_icon(surface, x, y, width, height, card_type)
        
        # Texto del tipo de carta
        font_small = pygame.font.Font(None, 18)
        text = font_small.render(card_type, True, TEXT_COLOR)
        text_rect = text.get_rect(center=(x + width//2, y + height - 15))
        surface.blit(text, text_rect)


def draw_deck(surface, x, y, size):
    """Dibuja el mazo."""
    # Stack effect - multiple cards
    for i in range(3):
        offset = i * 2
        pygame.draw.rect(surface, CARD_BACK, (x + offset, y + offset, 80, 110), border_radius=8)
        pygame.draw.rect(surface, (0, 0, 0), (x + offset, y + offset, 80, 110), 2, border_radius=8)
    
    # Count
    font = pygame.font.Font(None, 28)
    text = font.render(str(size), True, (255, 255, 255))
    text_rect = text.get_rect(center=(x + 40, y + 55))
    surface.blit(text, text_rect)


def draw_hand(surface, x, y, hand_dict, show_counts=True, clickable=False, hover_card=None, include_draw_button=False):
    """Dibuja una mano de cartas con contadores.
    Si clickable=True, retorna dict con {card_type: pygame.Rect} para detectar clicks.
    hover_card: tipo de carta sobre la que está el mouse (para efecto hover).
    include_draw_button: Si True, agrega un botón DRAW al final."""
    card_types = ['Defuse', 'Skip', 'Attack', 'SeeFuture', 'DrawBottom', 'Shuffle', 'Safe']
    card_width = 70
    card_height = 90
    spacing = 10
    
    card_rects = {}
    offset_x = 0
    for card_type in card_types:
        count = hand_dict.get(card_type, 0)
        if count > 0:
            card_rect = pygame.Rect(x + offset_x, y, card_width, card_height)
            
            # Efecto hover: elevar carta ligeramente
            card_y = y
            if hover_card == card_type and clickable:
                card_y -= 10  # Elevar carta
            
            draw_card_visual(surface, x + offset_x, card_y, card_width, card_height, card_type)
            
            # Borde brillante si es clickeable y tiene hover
            if hover_card == card_type and clickable and card_type != 'Defuse':
                pygame.draw.rect(surface, (255, 255, 100), (x + offset_x, card_y, card_width, card_height), 3, border_radius=8)
            
            if show_counts:
                # Badge con contador
                badge_radius = 15
                badge_x = x + offset_x + card_width - badge_radius
                badge_y = y + badge_radius
                pygame.draw.circle(surface, (255, 255, 255), (badge_x, badge_y), badge_radius)
                pygame.draw.circle(surface, (0, 0, 0), (badge_x, badge_y), badge_radius, 2)
                
                font = pygame.font.Font(None, 24)
                text = font.render(f"x{count}", True, (0, 0, 0))
                text_rect = text.get_rect(center=(badge_x, badge_y))
                surface.blit(text, text_rect)
            
            if clickable:
                card_rects[card_type] = card_rect
            
            offset_x += card_width + spacing
    
    # Botón DRAW como una carta más
    if include_draw_button and clickable:
        draw_button_rect = pygame.Rect(x + offset_x, y, card_width, card_height)
        
        # Efecto hover
        draw_y = y
        draw_color = SAFE_COLOR
        if hover_card == 'DRAW':
            draw_y -= 10
            draw_color = (120, 220, 120)
        
        # Dibujar botón como carta
        pygame.draw.rect(surface, draw_color, (x + offset_x, draw_y, card_width, card_height), border_radius=8)
        pygame.draw.rect(surface, (0, 0, 0), (x + offset_x, draw_y, card_width, card_height), 3, border_radius=8)
        
        # Texto DRAW
        font = pygame.font.Font(None, 22)
        text = font.render("DRAW", True, (255, 255, 255))
        text_rect = text.get_rect(center=(x + offset_x + card_width//2, draw_y + card_height//2))
        surface.blit(text, text_rect)
        
        card_rects['DRAW'] = draw_button_rect
    
    return card_rects if clickable else None


def draw_defuse_panel(surface, deck_size):
    """Panel para elegir dónde reinsertar la bomba."""
    panel_width = 420
    panel_height = 300
    panel_x = (WIDTH - panel_width) // 2
    panel_y = (HEIGHT - panel_height) // 2
    
    # Background
    pygame.draw.rect(surface, (255, 255, 255), (panel_x, panel_y, panel_width, panel_height), border_radius=10)
    pygame.draw.rect(surface, BOMB_COLOR, (panel_x, panel_y, panel_width, panel_height), 5, border_radius=10)
    
    # Title
    font_title = pygame.font.Font(None, 36)
    title = font_title.render("DEFUSE! Choose bomb position:", True, TITLE_COLOR)
    title_rect = title.get_rect(center=(WIDTH // 2, panel_y + 40))
    surface.blit(title, title_rect)
    
    # Buttons - centrados dentro del panel
    button_width = 110
    button_height = 50
    button_y = panel_y + 100
    button_spacing = 15
    total_buttons_width = 3 * button_width + 2 * button_spacing
    start_x = panel_x + (panel_width - total_buttons_width) // 2
    buttons = []
    
    positions = ["TOP", "MIDDLE", "BOTTOM"]
    for i, pos in enumerate(positions):
        button_x = start_x + i * (button_width + button_spacing)
        btn_rect = pygame.Rect(button_x, button_y, button_width, button_height)
        buttons.append((btn_rect, pos))
        
        pygame.draw.rect(surface, DEFUSE_COLOR, btn_rect, border_radius=8)
        pygame.draw.rect(surface, (0, 0, 0), btn_rect, 2, border_radius=8)
        
        font = pygame.font.Font(None, 28)
        text = font.render(pos, True, (255, 255, 255))
        text_rect = text.get_rect(center=btn_rect.center)
        surface.blit(text, text_rect)
    
    # Info
    font_info = pygame.font.Font(None, 24)
    info = font_info.render(f"Deck has {deck_size} cards remaining", True, TEXT_COLOR)
    info_rect = info.get_rect(center=(WIDTH // 2, panel_y + 200))
    surface.blit(info, info_rect)
    
    return buttons


def draw_log_panel(surface, game_log, show_log):
    """Panel del game log."""
    if not show_log:
        return
    
    panel_width = 400
    panel_height = HEIGHT - 100
    panel_x = WIDTH - panel_width - 20
    panel_y = 50
    
    # Background
    pygame.draw.rect(surface, (255, 255, 255, 230), (panel_x, panel_y, panel_width, panel_height), border_radius=10)
    pygame.draw.rect(surface, TITLE_COLOR, (panel_x, panel_y, panel_width, panel_height), 3, border_radius=10)
    
    # Title
    font_title = pygame.font.Font(None, 28)
    title = font_title.render("GAME LOG", True, TITLE_COLOR)
    title_rect = title.get_rect(center=(panel_x + panel_width // 2, panel_y + 25))
    surface.blit(title, title_rect)
    
    # Entries
    font_log = pygame.font.Font(None, 18)
    y_offset = panel_y + 60
    visible_entries = game_log.entries[-15:]  # Last 15 entries
    
    for entry in visible_entries:
        # Word wrap for long entries
        words = entry.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + word + " "
            if font_log.size(test_line)[0] < panel_width - 40:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word + " "
        if current_line:
            lines.append(current_line)
        
        for line in lines:
            if y_offset < panel_y + panel_height - 30:
                text = font_log.render(line.strip(), True, TEXT_COLOR)
                surface.blit(text, (panel_x + 20, y_offset))
                y_offset += 20


def draw_see_future_panel(surface, cards):
    """Panel para mostrar las cartas del See the Future.
    Returns: Lista de (rect, card_index) donde card_index es la posición en el deck (0=bottom, 2=top)
    """
    panel_width = 450
    panel_height = 250
    panel_x = (WIDTH - panel_width) // 2
    panel_y = (HEIGHT - panel_height) // 2
    
    # Background
    pygame.draw.rect(surface, (255, 255, 255), (panel_x, panel_y, panel_width, panel_height), border_radius=10)
    pygame.draw.rect(surface, SEE_FUTURE_COLOR, (panel_x, panel_y, panel_width, panel_height), 5, border_radius=10)
    
    # Title
    font_title = pygame.font.Font(None, 32)
    title = font_title.render("Choose a card to draw:", True, SEE_FUTURE_COLOR)
    title_rect = title.get_rect(center=(WIDTH // 2, panel_y + 30))
    surface.blit(title, title_rect)
    
    # Cards (right to left = top to bottom of deck)
    card_width = 90
    card_height = 110
    start_x = panel_x + (panel_width - len(cards) * (card_width + 20)) // 2
    
    card_rects = []
    for i, card in enumerate(reversed(cards)):  # Reversed so rightmost is top
        x = start_x + i * (card_width + 20)
        y = panel_y + 90
        draw_card_visual(surface, x, y, card_width, card_height, card)
        
        # Label (top = #1, bottom = #3)
        font_small = pygame.font.Font(None, 20)
        position = len(cards) - i
        label = font_small.render(f"#{position} (Top)" if position == 1 else f"#{position}", True, TEXT_COLOR)
        label_rect = label.get_rect(center=(x + card_width//2, panel_y + 70))
        surface.blit(label, label_rect)
        
        # Guardar rectángulo para detección de clics
        # card_index: 2=top, 1=middle, 0=bottom (índice en cards list)
        card_rect = pygame.Rect(x, y, card_width, card_height)
        card_rects.append((card_rect, len(cards) - 1 - i))  # Índice original en cards
    
    return card_rects


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Exploding Kittens - V2 Extended")
    clock = pygame.time.Clock()
    
    # Load agent
    q_net = load_agent_model()
    
    # Game state
    env = ExplodingKittensEnv()
    env.reset()
    
    anim = Animation()
    game_log = GameLog()
    game_log.add("=== GAME START ===")
    game_log.add("Human player goes first")
    
    turn = "human"
    done = False
    winner_text = ""
    defuse_mode = False
    defuse_buttons = []
    show_log = False
    waiting_for_animation = False
    
    # Font
    font = pygame.font.Font(None, 32)
    font_small = pygame.font.Font(None, 24)
    font_title = pygame.font.Font(None, 48)
    
    running = True
    card_rects = {}  # Inicializar card_rects fuera del loop
    while running:
        dt = clock.tick(FPS) / 1000.0
        mx, my = pygame.mouse.get_pos()
        
        # Handle animations
        if anim.delay_timer > 0:
            anim.delay_timer -= dt
            if anim.delay_timer <= 0:
                anim.delay_timer = 0
                if waiting_for_animation:
                    waiting_for_animation = False
                    if turn == "human" and not defuse_mode:
                        turn = "agent"
                    elif turn == "agent":
                        turn = "human"
                # Cerrar see_future solo si NO es interactivo (agente)
                if anim.see_future_cards and not anim.see_future_interactive:
                    anim.see_future_cards = None
        
        # Card play animation (Skip/Attack/new cards being played)
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
            anim.card_progress += dt * 2.0
            if anim.card_progress >= 1.0:
                anim.card_progress = 1.0
                anim.drawing_card = False
        
        if anim.explosion:
            anim.explosion_frame += dt * 10
            if anim.explosion_frame >= 15:
                anim.explosion = False
                anim.explosion_frame = 0
                waiting_for_animation = False  # Liberar el bloqueo
                
                # Si el juego terminó por explosión, ahora sí marcar done
                if env.done and not done:
                    done = True
                    if env.winner == 0:
                        winner_text = "AGENT WINS!"
                        game_log.add("=== GAME OVER: AGENT WINS ===")
                    else:
                        winner_text = "HUMAN WINS!"
                        game_log.add("=== GAME OVER: HUMAN WINS ===")
                # Si el agente necesita usar defuse, cambiar a modo defuse
                elif env.phase == "defuse" and env.defuse_pending_for_agent:
                    defuse_mode = True
                    turn = "agent"  # Mantener turno del agente para que elija posición
                # Si el humano necesita defuse
                elif env.phase == "defuse" and not env.defuse_pending_for_agent:
                    defuse_mode = True
                    turn = "human"
        
        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Restart button (when game over)
                if done:
                    restart_btn = pygame.Rect(WIDTH // 2 - 100, HEIGHT // 2 + 50, 200, 50)
                    if restart_btn.collidepoint(mx, my):
                        env.reset()
                        anim = Animation()
                        game_log = GameLog()
                        game_log.add("=== GAME RESTARTED ===")
                        done = False
                        turn = "agent"
                        winner_text = ""
                        continue
                    
                    # View Log button (en pantalla de game over)
                    view_log_btn = pygame.Rect(WIDTH // 2 - 100, HEIGHT // 2 + 120, 200, 50)
                    if view_log_btn.collidepoint(mx, my):
                        show_log = not show_log
                        continue
                        game_log.add("Human player goes first")
                        turn = "human"
                        done = False
                        winner_text = ""
                        defuse_mode = False
                        waiting_for_animation = False
                    continue
                
                # Log toggle button
                log_btn = pygame.Rect(WIDTH - 120, 10, 100, 30)
                if log_btn.collidepoint(mx, my):
                    show_log = not show_log
                    continue
                
                # Defuse position selection
                if defuse_mode:
                    for btn_rect, pos in defuse_buttons:
                        if btn_rect.collidepoint(mx, my):
                            # Human defuses
                            pos_map = {"TOP": "top", "MIDDLE": "middle", "BOTTOM": "bottom"}
                            env._reinsert_bomb_for_agent(pos_map[pos])
                            game_log.add(f"[HUMAN] Defused bomb to {pos}")
                            
                            defuse_mode = False
                            env.phase = "action"
                            
                            # Human turn is complete after defusing, agent goes next
                            if not env.done:
                                env.current_player = 0
                                env.turn_count += 1
                                waiting_for_animation = True
                                anim.delay_timer = 0.8  # Short delay before agent turn
                    continue
                
                # See Future card selection
                if anim.see_future_cards and see_future_rects and turn == "human":
                    for card_rect, card_index in see_future_rects:
                        if card_rect.collidepoint(mx, my):
                            # El jugador eligió una carta del deck visible
                            chosen_card = anim.see_future_cards[card_index]
                            game_log.add(f"[HUMAN] Chose to draw: {chosen_card}")
                            
                            # Remover la carta elegida del deck
                            # card_index: 0=bottom, 1=middle, 2=top
                            # deck[-3:] = [bottom, middle, top]
                            # Necesitamos remover de la posición correcta
                            deck_position = len(env.deck) - 3 + card_index
                            card = env.deck.pop(deck_position)
                            env.last_drawn[1] = card
                            
                            # Procesar la carta
                            if card == "Bomb":
                                anim.explosion = True
                                anim.explosion_frame = 0
                                waiting_for_animation = True
                                
                                if env.hands[1]["Defuse"] > 0:
                                    env.hands[1]["Defuse"] -= 1
                                    game_log.add("[HUMAN] Hit a BOMB! Used Defuse!")
                                    defuse_mode = True
                                else:
                                    env.done = True
                                    env.winner = 0
                                    winner_text = "AGENT WINS!"
                                    done = True
                                    game_log.add("=== GAME OVER: AGENT WINS ===")
                            elif card == 'Defuse':
                                env.hands[1]['Defuse'] += 1
                            elif card == 'Skip':
                                env.hands[1]['Skip'] += 1
                            elif card == 'Attack':
                                env.hands[1]['Attack'] += 1
                            elif card == 'SeeFuture':
                                env.hands[1]['SeeFuture'] += 1
                            elif card == 'DrawBottom':
                                env.hands[1]['DrawBottom'] += 1
                            elif card == 'Shuffle':
                                env.hands[1]['Shuffle'] += 1
                            else:
                                env.hands[1]['Safe'] += 1
                            
                            # Cerrar panel de See Future
                            anim.see_future_cards = None
                            anim.see_future_interactive = False
                            
                            # Reducir pending draws
                            env.pending_draws[1] = max(1, env.pending_draws[1] - 1)
                            
                            # Si ya no tiene más draws, cambiar turno
                            if env.pending_draws[1] == 1 and not defuse_mode:
                                env.current_player = 0
                                turn = "agent"
                            
                            continue
                            if not env.done:
                                env.current_player = 0
                                env.turn_count += 1
                                waiting_for_animation = True
                                anim.delay_timer = 0.8  # Short delay before agent turn
                            continue
                
                if not done and not waiting_for_animation and turn == "human":
                    # Mapeo de tipo de carta a acción
                    card_to_action = {
                        'Skip': 1,
                        'Attack': 2,
                        'SeeFuture': 6,
                        'DrawBottom': 7,
                        'Shuffle': 8,
                    }
                    
                    human_action = None
                    
                    # Check if clicked on any card in hand (ahora incluye DRAW)
                    if card_rects:
                        for card_type, rect in card_rects.items():
                            if rect.collidepoint(mx, my):
                                if card_type == 'DRAW':
                                    human_action = 0
                                else:
                                    action = card_to_action.get(card_type)
                                    if action is not None:
                                        human_action = action
                                break
                    
                    if human_action is not None:
                        # Set current player to human
                        env.current_player = 1
                        
                        # Apply human action
                        draws = env.pending_draws[1]
                        env.last_opp_action = 0
                        
                        parts = []
                        
                        if human_action == 1 and env.hands[1]["Skip"] > 0:
                            env.hands[1]["Skip"] -= 1
                            draws = max(0, draws - 1)
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
                        elif human_action == 6 and env.hands[1]["SeeFuture"] > 0:
                            env.hands[1]["SeeFuture"] -= 1
                            msg_action = "[HUMAN] Played SEE FUTURE - Choose a card to draw"
                            parts.append(msg_action)
                            game_log.add(msg_action)
                            # Show top 3 cards - el jugador debe hacer clic en una
                            if len(env.deck) >= 3:
                                anim.see_future_cards = env.deck[-3:]
                                anim.see_future_interactive = True  # Interactivo, debe elegir
                                # NO cambiar turno ni usar delay_timer
                                # El panel permanece abierto hasta que el jugador haga clic
                        elif human_action == 7 and env.hands[1]["DrawBottom"] > 0:
                            env.hands[1]["DrawBottom"] -= 1
                            msg_action = "[HUMAN] Played DRAW FROM BOTTOM"
                            parts.append(msg_action)
                            game_log.add(msg_action)
                            # Draw from bottom instead
                            if len(env.deck) > 0 and draws > 0:
                                card = env.deck.pop(0)  # Bottom
                                env.last_drawn[1] = card
                                env._process_drawn_card(1, card, is_agent=False)
                                draws -= 1
                                game_log.add(f"[HUMAN] Drew from BOTTOM: {card}")
                        elif human_action == 8 and env.hands[1]["Shuffle"] > 0:
                            env.hands[1]["Shuffle"] -= 1
                            random.shuffle(env.deck)
                            msg_action = "[HUMAN] Played SHUFFLE - Deck shuffled!"
                            parts.append(msg_action)
                            game_log.add(msg_action)
                        else:
                            if human_action in [1, 2, 6, 7, 8]:
                                msg_error = "[HUMAN] Card not available"
                                parts.append(msg_error)
                                game_log.add(msg_error)
                        
                        # Draw remaining cards (from top)
                        # IMPORTANTE: 
                        # - SeeFuture (6) y Shuffle (8) NO consumen draws, pero tampoco terminan el turno
                        # - El jugador DEBE draw después de usarlas
                        # - Skip reduce draws, Attack pasa draws al oponente
                        should_auto_draw = (human_action == 0) or (draws > 0 and human_action not in [1, 2])
                        
                        # Si jugó Skip y aún tiene draws, NO forzar draw automático
                        if human_action == 1 and draws > 0:
                            should_auto_draw = False
                            msg = f"You have {draws} draw(s) left. Play another card or draw."
                        
                        # Si jugó Attack, no hay draws para este jugador (draws = 0)
                        # Si jugó SeeFuture o Shuffle, DEBE draw después
                        if should_auto_draw and draws > 0:
                            waiting_for_animation = True
                            anim.delay_timer = 1.0 * draws
                            
                            cards_to_draw = draws  # Guardar cuántas cartas debe dibujar
                            for i in range(cards_to_draw):
                                if env.done:
                                    break
                                
                                # Start animation
                                anim.drawing_card = True
                                anim.card_pos = [WIDTH // 2 - 50, 250]
                                anim.card_target = [WIDTH - 200, 550]
                                anim.card_progress = 0
                                
                                # Draw card manually for human
                                if len(env.deck) > 0:
                                    card = env.deck.pop()
                                    env.last_drawn[1] = card
                                    draws -= 1  # Reducir contador local
                                    env.pending_draws[1] = max(1, draws)  # Actualizar env (mínimo 1 para próximo turno)
                                    
                                    anim.card_type = card
                                    msg_draw = f"[HUMAN] Drew: {card}"
                                    parts.append(msg_draw)
                                    game_log.add(msg_draw)
                                    
                                    if card == "Bomb":
                                        anim.explosion = True
                                        anim.explosion_frame = 0
                                        
                                        if env.hands[1]["Defuse"] > 0:
                                            env.hands[1]["Defuse"] -= 1
                                            msg_defuse = "[HUMAN] Used Defuse!"
                                            game_log.add(msg_defuse)
                                            defuse_mode = True
                                            waiting_for_animation = False
                                            break
                                        else:
                                            env.done = True
                                            env.winner = 0
                                            winner_text = "AGENT WINS!"
                                            done = True
                                            game_log.add("=== GAME OVER: AGENT WINS ===")
                                            break
                                    elif card == 'Defuse':
                                        env.hands[1]['Defuse'] += 1
                                    elif card == 'Skip':
                                        env.hands[1]['Skip'] += 1
                                    elif card == 'Attack':
                                        env.hands[1]['Attack'] += 1
                                    elif card == 'SeeFuture':
                                        env.hands[1]['SeeFuture'] += 1
                                    elif card == 'DrawBottom':
                                        env.hands[1]['DrawBottom'] += 1
                                    elif card == 'Shuffle':
                                        env.hands[1]['Shuffle'] += 1
                                    else:
                                        env.hands[1]['Safe'] += 1
                        
                        # Reset pending_draws ONLY cuando se completan todos los draws
                        # Si should_auto_draw fue ejecutado, ya se actualizó en el loop
                        # Si no dibujó (ej: jugó Skip/Attack), actualizar manualmente
                        if not should_auto_draw:
                            env.pending_draws[1] = draws  # Mantener los draws restantes
                            if draws == 0:
                                env.pending_draws[1] = 1  # Reset para próximo turno
                        
                        # Cambiar de turno SOLO si:
                        # 1. No hay draws pendientes (draws == 0)
                        # 2. Y ya dibujó todas las cartas requeridas (should_auto_draw ejecutado)
                        # NOTA: SeeFuture y Shuffle NO cambian turno hasta que se draw
                        if (draws == 0 or (should_auto_draw and not defuse_mode)):
                            if not waiting_for_animation and not defuse_mode:
                                env.current_player = 0
                                turn = "agent"
                        # Si aún tiene draws pendientes después de Skip, mantener turno humano
                        elif human_action == 1 and draws > 0:
                            turn = "human"  # Mantener turno para que pueda jugar otra carta
                        # Si jugó SeeFuture o Shuffle, mantener turno hasta que draw
                        elif human_action in [6, 8]:
                            turn = "human"  # Debe draw antes de pasar turno
        
        # Agent turn (automatic)
        if turn == "agent" and not done and not waiting_for_animation:
            agent_msg, agent_done = agent_turn(env, q_net, anim, game_log)
            
            # Si hay una animación de explosión activa, SIEMPRE esperar
            if anim.explosion:
                waiting_for_animation = True
            elif agent_done or env.done:
                # Solo marcar done si no hay animación pendiente
                done = True
                if env.winner == 0:
                    winner_text = "AGENT WINS!"
                    game_log.add("=== GAME OVER: AGENT WINS ===")
                else:
                    winner_text = "HUMAN WINS!"
                    game_log.add("=== GAME OVER: HUMAN WINS ===")
            else:
                waiting_for_animation = True
                anim.delay_timer = 1.5
        
        # Drawing
        screen.fill(BG_COLOR)
        
        # Title
        title = font_title.render("EXPLODING KITTENS - V2", True, TITLE_COLOR)
        title_rect = title.get_rect(center=(WIDTH // 2, 30))
        screen.blit(title, title_rect)
        
        # Deck
        draw_deck(screen, WIDTH // 2 - 50, 200, len(env.deck))
        deck_label = font_small.render(f"Deck: {len(env.deck)} cards", True, TEXT_COLOR)
        screen.blit(deck_label, (WIDTH // 2 - 50, 320))
        
        # Agent hand (top)
        agent_hand_y = 50
        draw_hand(screen, 50, agent_hand_y, env.hands[0], show_counts=True, clickable=False, hover_card=None)
        agent_label = font.render(f"AGENT - Pending draws: {env.pending_draws[0]}", True, TEXT_COLOR)
        screen.blit(agent_label, (50, agent_hand_y + 110))
        
        # Detect hover on human cards (incluir DRAW button)
        hover_card = None
        if turn == "human" and not done and not waiting_for_animation:
            for card_type in ['Skip', 'Attack', 'SeeFuture', 'DrawBottom', 'Shuffle', 'DRAW']:
                if card_rects and card_type in card_rects:
                    if card_rects[card_type].collidepoint(mx, my):
                        hover_card = card_type
                        break
        
        # Human hand (bottom) - CLICKEABLE con hover e incluye botón DRAW
        human_hand_y = HEIGHT - 180
        card_rects = draw_hand(screen, 50, human_hand_y, env.hands[1], show_counts=True, clickable=True, hover_card=hover_card, include_draw_button=True)
        
        # Pending draws text DEBAJO de las cartas
        human_label = font.render(f"YOU - Pending draws: {env.pending_draws[1]}", True, TEXT_COLOR)
        screen.blit(human_label, (50, human_hand_y + 110))
        
        # Turn indicator
        if not done:
            turn_text = f"Current Turn: {turn.upper()}"
            turn_surface = font.render(turn_text, True, TITLE_COLOR)
            screen.blit(turn_surface, (WIDTH // 2 - turn_surface.get_width() // 2, HEIGHT // 2 - 40))
        
        # Defuse panel (solo mostrar si NO hay explosión animándose)
        if defuse_mode and not anim.explosion:
            defuse_buttons = draw_defuse_panel(screen, len(env.deck))
        elif defuse_mode and anim.explosion:
            # Mostrar mensaje temporal mientras explota
            temp_font = pygame.font.Font(None, 48)
            temp_text = temp_font.render("💥 BOOM! 💥", True, (255, 200, 0))
            temp_rect = temp_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 150))
            screen.blit(temp_text, temp_rect)
        
        # See Future panel
        see_future_rects = []
        if anim.see_future_cards:
            see_future_rects = draw_see_future_panel(screen, anim.see_future_cards)
        
        # Explosion animation (ANTES del game over screen para que se vea)
        if anim.explosion:
            frame = int(anim.explosion_frame)
            radius = frame * 10
            alpha = max(0, 255 - frame * 17)
            
            explosion_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(explosion_surface, (255, 100, 0, alpha), (radius, radius), radius)
            screen.blit(explosion_surface, (WIDTH // 2 - radius, HEIGHT // 2 - radius))
            
            if frame < 8:
                boom_font = pygame.font.Font(None, 72)
                boom_text = boom_font.render("BOOM!", True, (255, 255, 0))
                boom_rect = boom_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
                screen.blit(boom_text, boom_rect)
        
        # Game over screen (DESPUÉS de la explosión)
        if done:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            screen.blit(overlay, (0, 0))
            
            winner_font = pygame.font.Font(None, 72)
            winner_surface = winner_font.render(winner_text, True, (255, 255, 100))
            winner_rect = winner_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 50))
            screen.blit(winner_surface, winner_rect)
            
            # Restart button
            restart_btn = pygame.Rect(WIDTH // 2 - 100, HEIGHT // 2 + 50, 200, 50)
            btn_color = (100, 255, 100) if restart_btn.collidepoint(mx, my) else (50, 200, 50)
            pygame.draw.rect(screen, btn_color, restart_btn, border_radius=10)
            pygame.draw.rect(screen, (255, 255, 255), restart_btn, 3, border_radius=10)
            
            restart_text = font.render("RESTART", True, (255, 255, 255))
            restart_rect = restart_text.get_rect(center=restart_btn.center)
            screen.blit(restart_text, restart_rect)
            
            # View Log button (en pantalla de game over)
            view_log_btn = pygame.Rect(WIDTH // 2 - 100, HEIGHT // 2 + 120, 200, 50)
            log_btn_color = (100, 200, 255) if view_log_btn.collidepoint(mx, my) else (50, 150, 200)
            pygame.draw.rect(screen, log_btn_color, view_log_btn, border_radius=10)
            pygame.draw.rect(screen, (255, 255, 255), view_log_btn, 3, border_radius=10)
            
            view_log_text = font.render("VIEW LOG", True, (255, 255, 255))
            view_log_rect = view_log_text.get_rect(center=view_log_btn.center)
            screen.blit(view_log_text, view_log_rect)
        
        # Log toggle button (siempre visible)
        log_btn = pygame.Rect(WIDTH - 120, 10, 100, 30)
        log_color = TITLE_COLOR if log_btn.collidepoint(mx, my) else ATTACK_COLOR
        pygame.draw.rect(screen, log_color, log_btn, border_radius=5)
        log_text = font_small.render("LOG", True, (255, 255, 255))
        log_rect = log_text.get_rect(center=log_btn.center)
        screen.blit(log_text, log_rect)
        
        # Log panel
        draw_log_panel(screen, game_log, show_log)
        
        # Card play animation (Skip/Attack/new cards)
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
            
            draw_card_visual(screen, x_centered, y_centered, width, height, anim.play_card_type)
            
            # Texto flotante indicando la acción
            font = pygame.font.Font(None, int(32 + 8 * anim.play_card_progress))
            action_text = f"{anim.play_card_from.upper()} plays {anim.play_card_type}!"
            
            # Color based on card type
            if anim.play_card_type == 'Attack':
                text_color = ATTACK_COLOR
            elif anim.play_card_type == 'Skip':
                text_color = SKIP_COLOR
            elif anim.play_card_type == 'SeeFuture':
                text_color = SEE_FUTURE_COLOR
            elif anim.play_card_type == 'DrawBottom':
                text_color = DRAW_BOTTOM_COLOR
            elif anim.play_card_type == 'Shuffle':
                text_color = SHUFFLE_COLOR
            else:
                text_color = TEXT_COLOR
            
            text_surf = font.render(action_text, True, text_color)
            text_rect = text_surf.get_rect(center=(x + 40, y - 40))
            
            # Sombra del texto
            shadow_surf = font.render(action_text, True, (50, 50, 50))
            shadow_rect = shadow_surf.get_rect(center=(x + 42, y - 38))
            screen.blit(shadow_surf, shadow_rect)
            screen.blit(text_surf, text_rect)
        
        # Card draw animation
        if anim.drawing_card and anim.card_progress < 1.0:
            t = anim.card_progress
            # Ease-in-out
            t = t * t * (3.0 - 2.0 * t)
            x = anim.card_pos[0] + (anim.card_target[0] - anim.card_pos[0]) * t
            y = anim.card_pos[1] + (anim.card_target[1] - anim.card_pos[1]) * t
            if anim.card_type:
                draw_card_visual(screen, int(x), int(y), 80, 110, anim.card_type)
        
        pygame.display.flip()
    
    pygame.quit()


if __name__ == "__main__":
    main()
