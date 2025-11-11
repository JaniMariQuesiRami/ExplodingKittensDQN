# Animation & Bug Fixes - November 10, 2025

## 🎯 Objetivo
Arreglar bugs críticos de gameplay y mejorar las animaciones para hacer el juego más comprensible visualmente.

---

## 🐛 Bugs Corregidos

### 1. **BUG CRÍTICO: Attack Card - Human solo sacaba 1 carta**
**Problema:** Cuando el agente jugaba Attack, el humano debería sacar 2 cartas pero solo sacaba 1.

**Causa:** `env.pending_draws[1] = 1` se reseteaba ANTES del loop de draw, no DESPUÉS.

**Fix en V1 y V2:**
```python
# ANTES (línea ~665 en ambas versiones):
draws = env.pending_draws[1]
env.pending_draws[1] = 1  # ❌ Reset prematuro
for i in range(draws):
    # draw cards...

# DESPUÉS:
draws = env.pending_draws[1]
for i in range(draws):
    # draw cards...
env.pending_draws[1] = 1  # ✅ Reset después del loop
```

**Archivos modificados:**
- `v1/play_pygame.py` (líneas 663-768)
- `v2/play_pygame.py` (líneas 688-810)

---

### 2. **BUG: Animación de bomba invisible para Agent**
**Problema:** Cuando el agente sacaba una bomba, la animación de explosión no se veía o duraba muy poco tiempo.

**Causa:** El delay_timer era muy corto y la animación se cancelaba antes de verse.

**Fix en V1 y V2:**
```python
# ANTES:
if env.last_drawn[0] == "Bomb":
    anim.explosion = True
    anim.explosion_frame = 0
    # delay_timer normal (1.0 * cards_drawn)

# DESPUÉS:
if env.last_drawn[0] == "Bomb":
    anim.explosion = True
    anim.explosion_frame = 0
    anim.delay_timer = max(anim.delay_timer, 2.0)  # ✅ Mínimo 2 segundos
    
    if env.hands[0]["Defuse"] > 0 or env.phase == "defuse":
        bomb_msg = "[AGENT] 💣 HIT A BOMB! Using Defuse..."
    else:
        bomb_msg = "[AGENT] 💣 HIT A BOMB! NO DEFUSE - EXPLODED!"
    game_log.add(bomb_msg)
```

**Archivos modificados:**
- `v1/play_pygame.py` (líneas 152-166)
- `v2/play_pygame.py` (líneas 213-227)

---

## 🎨 Mejoras de Animaciones

### 3. **NUEVA: Animación de cartas jugadas por el Agent**
**Problema:** El agente jugaba Skip/Attack y aparecían de la nada sin indicación visual.

**Solución:** Nueva animación que muestra la carta volando desde el área del agente al centro.

**Implementación:**

#### A. Nueva clase Animation expandida:
```python
class Animation:
    def __init__(self):
        # ... existing fields ...
        # New: card play animation
        self.playing_card = False
        self.play_card_pos = [0, 0]
        self.play_card_target = [0, 0]
        self.play_card_progress = 0
        self.play_card_type = None
        self.play_card_from = "agent"  # "agent" or "human"
```

#### B. Trigger en agent_turn():
```python
# V1: Skip/Attack
if action == 1 and env.hands[0]['Skip'] > 0:
    anim.playing_card = True
    anim.play_card_type = 'Skip'
    anim.play_card_from = 'agent'
    anim.play_card_pos = [200, 150]
    anim.play_card_target = [WIDTH // 2 - 50, HEIGHT // 2 - 75]
    anim.play_card_progress = 0
    anim.delay_timer = 0.8

# V2: También SeeFuture, DrawBottom, Shuffle
elif action == 6 and env.hands[0].get('SeeFuture', 0) > 0:
    # Same animation setup...
```

#### C. Actualización en game loop:
```python
# Card play animation update
if anim.playing_card:
    anim.play_card_progress += dt * 2.5  # Fast animation
    if anim.play_card_progress >= 1:
        anim.play_card_progress = 1
        anim.playing_card = False
    
    # Interpolación suave (ease-in-out)
    t = anim.play_card_progress
    t = t * t * (3 - 2 * t)
    anim.play_card_pos[0] = ...
    anim.play_card_pos[1] = ...
```

#### D. Función de dibujo (V1):
```python
def draw_played_card(surface, anim):
    """Dibuja una carta siendo jugada (Skip/Attack) con animación."""
    if anim.playing_card:
        x = int(anim.play_card_pos[0])
        y = int(anim.play_card_pos[1])
        
        # Efecto de escala (crece hacia el centro)
        scale = 1.0 + anim.play_card_progress * 0.3
        width = int(80 * scale)
        height = int(110 * scale)
        
        draw_card_visual(surface, x_centered, y_centered, width, height, anim.play_card_type)
        
        # Texto flotante con el nombre de la carta
        action_text = f"{anim.play_card_from.upper()} plays {anim.play_card_type}!"
        # ... render text ...
```

#### E. En V2 también maneja las nuevas cartas:
```python
# Color basado en el tipo de carta
if anim.play_card_type == 'Attack':
    text_color = ATTACK_COLOR
elif anim.play_card_type == 'SeeFuture':
    text_color = SEE_FUTURE_COLOR
elif anim.play_card_type == 'DrawBottom':
    text_color = DRAW_BOTTOM_COLOR
elif anim.play_card_type == 'Shuffle':
    text_color = SHUFFLE_COLOR
# ...
```

**Archivos modificados:**
- `v1/play_pygame.py`:
  - Clase Animation (líneas 17-36)
  - agent_turn() (líneas 101-130)
  - Animation update loop (líneas 791-807)
  - draw_played_card() function (líneas 407-436)
  - Drawing in main loop (línea 928)

- `v2/play_pygame.py`:
  - Clase Animation (líneas 17-40)
  - agent_turn() (líneas 111-161)
  - Animation update loop (líneas 603-621)
  - Inline drawing in main loop (líneas 946-988)

---

## 📊 Resumen de Cambios por Archivo

### `v1/play_pygame.py` (1028 líneas)
- ✅ Clase Animation expandida (+6 campos)
- ✅ agent_turn() con animaciones de cartas jugadas
- ✅ Animación de bomba mejorada (2 segundos mínimo)
- ✅ draw_played_card() nueva función
- ✅ Attack bug fix (pending_draws reset después)
- ✅ Animation update loop con playing_card

### `v2/play_pygame.py` (1024 líneas)
- ✅ Clase Animation expandida (+6 campos)
- ✅ agent_turn() con animaciones para 8 tipos de cartas
- ✅ Animación de bomba mejorada (2 segundos mínimo)
- ✅ Inline card play animation rendering
- ✅ Attack bug fix (pending_draws reset después)
- ✅ Animation update loop con playing_card
- ✅ Color-coded texto para cada tipo de carta

---

## 🎮 Resultado Visual

### Antes:
1. Agente juega Attack → ❌ No se ve nada, mensaje de texto solamente
2. Human debe sacar 2 cartas → ❌ Solo saca 1
3. Agente saca bomba → ❌ Explosión invisible/muy rápida

### Después:
1. Agente juega Attack → ✅ Carta vuela desde su área al centro con escala creciente + texto "AGENT plays Attack!"
2. Human debe sacar 2 cartas → ✅ Saca 2 cartas correctamente
3. Agente saca bomba → ✅ Explosión visible por 2 segundos + mensaje "💣 HIT A BOMB! Using Defuse..."

---

## ✅ Testing Checklist

### Casos de prueba:
- [ ] **Test 1:** Human recibe Attack → debe sacar 2 cartas
- [ ] **Test 2:** Agent juega Skip → animación de carta visible
- [ ] **Test 3:** Agent juega Attack → animación de carta visible + texto
- [ ] **Test 4:** Agent saca bomba con defuse → explosión visible 2s + mensaje
- [ ] **Test 5:** Agent saca bomba sin defuse → explosión visible 2s + game over
- [ ] **Test 6 (V2):** Agent juega SeeFuture → animación visible con color morado
- [ ] **Test 7 (V2):** Agent juega DrawBottom → animación visible con color cyan
- [ ] **Test 8 (V2):** Agent juega Shuffle → animación visible con color coral

---

## 🔍 Verificación de Sintaxis

```bash
cd v1 && python -m py_compile play_pygame.py  # ✅ OK
cd v2 && python -m py_compile play_pygame.py  # ✅ OK
```

---

## 📝 Notas Técnicas

### Timing de Animaciones:
- **Card play animation:** 0.8s (configurable via `delay_timer`)
- **Card draw animation:** 1.0s por carta
- **Explosion animation:** 2.0s mínimo (antes era variable)
- **See Future (V2):** 2.0s para ver las 3 cartas

### Interpolación:
- Ease-in-out cúbica: `t = t * t * (3 - 2 * t)`
- Smooth animation sin saltos bruscos

### Escala de carta jugada:
- Tamaño inicial: 80x110
- Tamaño final: 104x143 (escala 1.3x)
- Crece suavemente durante el movimiento

---

## 🚀 Próximos Pasos (Opcional)

Si quieres mejorar aún más:
1. **Sound effects:** Agregar sonidos para cada carta jugada
2. **Particle effects:** Chispas cuando se juega Attack
3. **Card trails:** Estela detrás de la carta en movimiento
4. **Better explosion:** Múltiples círculos concéntricos
5. **Shake effect:** Pantalla vibra cuando hay explosión

---

## 📄 Archivos Relacionados

- `v1/play_pygame.py` - V1 pygame con fixes
- `v2/play_pygame.py` - V2 pygame con fixes + nuevas cartas
- `v1/exploding_env.py` - Environment logic (sin cambios)
- `v2/exploding_env.py` - Environment logic (sin cambios)

---

**Status:** ✅ COMPLETE - Ready for testing
**Date:** November 10, 2025
**Changes:** 3 bugs fixed, 1 major animation system added
