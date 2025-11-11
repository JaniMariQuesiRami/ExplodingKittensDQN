# 🧠 ¿Qué Sabe la Política del DQN?

## 📋 Resumen Ejecutivo

La política (red neuronal DQN) **NO sabe directamente** qué cartas tiene en la mano. Solo recibe **11 números** como entrada (el "estado"), y debe inferir qué hacer basándose en esos números.

---

## 🔍 **Observación del Estado (11 valores)**

Cuando el agente toma una decisión, la red neuronal recibe estos **11 números**:

```python
# Archivo: exploding_env.py, líneas 187-217
def _get_obs(self):
    deck_size = len(self.deck)
    bombs = self._count_bombs_in_deck()
    h0 = self.hands[0]  # Mano del agente
    h1 = self.hands[1]  # Mano del oponente
    
    bomb_prob = (bombs / deck_size) if deck_size > 0 else 0.0
    opp_cards = h1['Defuse'] + h1['Skip'] + h1['Attack'] + h1['Safe']
    
    obs = np.array([
        deck_size_norm,           # 0. Tamaño del deck (normalizado)
        bombs_norm,               # 1. Número de bombas (normalizado)
        bomb_prob,                # 2. Probabilidad de bomba (%)
        float(h0['Defuse']),      # 3. Cuántos Defuse tengo
        float(h0['Skip']),        # 4. Cuántos Skip tengo
        float(h0['Attack']),      # 5. Cuántos Attack tengo
        float(self.pending_draws[0]),  # 6. Cuántas cartas debo robar
        opp_cards_norm,           # 7. Total de cartas del oponente (normalizado)
        last_opp_skip,            # 8. ¿Oponente usó Skip? (1.0 o 0.0)
        last_opp_attack,          # 9. ¿Oponente usó Attack? (1.0 o 0.0)
        phase_defuse,             # 10. ¿Estoy en fase defuse? (1.0 o 0.0)
    ], dtype=np.float32)
```

---

## ❌ **Lo Que NO Está en la Observación**

| Información | ¿Está en el estado? | Razón |
|-------------|---------------------|-------|
| **SeeFuture** | ❌ NO | No se incluye `h0['SeeFuture']` |
| **DrawBottom** | ❌ NO | No se incluye `h0['DrawBottom']` |
| **Shuffle** | ❌ NO | No se incluye `h0['Shuffle']` |
| **Safe cards** | ❌ NO | No se incluye `h0['Safe']` |
| Top 3 cartas del deck | ❌ NO | SeeFuture lo muestra al humano, pero no está en el estado |

---

## 🎯 **Por Qué la Política Puede Elegir Acciones Inválidas**

### **Ejemplo: Elegir SeeFuture sin tener la carta**

1. **Entrada de la red neuronal**:
   ```python
   state = [0.67, 0.15, 0.22, 1.0, 1.0, 0.0, 1.0, 0.20, 0.0, 0.0, 0.0]
   #         ^^^ deck  ^^^ bombs ^^^ Defuse Skip Attack pending_draws...
   # Nota: No dice cuántos SeeFuture tiene
   ```

2. **Salida de la red neuronal** (Q-values para 9 acciones):
   ```python
   Q-values = [0.5, 0.3, 0.4, 0.1, 0.1, 0.1, 0.9, 0.6, 0.2]
   #           ^^^ Draw      ^^^ Skip       ^^^ SeeFuture (más alto!)
   ```

3. **Decisión**:
   - La red dice: "Mejor acción = 6 (SeeFuture)" porque tiene Q-value = 0.9
   - **Pero el agente no tiene SeeFuture en mano**

4. **Ejecución en el entorno**:
   ```python
   # Archivo: exploding_env.py, líneas 295-300
   elif action == 6 and self.hands[0]['SeeFuture'] > 0:
       self.hands[0]['SeeFuture'] -= 1
       # ... ejecuta SeeFuture
   ```
   - **Condición falla**: `self.hands[0]['SeeFuture']` es 0
   - **No se ejecuta nada**, pero la acción **SÍ se registra** en el CSV

---

## 🔄 **Flujo Completo: Intento vs Ejecución**

### **Turno típico del agente:**

```
1. Red neuronal recibe estado (11 números)
   └─> [deck=0.67, bombs=0.15, ... Defuse=1, Skip=0, Attack=1, ...]
   
2. Red neuronal calcula Q-values para 9 acciones
   └─> [Q0=0.5, Q1=0.3, ... Q6=0.9 (SeeFuture), ...]
   
3. Política elige acción con mayor Q-value
   └─> action = 6 (SeeFuture)
   
4. Se registra en CSV: "Intentó acción 6"
   └─> actions_attempted_sequence += "6"
   
5. Entorno verifica si puede ejecutar acción 6
   └─> if self.hands[0]['SeeFuture'] > 0:  # FALSE!
   
6. NO se ejecuta, NO se consume carta
   └─> actions_executed_sequence NO incluye "6"
   
7. Entorno continúa al siguiente paso (robar cartas obligatorias)
```

---

## 📊 **Ejemplo Real de Tu CSV**

### **Juego #1** (del CSV anterior):
```
game_id: 1
turns: 3
actions_attempted_sequence: "0,6,7"
actions_executed_sequence: "6,7"  ← Nota: falta el 0
```

**¿Qué pasó?**

1. **Turno 1**: Red elige `0` (Draw)
   - Intento registrado: ✅
   - ¿Ejecutado? Depende de si había pending_draws > 0
   - Si pending_draws ya estaba en 1, el Draw es obligatorio al final del turno
   - Pero la acción `0` explícita puede no consumir nada

2. **Turno 2**: Red elige `6` (SeeFuture)
   - Intento registrado: ✅
   - ¿Ejecutado? ✅ (el agente SÍ tenía SeeFuture)
   - Se consumió la carta

3. **Turno 3**: Red elige `7` (DrawBottom)
   - Intento registrado: ✅
   - ¿Ejecutado? ✅ (el agente SÍ tenía DrawBottom)
   - Se robó del fondo del deck

---

## 💡 **¿Por Qué el Sistema Funciona Así?**

### **Ventajas:**
1. **Simplicidad del espacio de acción**: La red siempre tiene 9 opciones, no necesita saber cuántas cartas tiene
2. **Aprendizaje por prueba y error**: La red aprende que elegir acción 6 sin tener la carta **no sirve para nada** (no cambia el estado)
3. **Generalización**: La red debe aprender a inferir qué cartas tiene basándose en el historial

### **Desventajas:**
1. **Acciones desperdiciadas**: La red pierde turnos eligiendo acciones inválidas
2. **Espacio de búsqueda más grande**: La red debe aprender 9 acciones × múltiples estados
3. **CSV confuso**: Las secuencias "intentadas" incluyen acciones que no pasaron nada

---

## 🔧 **Solución Implementada: Filtrado de Acciones Válidas**

Hay una función `valid_actions_from_state()` que **limita** las opciones:

```python
# Archivo: exploding_env.py, líneas 454-460
def valid_actions_from_state(state):
    """Devuelve las acciones válidas dado el estado (fase normal o defuse)."""
    phase_defuse = state[10] > 0.5
    if phase_defuse:
        return [3, 4, 5]  # Solo posiciones de Defuse
    else:
        return [0, 1, 2, 6, 7, 8]  # Draw, Skip, Attack, SeeFuture, DrawBottom, Shuffle
```

**Pero esto NO verifica si el agente tiene las cartas en mano**, solo filtra por fase.

---

## 🎯 **Respuesta Directa a Tu Pregunta**

> **¿Por qué la política puede elegir esa carta si no la tiene?**

Porque la red neuronal **NO recibe información sobre SeeFuture, DrawBottom, Shuffle, ni Safe** en el vector de estado.

La red solo sabe:
- ✅ Cuántos Defuse tiene (posición 3 del estado)
- ✅ Cuántos Skip tiene (posición 4)
- ✅ Cuántos Attack tiene (posición 5)
- ❌ **NO sabe** cuántos SeeFuture tiene
- ❌ **NO sabe** cuántos DrawBottom tiene
- ❌ **NO sabe** cuántos Shuffle tiene

Por eso puede **intentar** usar SeeFuture incluso sin tenerlo, porque **no tiene esa información en la entrada**.

---

## 📈 **Impacto en el Entrenamiento**

### **Durante el entrenamiento:**
- La red aprende que elegir acción 6 (SeeFuture) cuando **no produce cambio** de estado → **reward bajo**
- Con el tiempo, aprende a elegir acción 6 solo cuando **históricamente ha funcionado**
- Esto es **aprendizaje implícito**: la red no sabe explícitamente si tiene la carta, pero aprende patrones de cuándo es útil intentarlo

### **Resultado:**
- El agente desarrolla una "intuición" de cuándo tiene SeeFuture basándose en:
  - ¿Robó una carta recientemente?
  - ¿Qué pasó la última vez que intentó SeeFuture?
  - ¿Cuántas cartas tiene el oponente?

---

## 🔍 **Verificación en el Código**

Puedes ver exactamente qué se incluye en el estado buscando en:
- **Archivo**: `v2/exploding_env.py`
- **Función**: `_get_obs()` (líneas 187-217)
- **Resultado**: Array de 11 valores, donde **NO** aparecen SeeFuture, DrawBottom, Shuffle

---

## 💬 **Conclusión**

La política DQN **no es omnisciente**. Solo ve:
1. Estado del deck (tamaño, bombas, probabilidad)
2. **Algunas** de sus cartas (Defuse, Skip, Attack)
3. Información del oponente (total de cartas, última acción)

Y debe **aprender por experiencia** cuándo tiene otras cartas (SeeFuture, etc.) basándose en patrones indirectos.

Por eso:
- ✅ **Puede intentar** cualquier acción (0-8)
- ❌ **No siempre se ejecuta** (si no tiene la carta)
- 📊 **CSV muestra ambos**: intentos y ejecuciones

---

**¿Te queda claro ahora por qué SeeFuture aparece tanto en los intentos pero no siempre en las ejecuciones?**
