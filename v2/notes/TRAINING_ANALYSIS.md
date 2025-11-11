# TRAINING_ANALYSIS.md - Análisis y Mejoras del Entrenamiento DQN

## 📊 Análisis de tus Resultados

### **Métricas Explicadas:**

| Métrica | Qué es | Rango | Tu Resultado | Importancia |
|---------|--------|-------|--------------|-------------|
| **WinRate_50** | % victorias últimos 50 eps | 0-100% | 38% → 92% → 74% | ⭐⭐⭐⭐⭐ **LA MÁS IMPORTANTE** |
| **R_media_50** | Reward promedio últimos 50 | -1 a +1 | -0.24 → 0.84 → 0.48 | ⭐⭐⭐⭐ Indica tendencia |
| **eps (epsilon)** | Probabilidad exploración | 0-100% | 96.7% → 1% | ⭐⭐⭐ Control interno |

### **Interpretación R_media:**
```
R_media = (wins - losses) / total_episodes

Ejemplos en 50 eps:
  0.84 = (46 wins, 4 losses, 0 draws) = 92% win rate
  0.48 = (37 wins, 13 losses, 0 draws) = 74% win rate
  0.00 = (25 wins, 25 losses, 0 draws) = 50% win rate
 -0.24 = (19 wins, 31 losses, 0 draws) = 38% win rate
```

---

## 🎯 **Problema Principal: ¿Por qué baja el WinRate?**

### **Tu curva de aprendizaje:**
```
Episodios    WinRate    Fase
1-500        38-54%     🔴 Exploración caótica
500-1000     54-64%     🟡 Aprendizaje gradual
1000-1450    64-92%     🟢 Mejora acelerada ← PICO
1450-1650    92-86%     🟠 Declive moderado
1650-2000    86-68-90%  🔴 Inestabilidad ← PROBLEMA
```

### **Causas del Declive:**

#### 1. **Overfitting al Oponente Heurístico** 🎯
**Qué pasa:**
- Tu agente aprende a explotar patrones específicos del oponente
- El oponente heurístico tiene probabilidades random (70% attack, etc.)
- Cuando el random sale diferente, el agente falla

**Evidencia en tus datos:**
```
Ep 1450: 92% win rate (explota patrones perfectamente)
Ep 1750: 76% win rate (patrones cambian, agente confundido)
Ep 1850: 90% win rate (re-aprende)
Ep 1950: 68% win rate (patrones cambian otra vez)
```

#### 2. **Catastrophic Forgetting** 🧠💥
**Qué pasa:**
- Al final (eps=0.01), casi no explora
- Solo ve escenarios donde ya es bueno
- Olvida cómo manejar situaciones raras

**Ejemplo:**
```python
# Episodio 1500+: eps=0.01
# Agente casi siempre gana → solo ve "estados ganadores"
# Si encuentra estado inusual → no sabe qué hacer
```

#### 3. **Replay Buffer Desbalanceado** 📦
**Qué pasa:**
- Buffer tiene 100K experiencias
- Incluye episodios 1-500 donde el agente era malo
- Al final, mezcla experiencias buenas (1500+) con malas (1-500)

**Solución:** Usar Prioritized Experience Replay

#### 4. **Target Network Update Frequency** 🎯↔️🎯
**Tu config actual:** Actualiza cada 100 episodios
- **Si muy frecuente:** Moving target, inestabilidad
- **Si muy lento:** Aprende de valores desactualizados

---

## 🔧 **Mejoras Implementadas en V2:**

### ✅ Ya tienes (en tu código actual):
1. **Double DQN** - Reduce overestimation de Q-values
2. **Dropout (0.2)** - Regularización contra overfitting
3. **Gradient Clipping (10.0)** - Evita gradientes explosivos
4. **Red más grande (256-256-128)** - Más capacidad de aprendizaje
5. **Learning rate bajo (5e-4)** - Más estable

### 🆕 Mejoras Adicionales Recomendadas:

---

## 💡 **Soluciones al Problema del Declive:**

### **Opción 1: Early Stopping (Más Simple)** ⭐
**Concepto:** Detener entrenamiento en el pico (ep ~1450)

```python
# Modificación simple en train_dqn():
best_win_rate = 0
patience = 5  # Episodios sin mejora antes de parar
no_improvement = 0

if (ep + 1) % 50 == 0:
    win_rate = np.mean(episode_wins[-50:])
    
    if win_rate > best_win_rate:
        best_win_rate = win_rate
        no_improvement = 0
        # Guardar mejor modelo
        torch.save(q_net.state_dict(), 'best_model.pth')
    else:
        no_improvement += 1
    
    if no_improvement >= patience:
        print(f"Early stopping at ep {ep+1}, best WR: {best_win_rate:.3f}")
        break
```

**Resultado esperado:** Detiene en ep ~1500 con 92% win rate

---

### **Opción 2: Epsilon Scheduling Mejorado** 📉
**Problema actual:** Epsilon baja linealmente muy rápido

```python
# ACTUAL (lineal):
eps: 0.967 → 0.01 en 1500 episodios

# MEJOR (cosine decay):
import math

def cosine_epsilon(ep, total_eps, eps_start, eps_end):
    if ep >= total_eps:
        return eps_end
    return eps_end + (eps_start - eps_end) * (1 + math.cos(math.pi * ep / total_eps)) / 2

# O exponential decay (más suave):
def exp_epsilon(ep, eps_start, eps_end, decay_rate=0.995):
    return max(eps_end, eps_start * (decay_rate ** ep))
```

**Ventaja:** Mantiene exploración por más tiempo, evita convergencia prematura

---

### **Opción 3: Prioritized Experience Replay** 🎯
**Concepto:** Samplear más frecuentemente las experiencias importantes

```python
class PrioritizedReplayBuffer:
    def __init__(self, capacity=100_000, alpha=0.6):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha  # Cuánto priorizar
    
    def push(self, transition, error):
        # error = |Q_predicted - Q_target|
        priority = (abs(error) + 1e-5) ** self.alpha
        self.buffer.append(transition)
        self.priorities.append(priority)
    
    def sample(self, batch_size, beta=0.4):
        # beta: importance sampling weight (0.4 → 1.0 durante training)
        probs = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        # ... return samples con weights
```

**Ventaja:** Aprende más rápido de errores grandes, ignora experiencias ya dominadas

---

### **Opción 4: Curriculum Learning** 📚
**Concepto:** Entrenar contra oponentes de dificultad creciente

```python
# Fase 1 (eps 0-500): Oponente random
# Fase 2 (eps 500-1000): Oponente heurístico simple
# Fase 3 (eps 1000-1500): Oponente heurístico inteligente
# Fase 4 (eps 1500+): Self-play (agente vs agente)

if ep < 500:
    env.opponent_difficulty = 'random'
elif ep < 1000:
    env.opponent_difficulty = 'simple'
elif ep < 1500:
    env.opponent_difficulty = 'intelligent'
else:
    env.opponent_difficulty = 'self_play'
```

**Ventaja:** Generalización mucho mejor, menos overfitting

---

### **Opción 5: Learning Rate Scheduling** 📊
**Concepto:** Reducir learning rate conforme avanza training

```python
# Opción A: Step decay
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
# lr: 5e-4 → 2.5e-4 (ep 500) → 1.25e-4 (ep 1000)

# Opción B: Cosine annealing
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000)

# Opción C: Reduce on plateau
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3
)
# Reduce lr si win rate no mejora en 3 checkpoints
```

**Ventaja:** Aprendizaje fino al final, menos oscilaciones

---

## 🎮 **Recomendaciones Prácticas:**

### **Para tu caso específico:**

#### **Solución Rápida (5 minutos):** 
✅ Usar modelo del episodio ~1450 (92% win rate)
```bash
# Ya tienes el modelo guardado
# Solo necesitas renombrar el checkpoint correcto
```

#### **Solución Media (30 minutos):**
1. Implementar **Early Stopping** (detiene en pico)
2. Cambiar **epsilon_decay** a exponencial (más exploración)
3. Reducir **target_update_interval** de 100 → 50 (más actualizado)

#### **Solución Avanzada (2-3 horas):**
1. Implementar **Prioritized Replay**
2. Implementar **Curriculum Learning** (3 fases de dificultad)
3. Usar **Learning Rate Scheduling**
4. **Self-play** en fase final

---

## 📈 **Curva Ideal vs Tu Curva:**

### **Curva Ideal:**
```
WinRate
100% |                    _______________
     |                 __/
 80% |              __/
 60% |          __/
 40% |      __/
 20% |  __/
  0% |_/
     +--------------------------------
     0     500    1000   1500   2000  Episodios
```

### **Tu Curva:**
```
WinRate
100% |                    /\
     |                   /  \__/\
 80% |                  /      \/\
 60% |              __/
 40% |      ___~~~
 20% |  __/
  0% |_/
     +--------------------------------
     0     500    1000   1500   2000  Episodios
               PROBLEMA: Inestabilidad →
```

---

## 🎯 **Métricas Objetivo:**

| Métrica | Valor Actual | Valor Ideal | Prioridad |
|---------|--------------|-------------|-----------|
| **Win Rate final** | 74% | 85-90% | ⭐⭐⭐⭐⭐ |
| **Win Rate pico** | 92% | 92-95% | ⭐⭐⭐⭐ |
| **Estabilidad (std últimos 500 eps)** | ~8% | <5% | ⭐⭐⭐⭐⭐ |
| **Win vs Random** | 76.5% | 90%+ | ⭐⭐⭐ |
| **Episodes to convergence** | ~1450 | <1000 | ⭐⭐ |

---

## 💻 **Código de Diagnóstico:**

```python
# Agregar al final de train_dqn() para análisis:

import matplotlib.pyplot as plt

def plot_training_curves(episode_wins, window=50):
    # Win rate smoothed
    win_rates = [np.mean(episode_wins[max(0, i-window):i+1]) 
                 for i in range(len(episode_wins))]
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(win_rates, alpha=0.8)
    plt.axhline(y=0.92, color='r', linestyle='--', label='Pico (92%)')
    plt.axhline(y=0.85, color='g', linestyle='--', label='Target (85%)')
    plt.xlabel('Episodio')
    plt.ylabel('Win Rate')
    plt.title(f'Win Rate (ventana {window})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Histogram de últimos 500
    plt.subplot(1, 2, 2)
    plt.hist(episode_wins[-500:], bins=2, alpha=0.7)
    plt.xlabel('Resultado (0=Loss, 1=Win)')
    plt.ylabel('Frecuencia')
    plt.title('Distribución últimos 500 eps')
    plt.savefig('training_analysis.png', dpi=150)
    plt.close()
    
    # Stats
    print(f"\n📊 Estadísticas finales:")
    print(f"Win rate últimos 500: {np.mean(episode_wins[-500:]):.3f}")
    print(f"Win rate pico: {max(win_rates):.3f}")
    print(f"Episodio del pico: {np.argmax(win_rates)}")
    print(f"Estabilidad (std últimos 500): {np.std(win_rates[-500:]):.3f}")

# Llamar después de train_dqn():
plot_training_curves(episode_wins)
```

---

## ✅ **Conclusión:**

### **Tu entrenamiento actual:**
- ✅ **Bueno:** Alcanza 92% win rate (excelente pico)
- ✅ **Bueno:** Win rate final 74% > 76.5% vs random (superó baseline)
- ⚠️ **Problema:** Inestabilidad después del pico (92% → 68%)
- ⚠️ **Problema:** No mantiene performance máxima

### **¿Qué hacer?**

1. **Solución inmediata:** Usa el modelo del pico (ep ~1450)
2. **Mejor entrenamiento:** Implementa Early Stopping + Epsilon exponencial
3. **Máximo rendimiento:** Prioritized Replay + Curriculum Learning

### **¿Cuál métrica priorizar?**
1. **WinRate_50** (la más importante) - Tu objetivo
2. **Estabilidad** (std pequeña) - Mantener performance
3. **R_media** (correlaciona con WinRate) - Indicador secundario
4. **eps** (solo para debugging) - Control interno

---

**Siguiente paso:** ¿Quieres que implemente alguna de estas mejoras?
