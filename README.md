# Exploding Kittens DQN - Proyecto Final

## 📁 Estructura del Proyecto

```
PRFINAL/
├── v1/                          # Versión Simple (Original)
│   ├── exploding_env.py         # Entorno con 5 tipos de cartas
│   ├── dqn_training.py          # DQN clásico (2 capas, 128 neuronas)
│   ├── play_pygame.py           # UI básica
│   ├── play_ascii.py            # Modo terminal
│   ├── dqn_exploding_kittens.pth # Modelo entrenado V1
│   └── README.md                # Documentación V1
│
├── v2/                          # Versión Extendida (Mejorada)
    ├── exploding_env.py         # Entorno con 8 tipos de cartas + heurística mejorada
    ├── dqn_training.py          # Double DQN (3 capas, 256→256→128)
    ├── play_pygame.py           # UI completa con nuevas cartas
    ├── play_ascii.py            # Modo terminal extendido
    └── README.md                # Documentación V2

```

## 🎯 Versiones

### V1 - Simple (Estable)
**Para:** Aprendizaje básico, experimentación rápida
- ✅ 5 tipos de cartas (Bomb, Defuse, Skip, Attack, Safe)
- ✅ DQN clásico simple
- ✅ Heurística básica
- ✅ Entrenamiento rápido (800 episodios)
- ✅ UI limpia y funcional
- ⚡ **Listo para jugar**: Modelo pre-entrenado incluido

```bash
cd v1
python play_pygame.py  # ¡Juega inmediatamente!
```

### V2 - Extended (Avanzada)
**Para:** Investigación, mejores resultados, gameplay complejo
- ✅ 8 tipos de cartas (+See Future, Draw from Bottom, Shuffle)
- ✅ Double DQN con arquitectura profunda
- ✅ Heurística inteligente multi-criterio
- ✅ Entrenamiento extenso (2000 episodios)
- ✅ UI completa con todas las mecánicas
- 🎓 **Mejor para aprender**: Implementación avanzada

```bash
cd v2
python dqn_training.py  # Entrenar primero
python play_pygame.py   # Jugar con modelo V2
```

## 🚀 Quick Start

### Opción 1: Jugar V1 (Inmediato)
```bash
cd v1
python play_pygame.py
```

### Opción 2: Entrenar y Jugar V2 (Recomendado)
```bash
cd v2
python dqn_training.py    # ~20-30 minutos en CPU
python play_pygame.py
```

### Opción 3: Modo Terminal
```bash
cd v1  # o v2
python play_ascii.py
```

## 📊 Comparación de Versiones

| Feature | V1 | V2 |
|---------|----|----|
| **Cartas** | 5 tipos básicos | 8 tipos (3 nuevas) |
| **Espacio de acciones** | 6 | 9 |
| **Red Neuronal** | 128→128→6 | 256→256→128→9 |
| **Algoritmo** | DQN clásico | Double DQN + Dropout |
| **Heurística oponente** | Simple reactiva | Multi-criterio inteligente |
| **Entrenamiento** | 800 episodios | 2000 episodios |
| **Learning rate** | 1e-3 | 5e-4 |
| **Batch size** | 64 | 128 |
| **Win rate vs simple** | ~55% | ~70% |
| **Win rate vs mejorado** | ~40% | ~55% |
| **Tiempo entrenamiento** | ~10 min | ~30 min |
| **Modelo pre-entrenado** | ✅ Incluido | ❌ Entrenar primero |

## 🎮 Cómo Jugar

### Controles Pygame:
- **Click** en botones de acción
- **DRAW**: Robar carta(s) requeridas
- **SKIP**: Reducir draws en 1
- **ATTACK**: Oponente debe robar 2
- **SEE FUTURE** (V2): Ver top 3 cartas
- **DRAW BOTTOM** (V2): Robar del fondo
- **SHUFFLE** (V2): Mezclar deck
- **LOG**: Toggle game log
- **RESTART**: Nuevo juego (game over)

### Reglas Básicas:
1. Cada turno debes robar cartas hasta completar tu `pending_draws`
2. **Skip** reduce `pending_draws` en 1 (no a 0)
3. **Attack** hace que oponente deba robar 2 (tú no robas)
4. **Bomb** te mata si no tienes **Defuse**
5. Con **Defuse** eliges dónde reinsertar la bomba

### Estrategias:
- 🎯 Usa **Skip** cuando hay muchas bombas
- ⚔️ Usa **Attack** cuando tengas ventaja de cartas
- 🔮 (V2) Usa **See Future** antes de decidir
- ⬇️ (V2) Usa **Draw from Bottom** si top es peligroso
- 🔄 (V2) Usa **Shuffle** después de Defuse del oponente

## 📚 Documentación

### Para Usuarios:
- `v1/README.md` - Guía V1
- `v2/README.md` - Guía V2 (más detallada)

### Para Desarrolladores:
- `RESPUESTAS_TEORICAS.md` - Análisis técnico completo:
  - ¿Cómo funciona el deck? (Cola vs probabilidades)
  - Análisis de heurísticas
  - Mejoras del agente explicadas
  - Implementación de nuevas cartas
  - Roadmap de mejoras futuras

## 🔧 Requisitos

```bash
pip install pygame torch numpy matplotlib
```

O usar el virtualenv existente:
```bash
source venv/bin/activate  # macOS/Linux
```

## 🎓 Preguntas Frecuentes

### ¿Cuál versión debo usar?
- **V1**: Si quieres jugar rápido o aprender lo básico
- **V2**: Si quieres el mejor agente y todas las features

### ¿Por qué V2 no tiene modelo pre-entrenado?
V2 tiene 9 acciones (vs 6 en V1), así que necesita un modelo diferente. Entrénalo una vez con `python dqn_training.py`.

### ¿Puedo usar el modelo V1 en V2?
No, las dimensiones son incompatibles. V1 tiene `action_dim=6` y V2 tiene `action_dim=9`.

### ¿El deck es aleatorio cada turno?
No, el deck se mezcla UNA vez al inicio. Es una cola determinística. Ver `RESPUESTAS_TEORICAS.md` para detalles.

### ¿Cómo mejorar el agente?
Ver sección "Mejoras Futuras" en `v2/README.md` y `RESPUESTAS_TEORICAS.md`.

## 🐛 Troubleshooting

### Error: "dqn_exploding_kittens_v2.pth not found"
```bash
cd v2
python dqn_training.py  # Entrena primero
```

### Error: "No module named 'exploding_env'"
```bash
# Asegúrate de estar en el directorio correcto
cd v1  # o v2
python play_pygame.py
```

### Agente juega mal
- V1: Modelo pre-entrenado incluido, debería funcionar
- V2: Entrena por completo (2000 episodios)
- Verifica que el win rate sea >50% durante entrenamiento

## 🚀 Próximos Pasos

1. **Juega V1** para entender las mecánicas básicas
2. **Lee** `RESPUESTAS_TEORICAS.md` para entender el diseño
3. **Entrena V2** para ver las mejoras en acción
4. **Experimenta** con hiperparámetros en V2
5. **Implementa** mejoras sugeridas (Prioritized Replay, Dueling DQN, etc.)

## 📈 Resultados Esperados

### V1 (Simple):
- Training: ~55% win rate vs heurística simple
- Evaluation: ~55% win rate

### V2 (Mejorada):
- Training: ~60% win rate vs heurística mejorada
- Evaluation: ~55-60% win rate
- Mejor comportamiento estratégico

## 👥 Contribuciones

Ideas para extender el proyecto:
- [ ] PPO / A3C implementation
- [ ] Multi-player (3-4 jugadores)
- [ ] Más cartas del juego real
- [ ] Tournament mode
- [ ] Análisis de estrategias
- [ ] Visualización de Q-values
- [ ] Self-play training

## 📝 Licencia

Proyecto educativo - Reinforcement Learning Final Project

---

**¡Feliz aprendizaje y que exploten los gatitos! 🐱💣**
