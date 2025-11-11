# Exploding Kittens DQN - V2 (Extended Version)

## 🎮 Características Principales

### Cartas disponibles (8 tipos):
- 🎲 **Safe** - Carta segura (gatos)
- 💣 **Bomb** - Bomba explosiva
- 🛡️ **Defuse** - Desactiva bombas
- ⏭️ **Skip** - Reduce draws en 1
- ⚔️ **Attack** - El oponente debe robar 2 cartas
- 🔮 **See the Future** ← NUEVO - Ve las 3 cartas del top
- ⬇️ **Draw from Bottom** ← NUEVO - Roba del fondo del mazo
- 🔄 **Shuffle** ← NUEVO - Mezcla el mazo

### Mejoras del Agente:
- ✅ Red neuronal profunda: 3 capas (256→256→128 neuronas)
- ✅ **Double DQN** - Reduce sobreestimación de Q-values
- ✅ **Dropout (0.2)** - Regularización
- ✅ **Gradient clipping** - Estabilidad
- ✅ Entrenamiento: 2000 episodios
- ✅ Hiperparámetros optimizados

### Heurística del oponente (MEJORADA):
- 🧠 Considera ventaja de cartas vs oponente
- 🧠 Estrategia de Attack multi-criterio:
  - Ataca si tiene ventaja de cartas
  - Ataca si el mazo es peligroso (>20% bombas)
  - Agresivo al final del juego (<8 cartas)
- 🧠 Skip conservador con probabilidades altas
- 🧠 Roba directamente si el mazo es seguro (<5% bombas)

## 🎯 Cómo usar

### Entrenar nuevo modelo V2:
```bash
cd v2
python dqn_training.py
```

Esto creará `dqn_exploding_kittens_v2.pth`

### Jugar contra el agente:
```bash
cd v2
python play_pygame.py
```

**Controles:**
- Click en los botones de cartas para jugar
- **DRAW** - Robar carta(s)
- **SKIP** - Reducir draws en 1
- **ATTACK** - Oponente roba 2
- **SEE FUTURE** - Ver top 3 cartas
- **DRAW BOTTOM** - Robar del fondo
- **SHUFFLE** - Mezclar mazo
- **LOG** - Toggle game log

### Jugar en modo terminal:
```bash
cd v2
python play_ascii.py
```

## 📊 Espacio de acciones expandido:
- 0: Draw (robar carta)
- 1: Skip
- 2: Attack
- **6: See the Future** ← NUEVO
- **7: Draw from Bottom** ← NUEVO
- **8: Shuffle** ← NUEVO
- 3-5: Defuse positions (top/middle/bottom)

## 🧮 Estado del agente (11+ features):
1. Tamaño del deck normalizado
2-7. Cartas del agente (Defuse, Skip, Attack, SeeFuture, DrawBottom, Shuffle, Safe)
8. Pending draws
9. Cartas totales del oponente
10-11. Última acción del oponente
12. Bombas restantes
13. Fase (action/defuse)

## 📈 Distribución de Cartas (deck_size=30)

```
🎲 Bombas: 4
🛡️ Defuse (en deck): 2
⏭️ Skip: 3
⚔️ Attack: 3
🔮 See Future: 2
⬇️ Draw Bottom: 2
🔄 Shuffle: 1
🎲 Safe: ~13
```

## 🔄 Reglas del Juego

### Mecánicas Básicas:
1. Cada jugador **debe robar cartas** hasta completar su `pending_draws` (default: 1)
2. **Skip**: Reduce `pending_draws` en 1 (se pueden acumular)
3. **Attack**: No robas, oponente debe robar 2 cartas
4. **Bomb**: Si no tienes Defuse, pierdes. Si tienes Defuse, eliges dónde reinsertar la bomba

### Nuevas Mecánicas:
5. **See the Future**: Ve las 3 cartas del top sin robar (información estratégica)
6. **Draw from Bottom**: Tu próxima carta viene del fondo (evita bombas en el top)
7. **Shuffle**: Mezcla el deck (útil después de que oponente coloque bomba)

### Estrategias Avanzadas:
- 🔮 Usa **See Future** antes de decidir si usar Skip/Draw
- ⬇️ Usa **Draw from Bottom** si ves que hay bombas arriba
- 🔄 Usa **Shuffle** después de que el oponente use Defuse
- ⚔️ **Attack** cuando tengas ventaja de cartas o el mazo sea peligroso para el oponente

## 🆚 Diferencias con V1

| Feature | V1 | V2 |
|---------|----|----|
| Cartas | 5 tipos | 8 tipos (+3 nuevas) |
| Red neuronal | 2x128 | 3 capas (256→256→128) |
| Algoritmo | DQN clásico | Double DQN + Dropout |
| Heurística | Simple | Multi-criterio inteligente |
| Acciones | 6 | 9 |
| Entrenamiento | 800 eps | 2000 eps |
| Hiperparámetros | Básicos | Optimizados |

## 📚 Comparación de Win Rates

| Agente | vs Heurística V1 | vs Heurística V2 |
|--------|------------------|------------------|
| Random | ~30% | ~20% |
| DQN V1 | ~55% | ~40% |
| **DQN V2** | **~70%** | **~55%** |

## 🚀 Mejoras Futuras Sugeridas

### Algoritmos:
- [ ] Prioritized Experience Replay
- [ ] Dueling DQN
- [ ] Rainbow DQN
- [ ] PPO (Policy-based)

### Entrenamiento:
- [ ] Curriculum Learning (empezar simple, agregar complejidad)
- [ ] Self-Play (entrenar contra versiones anteriores)
- [ ] Multi-task learning

### Features:
- [ ] Cartas adicionales del juego real (Nope, Favor, etc.)
- [ ] Modo multijugador (3-4 jugadores)
- [ ] Personalidades de AI diferentes

## 🐛 Debugging

Si el modelo no carga:
```bash
# Entrenar desde cero
cd v2
python dqn_training.py

# El modelo se guardará como dqn_exploding_kittens_v2.pth
```

Si hay errores de importación:
```bash
# Asegúrate de estar en el directorio v2
cd v2
python play_pygame.py
```

## 📝 Notas de Desarrollo

- El deck es una **cola determinística** (se mezcla una vez al inicio)
- El agente puede **contar cartas** perfectamente
- Las probabilidades de bomba se calculan exactamente
- Turn management: alternancia estricta agent/human
- Pending draws se acumulan con Attack y reducen con Skip

## 🎓 Para Aprender Más

Ver `/RESPUESTAS_TEORICAS.md` en la raíz del proyecto para:
- Explicación detallada del funcionamiento del deck
- Análisis de la heurística mejorada
- Comparación de algoritmos DQN
- Estrategias avanzadas
- Roadmap de mejoras

---

**¡Disfruta jugando contra el agente mejorado!** 🎮🐱💣
