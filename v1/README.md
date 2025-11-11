# Exploding Kittens DQN - V1 (Simple Version)

## Características

### Cartas disponibles:
- 🎲 **Safe** - Carta segura
- 💣 **Bomb** - Bomba explosiva
- 🛡️ **Defuse** - Desactiva bombas
- ⏭️ **Skip** - Salta tu turno sin robar
- ⚔️ **Attack** - El oponente roba 2 cartas

### Agente:
- Red neuronal simple: 2 capas ocultas de 128 neuronas
- DQN clásico (no Double DQN)
- Entrenamiento: 800-2000 episodios

### Heurística del oponente:
- Simple y predecible
- Usa Skip si deck≤10 y probabilidad bomba>10%
- Usa Attack si probabilidad bomba>15% (70% random)
- Por defecto: Draw

## Cómo usar

### Entrenar nuevo modelo:
```bash
cd v1
python dqn_training.py
```

### Jugar contra el agente:
```bash
cd v1
python play_pygame.py
```

### Jugar en modo terminal:
```bash
cd v1
python play_ascii.py
```

## Espacio de acciones:
- 0: Draw (robar carta)
- 1: Skip
- 2: Attack
- 3-5: Defuse positions (top/middle/bottom)

## Estado del agente (11 features):
1. Tamaño del deck normalizado
2-4. Cartas del agente (Defuse, Skip, Attack, Safe)
5. Pending draws
6. Cartas totales del oponente
7-8. Última acción del oponente (Skip, Attack)
9. Bombas restantes
10. Fase (action/defuse)
