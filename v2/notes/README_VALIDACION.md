# 📊 Sistema de Validación y Logging - V2

## 🎯 ¿Qué se implementó?

### **1. Heurística de Entrenamiento V2**

#### **Arquitectura de Red Neuronal**
```
Entrada (state_dim) 
    ↓
Linear(256) + ReLU + Dropout(0.2)
    ↓
Linear(256) + ReLU + Dropout(0.2)
    ↓
Linear(128) + ReLU
    ↓
Linear(9 acciones)
```

#### **Técnicas de Optimización**
| Técnica | Valor/Estado | Descripción |
|---------|--------------|-------------|
| **Double DQN** | ✅ Activado | Reduce sobreestimación de Q-values |
| **Gradient Clipping** | max_norm=10.0 | Previene gradientes explosivos |
| **Target Network** | Update cada 100 eps | Mayor estabilidad (antes 50) |
| **Epsilon Decay** | Exponencial (0.9965) | Exploración más suave |
| **Early Stopping** | Patience=10 checkpoints | Guarda mejor modelo |
| **Replay Buffer** | 100,000 experiencias | Memoria de largo plazo |

#### **Hiperparámetros**
```python
batch_size = 128              # ↑ de 64 (más estable)
lr = 5e-4                     # ↓ de 1e-3 (menos agresivo)
epsilon_end = 0.01            # ↓ de 0.05 (más explotación)
gamma = 0.99                  # Factor de descuento
hidden_dim = 256              # ↑ de 128 (más capacidad)
```

---

## 📈 Sistema de Validación con Logging

### **Archivos Generados**

#### **1. CSVs de Validación**
Se generan 2 archivos CSV por entrenamiento:

```csv
# validation_dqn_YYYYMMDD_HHMMSS.csv
game_id,turns,total_reward,won,actions_sequence
1,15,8.5,1,"0,1,2,0,6,8,0,1,..."
2,23,-2.1,0,"0,0,1,2,0,..."
...
```

```csv
# validation_random_YYYYMMDD_HHMMSS.csv
game_id,turns,total_reward,won,actions_sequence
1,28,-5.2,0,"0,2,0,1,0,..."
...
```

#### **2. Gráficas de Entrenamiento**
`training_curves_v2.png`: 2 gráficas lado a lado
- **Izquierda**: Recompensa media (ventana móvil 50 eps)
- **Derecha**: Win rate aproximado (ventana 50 eps)

#### **3. Gráficas de Validación**
`validation_analysis_YYYYMMDD_HHMMSS.png`: 4 gráficas
- **Top-Left**: Comparación Win Rate (DQN vs Random)
- **Top-Right**: Histograma de turnos por juego
- **Bottom-Left**: Distribución de rewards
- **Bottom-Right**: % de uso de cada acción (DQN)

---

## 🚀 Cómo Usar

### **1. Entrenar con Validación**
```bash
cd v2
python dqn_training.py
```

**Salida esperada:**
```
Usando dispositivo: mps
Ep 50/2000 | R_media_50=2.340 | WinRate_50=0.720 | eps=0.862
Ep 100/2000 | R_media_50=4.125 | WinRate_50=0.840 | eps=0.743
  ✅ Nuevo mejor modelo! WinRate: 0.840
...
🛑 Early Stopping activado!
   Mejor WinRate: 0.960 en episodio 800
   Modelo restaurado al episodio 800

============================================================
🧪 FASE DE VALIDACIÓN
============================================================

🤖 Evaluando DQN Agent (400 juegos)...
📝 Log guardado en: validation_dqn_20241110_143022.csv

============================================================
📊 RESUMEN DE VALIDACIÓN - DQN Agent
============================================================
Juegos totales: 400
Win Rate: 96.00% (384/400)
Turnos promedio: 12.45 ± 3.21
Reward promedio: 9.23 ± 2.15

Acción          Total      %         
-----------------------------------
Draw            1245       28.50%
Skip            892        20.41%
Attack          654        14.96%
SeeFuture       789        18.05%
DrawBottom      423        9.68%
Shuffle         365        8.35%
...

🎲 Evaluando Random Agent (400 juegos)...
...
```

### **2. Analizar CSVs**
```bash
# Analizar un solo CSV
python analyze_validation.py validation_dqn_20241110_143022.csv

# Comparar dos CSVs (DQN vs Random)
python analyze_validation.py validation_dqn_*.csv validation_random_*.csv
```

**Ejemplo de salida:**
```
======================================================================
📊 ANÁLISIS DE VALIDACIÓN: validation_dqn_20241110_143022.csv
======================================================================

🎮 MÉTRICAS GENERALES:
  Total de juegos: 400
  Victorias: 384 (96.00%)
  Derrotas: 16

📈 ESTADÍSTICAS DE TURNOS:
  Promedio: 12.45 ± 3.21
  Mediana: 12
  Rango: [6, 24]

💰 ESTADÍSTICAS DE REWARD:
  Promedio: 9.23 ± 2.15
  Mediana: 9.50
  Rango: [-3.20, 15.80]

🎯 DISTRIBUCIÓN DE ACCIONES:
  Acción          Conteo     %          █                   
  -------------------------------------------------------
  Draw            1245       28.50%     █████
  Skip            892        20.41%     ████
  Attack          654        14.96%     ██
  SeeFuture       789        18.05%     ███
  DrawBottom      423        9.68%      █
  Shuffle         365        8.35%      █

🏆 TOP 5 JUEGOS MÁS LARGOS:
  1. Game #127: 24 turnos, reward=12.30 ✅ WIN
  2. Game #89: 23 turnos, reward=11.80 ✅ WIN
  ...

⚡ TOP 5 JUEGOS MÁS CORTOS:
  1. Game #234: 6 turnos, reward=-2.10 ❌ LOSS
  2. Game #45: 7 turnos, reward=8.50 ✅ WIN
  ...

💎 EJEMPLO DE SECUENCIA DE ACCIONES (Game #1):
  Draw → Skip → SeeFuture → Draw → Shuffle → Draw → Skip → Attack...
```

---

## 🔍 Interpretación de Métricas

### **Acciones (0-8)**
| ID | Nombre | Descripción |
|----|--------|-------------|
| 0 | Draw | Robar carta del deck |
| 1 | Skip | Saltar turno (evita robar) |
| 2 | Attack | Termina turno, oponente roba 2 |
| 3-5 | Defuse1-3 | Desactivar bomba (3 slots) |
| 6 | SeeFuture | Ver 3 cartas del deck |
| 7 | DrawBottom | Robar del fondo del deck |
| 8 | Shuffle | Barajar el deck |

### **Win Rate Esperado**
- **Random Agent**: ~35-45%
- **DQN V2 (entrenado)**: ~92-96%
- **Meta**: >85% consistente

### **Turnos Promedio**
- **Random**: ~18-25 turnos (juega ineficientemente)
- **DQN V2**: ~10-15 turnos (estrategia eficiente)

### **Uso de Acciones Óptimo (DQN)**
- **Draw**: ~25-30% (principal acción)
- **Skip**: ~15-25% (evita bombas)
- **SeeFuture**: ~15-20% (planificación)
- **Attack**: ~10-15% (presión al oponente)
- **Shuffle**: ~5-10% (reorganizar deck peligroso)

---

## 📁 Estructura de Archivos

```
v2/
├── dqn_training.py                    # ← Script principal de entrenamiento
├── analyze_validation.py              # ← Análisis de CSVs
├── training_curves_v2.png             # ← Gráficas de entrenamiento
├── validation_analysis_*.png          # ← Gráficas de validación
├── validation_dqn_*.csv               # ← Log de juegos DQN
├── validation_random_*.csv            # ← Log de juegos Random
├── dqn_exploding_kittens_v2.pth       # ← Modelo final
└── best_model_checkpoint.pth          # ← Checkpoint del mejor modelo
```

---

## 🧪 Diferencias: Entrenamiento vs Validación

### **ENTRENAMIENTO** (Durante `train_dqn()`)
- **Objetivo**: Aprender política óptima
- **Epsilon**: Decae de 1.0 → 0.01 (exploración → explotación)
- **Replay Buffer**: Se llena con experiencias
- **Actualizaciones**: Gradientes, target network, early stopping
- **Métricas**: Reward medio, win rate aproximado (ventana 50 eps)
- **Gráficas**: Curvas de aprendizaje durante entrenamiento

### **VALIDACIÓN** (Después de entrenar)
- **Objetivo**: Evaluar rendimiento real del modelo entrenado
- **Epsilon**: 0 (sin exploración, solo explotación)
- **Sin aprendizaje**: No se actualizan pesos
- **Métricas detalladas**: 
  - Win rate exacto (400 juegos)
  - Distribución de turnos
  - Distribución de rewards
  - Análisis de acciones tomadas
  - CSV con cada juego individual
- **Gráficas**: Histogramas, comparaciones, análisis de comportamiento

---

## 💡 Tips para Interpretación

### **Si Win Rate < 85%**
- ✅ Entrenar más episodios (aumentar `num_episodes`)
- ✅ Ajustar `epsilon_decay_rate` (más exploración)
- ✅ Revisar arquitectura (aumentar `hidden_dim`)

### **Si Win Rate > 95% pero turnos muy largos**
- ⚠️ Posible sobreajuste a estrategia defensiva
- Revisar función de reward (penalizar turnos largos)

### **Si usa mucho Draw y poco SeeFuture**
- ⚠️ No está usando información disponible
- Reward de SeeFuture podría ser muy bajo

### **Comparación DQN vs Random**
- DQN debería tener:
  - ✅ Win rate +50-60% mayor
  - ✅ Turnos promedio -30% menor
  - ✅ Reward promedio significativamente mayor
  - ✅ Mayor uso de Skip/SeeFuture/Shuffle

---

## 🎓 Preguntas Frecuentes

**Q: ¿Por qué hay 2 gráficas separadas (training y validation)?**  
A: Training muestra el proceso de aprendizaje (con ruido). Validation mide el rendimiento real sin exploración.

**Q: ¿Qué significa "actions_sequence" en el CSV?**  
A: La secuencia completa de acciones del juego. Ejemplo: `"0,1,6,0,2"` = Draw, Skip, SeeFuture, Draw, Attack.

**Q: ¿Puedo comparar CSVs de diferentes entrenamientos?**  
A: Sí! Usa `analyze_validation.py archivo1.csv archivo2.csv` para comparar.

**Q: ¿Cómo sé si el early stopping funcionó bien?**  
A: Si el modelo restaurado tiene win rate similar al pico de entrenamiento (±2%).

---

**Autor**: Sistema de Entrenamiento DQN V2  
**Fecha**: Noviembre 2024  
**Versión**: 2.0 (Con validación exhaustiva)
