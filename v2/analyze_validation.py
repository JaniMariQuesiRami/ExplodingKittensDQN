"""
Script para analizar los CSVs de validación generados por dqn_training.py
Uso: python analyze_validation.py validation_dqn_YYYYMMDD_HHMMSS.csv
"""

import csv
import sys
import numpy as np
from collections import Counter

def analyze_csv(filename):
    """Lee y analiza un CSV de validación"""
    games = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            games.append({
                'game_id': int(row['game_id']),
                'turns': int(row['turns']),
                'total_reward': float(row['total_reward']),
                'won': int(row['won']),
                # Prefer executed sequence if available, otherwise attempted
                'actions': list(map(int, (row.get('actions_executed_sequence') or row.get('actions_attempted_sequence') or '').split(','))) if (row.get('actions_executed_sequence') or row.get('actions_attempted_sequence')) else []
            })
    
    # Estadísticas generales
    total_games = len(games)
    wins = sum(g['won'] for g in games)
    win_rate = wins / total_games
    
    turns_list = [g['turns'] for g in games]
    rewards_list = [g['total_reward'] for g in games]
    
    # Análisis de acciones
    all_actions = []
    for g in games:
        all_actions.extend(g['actions'])
    
    action_counter = Counter(all_actions)
    action_names = {
        0: "Draw", 1: "Skip", 2: "Attack",
        3: "Defuse1", 4: "Defuse2", 5: "Defuse3",
        6: "SeeFuture", 7: "DrawBottom", 8: "Shuffle"
    }
    
    print(f"\n{'='*70}")
    print(f"📊 ANÁLISIS DE VALIDACIÓN: {filename}")
    print(f"{'='*70}")
    print(f"\n🎮 MÉTRICAS GENERALES:")
    print(f"  Total de juegos: {total_games}")
    print(f"  Victorias: {wins} ({win_rate:.2%})")
    print(f"  Derrotas: {total_games - wins}")
    
    print(f"\n📈 ESTADÍSTICAS DE TURNOS:")
    print(f"  Promedio: {np.mean(turns_list):.2f} ± {np.std(turns_list):.2f}")
    print(f"  Mediana: {np.median(turns_list):.0f}")
    print(f"  Rango: [{min(turns_list)}, {max(turns_list)}]")
    
    print(f"\n💰 ESTADÍSTICAS DE REWARD:")
    print(f"  Promedio: {np.mean(rewards_list):.2f} ± {np.std(rewards_list):.2f}")
    print(f"  Mediana: {np.median(rewards_list):.2f}")
    print(f"  Rango: [{min(rewards_list):.2f}, {max(rewards_list):.2f}]")
    
    print(f"\n🎯 DISTRIBUCIÓN DE ACCIONES:")
    print(f"  {'Acción':<15} {'Conteo':<10} {'%':<10} {'█':<20}")
    print(f"  {'-'*55}")
    
    total_actions = sum(action_counter.values())
    for action_id in sorted(action_counter.keys()):
        count = action_counter[action_id]
        pct = (count / total_actions) * 100
        bar_length = int(pct / 5)  # Escala: 1 char = 5%
        bar = '█' * bar_length
        print(f"  {action_names[action_id]:<15} {count:<10} {pct:>6.2f}%   {bar}")
    
    print(f"\n🏆 TOP 5 JUEGOS MÁS LARGOS:")
    longest_games = sorted(games, key=lambda g: g['turns'], reverse=True)[:5]
    for i, g in enumerate(longest_games, 1):
        result = "✅ WIN" if g['won'] else "❌ LOSS"
        print(f"  {i}. Game #{g['game_id']}: {g['turns']} turnos, reward={g['total_reward']:.2f} {result}")
    
    print(f"\n⚡ TOP 5 JUEGOS MÁS CORTOS:")
    shortest_games = sorted(games, key=lambda g: g['turns'])[:5]
    for i, g in enumerate(shortest_games, 1):
        result = "✅ WIN" if g['won'] else "❌ LOSS"
        print(f"  {i}. Game #{g['game_id']}: {g['turns']} turnos, reward={g['total_reward']:.2f} {result}")
    
    if games and games[0]['actions']:
        print(f"\n💎 EJEMPLO DE SECUENCIA DE ACCIONES (Game #{games[0]['game_id']}):")
        example_actions = games[0]['actions'][:20]  # Primeras 20 acciones
        action_str = " → ".join([action_names[a] for a in example_actions])
        print(f"  {action_str}...")
    
    print(f"\n{'='*70}\n")
    
    return {
        'win_rate': win_rate,
        'avg_turns': np.mean(turns_list),
        'avg_reward': np.mean(rewards_list),
        'action_distribution': action_counter
    }


def compare_csvs(csv1, csv2):
    """Compara dos CSVs (ej: DQN vs Random)"""
    print("\n" + "="*70)
    print("⚖️  COMPARACIÓN ENTRE AGENTES")
    print("="*70)
    
    stats1 = analyze_csv(csv1)
    stats2 = analyze_csv(csv2)
    
    print(f"\n📊 RESUMEN COMPARATIVO:")
    print(f"  {'Métrica':<20} {'Archivo 1':<20} {'Archivo 2':<20} {'Diferencia':<15}")
    print(f"  {'-'*75}")
    print(f"  {'Win Rate':<20} {stats1['win_rate']:<20.2%} {stats2['win_rate']:<20.2%} {stats1['win_rate']-stats2['win_rate']:>+14.2%}")
    print(f"  {'Turnos promedio':<20} {stats1['avg_turns']:<20.2f} {stats2['avg_turns']:<20.2f} {stats1['avg_turns']-stats2['avg_turns']:>+14.2f}")
    print(f"  {'Reward promedio':<20} {stats1['avg_reward']:<20.2f} {stats2['avg_reward']:<20.2f} {stats1['avg_reward']-stats2['avg_reward']:>+14.2f}")
    print("="*70 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Uso: python analyze_validation.py <csv_file> [csv_file2]")
        print("\nEjemplo:")
        print("  python analyze_validation.py validation_dqn_20241110_143022.csv")
        print("  python analyze_validation.py validation_dqn_*.csv validation_random_*.csv")
        sys.exit(1)
    
    csv_file1 = sys.argv[1]
    
    if len(sys.argv) == 3:
        csv_file2 = sys.argv[2]
        compare_csvs(csv_file1, csv_file2)
    else:
        analyze_csv(csv_file1)
