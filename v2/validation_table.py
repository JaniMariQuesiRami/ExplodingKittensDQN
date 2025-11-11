"""
Script para generar tabla de validación detallada en texto plano
Uso: python validation_table.py validation_dqn_YYYYMMDD_HHMMSS.csv
"""

import csv
import sys
import numpy as np
import os
from collections import Counter

def generate_validation_table(filename, output_file=None):
    """Genera tabla detallada de validación en formato texto"""
    
    # Leer CSV
    games = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            games.append({
                'game_id': int(row['game_id']),
                'turns': int(row['turns']),
                'total_reward': float(row['total_reward']),
                'won': int(row['won']),
                # Prefer executed sequence if present, otherwise attempted
                'actions': list(map(int, (row.get('actions_executed_sequence') or row.get('actions_attempted_sequence') or '').split(','))) if (row.get('actions_executed_sequence') or row.get('actions_attempted_sequence')) else []
            })
    
    # Análisis de acciones
    action_counts = {i: 0 for i in range(9)}
    for game in games:
        for a in game['actions']:
            action_counts[a] += 1
    
    total_actions = sum(action_counts.values())
    action_names = {
        0: "Draw", 1: "Skip", 2: "Attack",
        3: "Defuse1", 4: "Defuse2", 5: "Defuse3",
        6: "SeeFuture", 7: "DrawBottom", 8: "Shuffle"
    }
    
    # Estadísticas generales
    wins = sum(g['won'] for g in games)
    win_rate = wins / len(games)
    turns_list = [g['turns'] for g in games]
    rewards_list = [g['total_reward'] for g in games]
    
    # Generar tabla
    output = []
    output.append("=" * 100)
    output.append(f"TABLA DE VALIDACIÓN DETALLADA: {filename}")
    output.append("=" * 100)
    
    # RESUMEN GENERAL
    output.append("\n" + "─" * 100)
    output.append("RESUMEN GENERAL")
    output.append("─" * 100)
    output.append(f"{'Métrica':<30} {'Valor':<20} {'Detalles':<50}")
    output.append("─" * 100)
    output.append(f"{'Total de juegos':<30} {len(games):<20} {'':<50}")
    output.append(f"{'Victorias':<30} {wins:<20} {f'{win_rate:.2%} win rate':<50}")
    output.append(f"{'Derrotas':<30} {len(games)-wins:<20} {f'{(1-win_rate):.2%} loss rate':<50}")
    output.append(f"{'Turnos promedio':<30} {np.mean(turns_list):<20.2f} {f'± {np.std(turns_list):.2f} std, mediana={np.median(turns_list):.0f}':<50}")
    output.append(f"{'Turnos [min, max]':<30} {f'[{min(turns_list)}, {max(turns_list)}]':<20} {'':<50}")
    output.append(f"{'Reward promedio':<30} {np.mean(rewards_list):<20.2f} {f'± {np.std(rewards_list):.2f} std, mediana={np.median(rewards_list):.2f}':<50}")
    
    # DISTRIBUCIÓN DE ACCIONES
    output.append("\n" + "─" * 100)
    output.append("DISTRIBUCIÓN DE ACCIONES")
    output.append("─" * 100)
    output.append(f"{'Acción':<15} {'ID':<5} {'Conteo':<12} {'%':<10} {'Acciones/Juego':<20} {'Barra Visual':<30}")
    output.append("─" * 100)
    
    for i in range(9):
        count = action_counts[i]
        pct = (count / total_actions * 100) if total_actions > 0 else 0
        per_game = count / len(games)
        bar_length = int(pct / 2)  # Escala: 1 char = 2%
        bar = '█' * bar_length
        output.append(f"{action_names[i]:<15} {i:<5} {count:<12} {pct:>6.2f}%   {per_game:>6.2f} veces/juego   {bar:<30}")
    
    output.append("─" * 100)
    output.append(f"{'TOTAL':<15} {'':<5} {total_actions:<12} {'100.00%':<10} {total_actions/len(games):>6.2f} veces/juego")
    
    # ANÁLISIS POR RESULTADO
    output.append("\n" + "─" * 100)
    output.append("ANÁLISIS: VICTORIAS vs DERROTAS")
    output.append("─" * 100)
    
    wins_games = [g for g in games if g['won'] == 1]
    loss_games = [g for g in games if g['won'] == 0]
    
    output.append(f"{'Métrica':<30} {'Victorias (n={len(wins_games)})':<25} {'Derrotas (n={len(loss_games)})':<25} {'Diferencia':<20}")
    output.append("─" * 100)
    
    wins_turns = [g['turns'] for g in wins_games]
    loss_turns = [g['turns'] for g in loss_games]
    output.append(f"{'Turnos promedio':<30} {np.mean(wins_turns):>10.2f} ± {np.std(wins_turns):<7.2f} {np.mean(loss_turns):>10.2f} ± {np.std(loss_turns):<7.2f} {np.mean(wins_turns)-np.mean(loss_turns):>+10.2f}")
    
    wins_actions = sum(len(g['actions']) for g in wins_games)
    loss_actions = sum(len(g['actions']) for g in loss_games)
    output.append(f"{'Acciones promedio':<30} {wins_actions/len(wins_games):>10.2f}               {loss_actions/len(loss_games):>10.2f}               {(wins_actions/len(wins_games))-(loss_actions/len(loss_games)):>+10.2f}")
    
    # Acción más usada en wins vs loss
    wins_action_counts = Counter()
    loss_action_counts = Counter()
    for g in wins_games:
        wins_action_counts.update(g['actions'])
    for g in loss_games:
        loss_action_counts.update(g['actions'])
    
    output.append("\n" + "─" * 100)
    output.append("ACCIONES MÁS FRECUENTES")
    output.append("─" * 100)
    output.append(f"{'Rank':<6} {'En Victorias':<25} {'% Wins':<15} {'En Derrotas':<25} {'% Loss':<15}")
    output.append("─" * 100)
    
    for rank in range(3):
        if rank < len(wins_action_counts.most_common()):
            win_action, win_count = wins_action_counts.most_common()[rank]
            win_pct = win_count / wins_actions * 100
            win_str = f"{action_names[win_action]} ({win_count})"
        else:
            win_str, win_pct = "-", 0
        
        if rank < len(loss_action_counts.most_common()):
            loss_action, loss_count = loss_action_counts.most_common()[rank]
            loss_pct = loss_count / loss_actions * 100
            loss_str = f"{action_names[loss_action]} ({loss_count})"
        else:
            loss_str, loss_pct = "-", 0
        
        output.append(f"{rank+1:<6} {win_str:<25} {win_pct:>6.2f}%        {loss_str:<25} {loss_pct:>6.2f}%")
    
    # TOP JUEGOS
    output.append("\n" + "─" * 100)
    output.append("TOP 10 JUEGOS MÁS LARGOS")
    output.append("─" * 100)
    output.append(f"{'Rank':<6} {'Game ID':<10} {'Turnos':<10} {'Reward':<10} {'Resultado':<12} {'Primeras 10 Acciones':<50}")
    output.append("─" * 100)
    
    longest = sorted(games, key=lambda g: g['turns'], reverse=True)[:10]
    for rank, g in enumerate(longest, 1):
        result = "WIN" if g['won'] else "LOSS"
        actions_preview = ",".join(map(str, g['actions'][:10]))
        if len(g['actions']) > 10:
            actions_preview += "..."
        output.append(f"{rank:<6} #{g['game_id']:<9} {g['turns']:<10} {g['total_reward']:<10.2f} {result:<12} {actions_preview:<50}")
    
    output.append("\n" + "─" * 100)
    output.append("TOP 10 JUEGOS MÁS CORTOS")
    output.append("─" * 100)
    output.append(f"{'Rank':<6} {'Game ID':<10} {'Turnos':<10} {'Reward':<10} {'Resultado':<12} {'Secuencia Completa':<50}")
    output.append("─" * 100)
    
    shortest = sorted(games, key=lambda g: g['turns'])[:10]
    for rank, g in enumerate(shortest, 1):
        result = "WIN" if g['won'] else "LOSS"
        actions_str = ",".join(map(str, g['actions']))
        output.append(f"{rank:<6} #{g['game_id']:<9} {g['turns']:<10} {g['total_reward']:<10.2f} {result:<12} {actions_str:<50}")
    
    # DISTRIBUCIÓN DE TURNOS
    output.append("\n" + "─" * 100)
    output.append("DISTRIBUCIÓN DE TURNOS (Histograma)")
    output.append("─" * 100)
    
    # Crear bins
    bins = [0, 3, 5, 7, 10, 15, 25]
    bin_labels = ["0-2", "3-4", "5-6", "7-9", "10-14", "15+"]
    bin_counts = [0] * len(bin_labels)
    
    for turns in turns_list:
        for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
            if low < turns <= high:
                bin_counts[i] += 1
                break
        else:
            bin_counts[-1] += 1
    
    output.append(f"{'Rango Turnos':<15} {'Juegos':<10} {'%':<10} {'Barra Visual':<50}")
    output.append("─" * 100)
    for label, count in zip(bin_labels, bin_counts):
        pct = count / len(games) * 100
        bar_length = int(pct / 2)
        bar = '█' * bar_length
        output.append(f"{label:<15} {count:<10} {pct:>6.2f}%   {bar:<50}")
    
    output.append("\n" + "=" * 100)
    output.append("FIN DEL REPORTE")
    output.append("=" * 100)
    
    # Escribir a archivo o imprimir
    report_text = "\n".join(output)
    
    if output_file:
        os.makedirs('output', exist_ok=True)
        output_path = os.path.join('output', os.path.basename(output_file))
        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"✅ Tabla guardada en: {output_path}")
    else:
        print(report_text)
    
    return report_text


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Uso: python validation_table.py <csv_file> [output_txt]")
        print("\nEjemplo:")
        print("  python validation_table.py validation_dqn_20241110_203117.csv")
        print("  python validation_table.py validation_dqn_20241110_203117.csv reporte.txt")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    generate_validation_table(csv_file, output_file)
