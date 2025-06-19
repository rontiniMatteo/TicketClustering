#!/usr/bin/env python3

import csv
import sys
import argparse
import os
import glob

# Aumenta il limite per campi di grandi dimensioni
csv.field_size_limit(10**8)  # 100 milioni di caratteri


def find_csv_file(directory):
    """
    Trova un unico file CSV nella directory specificata.
    """
    csv_files = glob.glob(os.path.join(directory, '*.csv'))
    if not csv_files:
        print(f"Nessun file CSV trovato in {directory}.", file=sys.stderr)
        sys.exit(1)
    if len(csv_files) > 1:
        print(
            f"Trovati più CSV in {directory}: {', '.join(os.path.basename(f) for f in csv_files)}. "
            "Specificare l'input con --input-file.",
            file=sys.stderr
        )
        sys.exit(1)
    return csv_files[0]


def filter_csv(
    input_path,
    output_path,
    removed_path,
    delimiter=';',
    column_d_index=3,
    img_pattern='<img src'
):
    """
    Legge input_path, filtra i record secondo queste regole:
      - colonna A (index 0): deve contenere esattamente 'S'; se inizia con '<' o diverso da 'S', viene rimossa
      - colonna D (index=column_d_index):
        * inizio '<span'
        * inizio '<ul>'
        * inizio 'divisione destinazione'
        * contiene img_pattern ('<img src')
      - colonna C (index=2):
        * inizio uno dei prefissi:
          'cambio di mansione', 'creazione utenza', 'ritiro postazione',
          'cambio di centro', 'richiesta di inoltro', 'richiesta account',
          'installazione ms', 'richiesta di installazione hardware',
          'cambiamento centro di costo'
    Le righe rimosse vengono salvate in removed_path, le restanti in output_path.
    """
    remove_c_prefixes = [
        'cambio di mansione',
        'creazione utenza',
        'ritiro postazione',
        'cambio di centro',
        'richiesta di inoltro',
        'richiesta account',
        'installazione ms',
        'richiesta di installazione hardware',
        'cambiamento centro di costo'
    ]

    with open(input_path, newline='', encoding='utf-8') as infile, \
         open(output_path, 'w', newline='', encoding='utf-8') as outfile, \
         open(removed_path, 'w', newline='', encoding='utf-8') as remfile:
        reader = csv.reader(infile, delimiter=delimiter)
        writer = csv.writer(outfile, delimiter=delimiter)
        rem_writer = csv.writer(remfile, delimiter=delimiter)

        # Lettura header con gestione file vuoto
        try:
            header = next(reader)
        except StopIteration:
            print(f"Il file '{input_path}' è vuoto.", file=sys.stderr)
            sys.exit(1)
        writer.writerow(header)
        rem_writer.writerow(header)

        removed = 0
        kept = 0
        for row in reader:
            # Colonna A: deve essere esattamente 'S'; se inizia con '<' o diverso, rimuovi
            cell_a = row[0] if len(row) > 0 else ''
            stripped_a = cell_a.strip()
            a_remove = stripped_a != 'S' or cell_a.lstrip().startswith('<')

            # Colonna D
            cell_d = row[column_d_index] if len(row) > column_d_index else ''
            d_str = cell_d.lstrip()
            d_lower = d_str.lower()
            d_remove = (
                d_lower.startswith('<span') or
                d_lower.startswith('<ul>') or
                d_lower.startswith('divisione destinazione') or
                img_pattern in cell_d
            )

            # Colonna C
            cell_c = row[2] if len(row) > 2 else ''
            c_str = cell_c.lstrip()
            c_lower = c_str.lower()
            c_remove = any(c_lower.startswith(pref) for pref in remove_c_prefixes)

            # Se uno qualsiasi dei criteri è soddisfatto, rimuovi
            if a_remove or d_remove or c_remove:
                rem_writer.writerow(row)
                removed += 1
            else:
                writer.writerow(row)
                kept += 1

    print(f"Filtrate {removed} righe in '{removed_path}'; {kept} righe mantenute in '{output_path}'")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Filtra un CSV rimuovendo righe in base a criteri su colonne A, C e D."
    )
    parser.add_argument(
        '--input-file', '-i',
        help='Percorso al file CSV di input'
    )
    parser.add_argument(
        '--output-file', '-o',
        default=os.path.join(base_dir, 'filtered.csv'),
        help='Percorso al CSV di output per righe mantenute'
    )
    parser.add_argument(
        '--removed-file', '-r',
        default=os.path.join(base_dir, 'removed.csv'),
        help='Percorso al CSV di output per righe rimosse'
    )
    parser.add_argument(
        '--delimiter', '-d',
        default=';',
        help="Delimitatore di campo (default: ';')"
    )
    parser.add_argument(
        '--column-d-index', '-c',
        type=int,
        default=3,
        help='Indice zero-based della colonna D (default: 3)'
    )
    parser.add_argument(
        '--img-pattern', '-p',
        default='<img src',
        help='Pattern immagine da filtrare (default: "<img src")'
    )
    args = parser.parse_args()

    if args.input_file:
        input_path = args.input_file
        if not os.path.isfile(input_path):
            print(f"File '{input_path}' non trovato.", file=sys.stderr)
            sys.exit(1)
    else:
        input_path = find_csv_file(base_dir)

    filter_csv(
        input_path,
        args.output_file,
        args.removed_file,
        delimiter=args.delimiter,
        column_d_index=args.column_d_index,
        img_pattern=args.img_pattern
    )


if __name__ == '__main__':
    main()
