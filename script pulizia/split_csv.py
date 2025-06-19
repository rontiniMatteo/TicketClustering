#!/usr/bin/env python3

import csv
import sys
import argparse
import os
import glob

# Aumenta il limite per campi di grandi dimensioni
txt_limit = 10**8
csv.field_size_limit(txt_limit)  # 100 milioni di caratteri


def find_csv_file(directory):
    """
    Cerca un file CSV nella directory specificata. Se ne trova uno solo lo restituisce, altrimenti solleva errore.
    """
    csv_files = glob.glob(os.path.join(directory, '*.csv'))
    if not csv_files:
        print(f"Nessun file CSV trovato in {directory}.", file=sys.stderr)
        sys.exit(1)
    if len(csv_files) > 1:
        print(f"Trovati più CSV in {directory}: {', '.join(os.path.basename(f) for f in csv_files)}. Specificane uno chiaramente.", file=sys.stderr)
        sys.exit(1)
    return csv_files[0]


def split_csv(input_file, output_dir, chunk_size=10000, delimiter=';'):
    """
    Divide un file CSV in più file più piccoli, ognuno contenente fino a `chunk_size` righe di dati.
    Include solo le colonne B, C, D e P (indici 1,2,3,15).

    Args:
        input_file (str): Percorso al CSV di input.
        output_dir (str): Cartella dove salvare i file splittati.
        chunk_size (int): Numero massimo di righe per file (escludendo l'intestazione).
        delimiter (str): Carattere delimitatore dei campi (default: ';').
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        header = next(reader)

        # Seleziona gli indici delle colonne B, C, D, P
        selected_indices = [1, 2, 3, 15]
        selected_header = [header[i] for i in selected_indices]

        file_count = 1
        buffer = []

        for row in reader:
            # Estrai solo le colonne desiderate
            selected_row = [row[i] for i in selected_indices]
            buffer.append(selected_row)

            if len(buffer) >= chunk_size:
                chunk_path = os.path.join(output_dir, f"part_{file_count:04d}.csv")
                with open(chunk_path, 'w', newline='', encoding='utf-8') as outcsv:
                    writer = csv.writer(outcsv, delimiter=delimiter)
                    writer.writerow(selected_header)
                    writer.writerows(buffer)
                print(f"Creato {os.path.basename(chunk_path)} con {len(buffer)} record.")
                file_count += 1
                buffer = []

        # Scrivi eventuali righe rimanenti
        if buffer:
            chunk_path = os.path.join(output_dir, f"part_{file_count:04d}.csv")
            with open(chunk_path, 'w', newline='', encoding='utf-8') as outcsv:
                writer = csv.writer(outcsv, delimiter=delimiter)
                writer.writerow(selected_header)
                writer.writerows(buffer)
            print(f"Creato {os.path.basename(chunk_path)} con {len(buffer)} record.")


def main():
    # Directory dello script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="Suddivide un file CSV in più file di dimensione specificata e seleziona solo alcune colonne."
    )
    parser.add_argument(
        'input_file',
        nargs='?',  # opzionale
        help='Nome del file CSV nella stessa cartella dello script (default: cerca automaticamente un CSV).'
    )
    parser.add_argument(
        '-o', '--output-dir',
        default=os.path.join(base_dir, 'output_chunks'),
        help='Directory per salvare i file splittati (default: ./output_chunks).'
    )
    parser.add_argument(
        '-c', '--chunk-size',
        type=int,
        default=10000,
        help='Numero massimo di righe dati per file di output (default: 10000).'
    )
    parser.add_argument(
        '-d', '--delimiter',
        default=';',
        help="Delimitatore dei campi CSV (default: ';')."
    )
    args = parser.parse_args()

    # Determina il file CSV di input
    if args.input_file:
        input_path = os.path.join(base_dir, args.input_file)
        if not os.path.isfile(input_path):
            print(f"File {args.input_file} non trovato in {base_dir}.", file=sys.stderr)
            sys.exit(1)
    else:
        input_path = find_csv_file(base_dir)

    split_csv(input_path, args.output_dir, args.chunk_size, args.delimiter)


if __name__ == '__main__':
    main()
