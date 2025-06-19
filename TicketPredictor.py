import os
import numpy as np
import pandas as pd
from datetime import datetime
from Load_Tickets import TicketEmbedder
from hdbscan import approximate_predict
import joblib

class TicketPredictor:
    """
    Classe per caricare il modello HDBSCAN allenato, prevedere nuovi ticket
    e salvare etichette micro-cluster.
    """
    def __init__(self,
                 features_npz_path: str,
                 tickets_csv_name: str,
                 output_dir: str = "output_cluster"):
        # Percorsi
        self.features_npz_path = features_npz_path
        self.tickets_csv_name = tickets_csv_name
        self.output_dir = output_dir

        # Verifica modello HDBSCAN
        self.model_path = os.path.join(self.output_dir, "hdbscan_model.joblib")
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f"Modello non trovato in '{self.model_path}'. Esegui prima MainClustering per salvarlo.")

        # Istanzio embedder e carico il clusterer
        self.embedder = TicketEmbedder()
        self.clusterer = joblib.load(self.model_path)

        # Carico metadata ticket
        csv_path = os.path.join(os.getcwd(), "output_chunks", self.tickets_csv_name)
        df = pd.read_csv(csv_path, sep=';', encoding='latin-1')

        # Verifica colonna SGSEGNALID
        if 'SGSEGNALID' not in df.columns:
            raise KeyError("Colonna 'SGSEGNALID' non trovata nei metadata dei ticket.")

        # Uso direttamente le colonne SGSEGNALID, SGTITLE, SGDESCRI, SGCREATEAT
        self.df_tickets = df

        # Carico embeddings pre-calcolati e allineo con i metadata
        data = np.load(self.features_npz_path)
        X_full = data['X']
        m = min(len(X_full), len(self.df_tickets))
        self.X = X_full[:m]
        self.df_tickets = self.df_tickets.iloc[:m].reset_index(drop=True)

        # File di output
        self.out_csv = os.path.join(self.output_dir, "tickets_with_micro_labels.csv")

    def predict_ticket(self,
                       segnalid: str,
                       title: str,
                       description: str,
                       strength_threshold: float = 0.2):
        """
        Predice label_micro per un nuovo ticket.
        """
        # Timestamp di creazione
        created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

        # Generazione embedding
        try:
            _, vec = self.embedder.embed(segnalid, title, description, created_at)
        except Exception as e:
            print(f"[TicketPredictor] Errore durante embed(): {e}")
            return None

        emb = vec.reshape(1, -1)
        labels_pred, strengths = approximate_predict(self.clusterer, emb)
        label = int(labels_pred[0])
        strength = float(strengths[0])
        if label != -1 and strength < strength_threshold:
            label = -1

        # Lettura o creazione CSV di output
        if os.path.isfile(self.out_csv):
            df_out = pd.read_csv(self.out_csv, sep=';')
            if 'SGSEGNALID' not in df_out.columns:
                raise KeyError("Il CSV di output non contiene la colonna 'SGSEGNALID'.")
        else:
            cols = ['SGSEGNALID', 'SGTITLE', 'SGDESCRI', 'SGCREATEAT', 'label_micro']
            df_out = pd.DataFrame(columns=cols)

        # Nuova riga
        new_row = {
            'SGSEGNALID': segnalid,
            'SGTITLE': title,
            'SGDESCRI': description,
            'SGCREATEAT': created_at,
            'label_micro': label
        }
        df_out = pd.concat([df_out, pd.DataFrame([new_row])], ignore_index=True)
        df_out.to_csv(self.out_csv, sep=';', index=False)

        return label

    def interactive_loop(self):
        """
        Loop CLI per inserire nuovi ticket.
        """
        existing = set(self.df_tickets['SGSEGNALID'].astype(str))
        if os.path.isfile(self.out_csv):
            df_prev = pd.read_csv(self.out_csv, sep=';')
            existing |= set(df_prev['SGSEGNALID'].astype(str))

        while True:
            ans = input("Nuovo ticket? (s/n): ").strip().lower()
            if ans != 's':
                print("Terminato.")
                break

            tid = ''
            while not tid or tid in existing:
                tid = input("SGSEGNALID: ").strip()
                if tid in existing:
                    print("ID già esistente, riprova.")
            title = input("SGTITLE: ").strip()
            description = input("SGDESCRI: ").strip()

            label = self.predict_ticket(tid, title, description)
            if label is None:
                continue

            if label == -1:
                print(f"Ticket {tid}: RUMORE (label_micro={label})\n")
            else:
                print(f"Ticket {tid}: cluster_micro={label}\n")

if __name__ == "__main__":
    predictor = TicketPredictor(
        features_npz_path="features.npz",
        tickets_csv_name="fattoAmano.csv",
        output_dir="output_cluster"
    )
    predictor.interactive_loop()
