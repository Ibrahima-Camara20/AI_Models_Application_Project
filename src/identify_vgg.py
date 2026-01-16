import os
import pandas as pd
from deepface import DeepFace
MODEL_NAME = "VGG-Face"

def recognize_celebrities(input_faces_dir, db_path):
    try:
        # Vérification des dossiers
        if not os.path.exists(input_faces_dir):
            raise FileNotFoundError(f"Le dossier {input_faces_dir} n'existe pas. Lancez l'étape 1.2.")
        
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Le dossier {db_path} n'existe pas. Créez-le et ajoutez des photos de célébrités.")

        print(f"[INFO] Chargement du modèle {MODEL_NAME} et début de la reconnaissance...")

        # Liste des visages à identifier
        try:
            unknown_faces = [f for f in os.listdir(input_faces_dir) if f.endswith(('.jpg', '.png'))]
        except Exception as e:
            raise RuntimeError(f"Erreur lors de la lecture du dossier {input_faces_dir}: {e}")
        
        if len(unknown_faces) == 0:
            print(f"[ATTENTION] Aucune image trouvée dans {input_faces_dir}")
            return
        
        print(f"[INFO] {len(unknown_faces)} visage(s) à traiter")
        results = []
        error_count = 0

        for face_file in unknown_faces:
            face_path = os.path.join(input_faces_dir, face_file)
            
            try:
                # Vérification que le fichier existe et est lisible
                if not os.path.isfile(face_path):
                    print(f"[SKIP] {face_file} n'est pas un fichier valide")
                    continue
                
                # DeepFace.find compare l'image avec toutes celles dans db_path
                try:
                    dfs = DeepFace.find(
                        img_path = face_path, 
                        db_path = db_path, 
                        model_name = MODEL_NAME, 
                        enforce_detection = False, # RetinaFace a déjà fait la détection
                        silent = True
                    )
                except Exception as e:
                    print(f"[ERREUR] Échec DeepFace.find pour {face_file}: {e}")
                    results.append((face_file, "Error"))
                    error_count += 1
                    continue
                
                # DeepFace renvoie une liste de DataFrames
                if len(dfs) > 0 and not dfs[0].empty:
                    try:
                        df = dfs[0]
                        # La première ligne est la correspondance la plus proche
                        best_match_path = df.iloc[0]['identity']
                        
                        # Extraction du nom depuis le fichier trouvé
                        recognized_name = os.path.basename(best_match_path).split('.')[0]
                        
                        print(f"[SUCCÈS] {face_file} est identifié comme : {recognized_name}")
                        results.append((face_file, recognized_name))
                    except Exception as e:
                        print(f"[ERREUR] Échec de traitement du résultat pour {face_file}: {e}")
                        results.append((face_file, "Error"))
                        error_count += 1
                else:
                    print(f"[INCONNU] Aucune correspondance trouvée pour {face_file}")
                    results.append((face_file, "Unknown"))

            except Exception as e:
                print(f"[ERREUR] Problème inattendu sur {face_file} : {e}")
                results.append((face_file, "Error"))
                error_count += 1

        # --- Sauvegarde des résultats ---
        if len(results) > 0:
            try:
                df_results = pd.DataFrame(results, columns=["Fichier_Source", "Prediction_VGG"])
                output_csv = "resultats_reconnaissance.csv"
                df_results.to_csv(output_csv, index=False)
                print(f"\n[INFO] Résultats sauvegardés dans {output_csv}")
                print(f"[INFO] {len(results)} visages traités")
                if error_count > 0:
                    print(f"[ATTENTION] {error_count} erreur(s) rencontrée(s) pendant la reconnaissance.")
            except Exception as e:
                raise RuntimeError(f"Échec de sauvegarde des résultats CSV: {e}")
        else:
            print("[ATTENTION] Aucun résultat à sauvegarder")
            
    except Exception as e:
        print(f"[ERREUR CRITIQUE] Échec de la reconnaissance: {e}")
        raise