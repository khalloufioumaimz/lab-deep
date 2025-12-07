# lab_script.py - Version pour débutants

# ============ PARTIE 1 : IMPORT DES BIBLIOTHÈQUES ============
print("🔧 Importation des bibliothèques...")
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

print("✅ Bibliothèques importées avec succès!")

# ============ PARTIE 2 : CONFIGURATION SIMPLE ============
print("\n⚙️ Configuration du lab...")

# Choix du modèle (commencez avec 'small' c'est un bon compromis)
MODEL_CHOICE = "openai/whisper-small"  # Essayez aussi: "openai/whisper-tiny" pour plus rapide

# Chemin vers vos fichiers
AUDIO_FILES = ["audio/test1.wav", "audio/test2.wav", "audio/test3.wav"]
REF_FILES = ["references/test1.txt", "references/test2.txt", "references/test3.txt"]

print(f"Modèle sélectionné: {MODEL_CHOICE}")
print(f"Nombre de fichiers: {len(AUDIO_FILES)}")

# ============ PARTIE 3 : CHARGER UN MODÈLE WHISPER SIMPLE ============
print("\n🤖 Chargement du modèle Whisper...")

# Cette méthode est plus simple pour débuter
transcriber = pipeline(
    "automatic-speech-recognition",
    model=MODEL_CHOICE,
    device="cpu"  # Mettez "cuda:0" si vous avez une carte graphique NVIDIA
)

print("✅ Modèle Whisper chargé!")

# ============ PARTIE 4 : FONCTION POUR LIRE LES FICHIERS ============
def lire_fichier_audio(chemin):
    """Lit un fichier audio et le prépare pour Whisper"""
    print(f"🎵 Lecture de: {chemin}")
    
    # Charger l'audio
    audio, sr = librosa.load(chemin, sr=16000)
    print(f"   Durée: {len(audio)/sr:.2f} secondes")
    print(f"   Fréquence d'échantillonnage: {sr} Hz")
    
    return audio, sr

def lire_texte_reference(chemin):
    """Lit le texte de référence"""
    with open(chemin, 'r', encoding='utf-8') as f:
        texte = f.read().strip()
    print(f"📖 Texte de référence lu ({len(texte)} caractères)")
    return texte

# ============ PARTIE 5 : SEGMENTATION MANUELLE (pour débuter) ============
def segmenter_audio_simple(audio, sr, duree_segment=10):
    """
    Découpe l'audio en segments de durée fixe
    (Plus simple que VAD pour commencer)
    """
    print("✂️ Découpage de l'audio...")
    
    # Calculer la taille d'un segment en échantillons
    segment_samples = int(duree_segment * sr)
    
    segments = []
    n_segments = len(audio) // segment_samples + 1
    
    for i in range(n_segments):
        debut = i * segment_samples
        fin = min((i + 1) * segment_samples, len(audio))
        
        if fin - debut > sr * 0.5:  # Ignorer les segments trop courts (<0.5s)
            segment_audio = audio[debut:fin]
            segments.append({
                'id': i,
                'audio': segment_audio,
                'debut_temps': debut / sr,
                'fin_temps': fin / sr,
                'duree': (fin - debut) / sr
            })
    
    print(f"   {len(segments)} segments créés")
    return segments

# ============ PARTIE 6 : TRANSCRIPTION ============
def transcrire_audio(chemin_audio):
    """Transcrit un fichier audio complet"""
    print(f"\n🎤 Transcription de {chemin_audio}...")
    
    # Transcription simple (tout l'audio d'un coup)
    resultat = transcriber(chemin_audio)
    transcription = resultat['text']
    
    print(f"📝 Transcription obtenue:")
    print(f"   '{transcription[:100]}...'" if len(transcription) > 100 else f"   '{transcription}'")
    
    return transcription

# ============ PARTIE 7 : CALCUL DU WER (simplifié) ============
def calculer_erreurs(transcription, reference):
    """Calcule le pourcentage d'erreurs simplement"""
    # Conversion en minuscules et suppression de la ponctuation
    import re
    
    def nettoyer_texte(texte):
        texte = texte.lower()
        texte = re.sub(r'[^\w\s]', '', texte)  # Enlève ponctuation
        texte = re.sub(r'\s+', ' ', texte)     # Espaces multiples -> simple
        return texte.strip()
    
    trans_clean = nettoyer_texte(transcription)
    ref_clean = nettoyer_texte(reference)
    
    # Séparer en mots
    mots_trans = trans_clean.split()
    mots_ref = ref_clean.split()
    
    # Calcul simple (approximatif)
    n_mots_ref = len(mots_ref)
    
    if n_mots_ref == 0:
        return {"erreur": 100.0, "details": "Référence vide"}
    
    # Pour débuter, on fait une comparaison simple
    # Note: C'est une simplification, pas le vrai WER
    mots_corrects = sum(1 for i in range(min(len(mots_trans), len(mots_ref))) 
                       if mots_trans[i] == mots_ref[i])
    
    pourcentage_erreur = (1 - mots_corrects/n_mots_ref) * 100
    
    return {
        "erreur_approximative": pourcentage_erreur,
        "mots_reference": n_mots_ref,
        "mots_transcription": len(mots_trans),
        "mots_corrects": mots_corrects
    }

# ============ PARTIE 8 : VISUALISATION ============
def afficher_spectrogramme(audio, sr, titre):
    """Affiche un spectrogramme simple"""
    print(f"\n📊 Création du spectrogramme pour {titre}...")
    
    plt.figure(figsize=(12, 4))
    
    # Créer le spectrogramme
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogramme: {titre}')
    plt.xlabel('Temps (s)')
    plt.ylabel('Fréquence (Hz)')
    
    # Sauvegarder l'image
    plt.savefig(f'spectrogramme_{titre}.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Spectrogramme sauvegardé!")

# ============ PARTIE 9 : EXÉCUTION PRINCIPALE ============
def main():
    """Fonction principale qui exécute tout le lab"""
    print("\n" + "="*50)
    print("        LAB STT - DÉBUT DE L'EXPÉRIENCE")
    print("="*50)
    
    resultats = []
    
    # Pour chaque fichier audio
    for i, (audio_file, ref_file) in enumerate(zip(AUDIO_FILES, REF_FILES)):
        print(f"\n{'='*40}")
        print(f"EXPÉRIENCE {i+1}: {audio_file}")
        print(f"{'='*40}")
        
        try:
            # 1. Lire les fichiers
            audio, sr = lire_fichier_audio(audio_file)
            reference = lire_texte_reference(ref_file)
            
            # 2. Afficher le spectrogramme
            afficher_spectrogramme(audio, sr, f"test{i+1}")
            
            # 3. Transcrire
            transcription = transcrire_audio(audio_file)
            
            # 4. Calculer les erreurs
            erreurs = calculer_erreurs(transcription, reference)
            
            # 5. Afficher les résultats
            print(f"\n📊 RÉSULTATS pour test{i+1}:")
            print(f"   Taux d'erreur approximatif: {erreurs['erreur_approximative']:.2f}%")
            print(f"   Mots dans la référence: {erreurs['mots_reference']}")
            print(f"   Mots dans la transcription: {erreurs['mots_transcription']}")
            print(f"   Mots corrects: {erreurs['mots_corrects']}")
            
            # Sauvegarder les résultats
            resultats.append({
                'fichier': audio_file,
                'erreur': erreurs['erreur_approximative'],
                'transcription': transcription,
                'reference': reference
            })
            
        except Exception as e:
            print(f"❌ Erreur avec {audio_file}: {e}")
    
    # ============ RÉSUMÉ FINAL ============
    print("\n" + "="*50)
    print("            RÉSUMÉ DES RÉSULTATS")
    print("="*50)
    
    for i, res in enumerate(resultats):
        print(f"\nTest {i+1}:")
        print(f"  Fichier: {res['fichier']}")
        print(f"  Erreur: {res['erreur']:.2f}%")
    
    # Moyenne des erreurs
    if resultats:
        moyenne = sum(r['erreur'] for r in resultats) / len(resultats)
        print(f"\n📈 MOYENNE GLOBALE: {moyenne:.2f}% d'erreur")
    
    print("\n✅ Lab terminé avec succès!")

# ============ LANCER LE LAB ============
if __name__ == "__main__":
    main()