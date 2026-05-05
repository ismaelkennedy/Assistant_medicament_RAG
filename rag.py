import os
import json
import faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
INDEX_PATH = "index/medicaments.index"
CHUNKS_PATH = "index/chunks.json"
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_K = 8          # nombre de chunks récupérés par recherche
SEUIL_SCORE = 5.5  # score L2 max au-delà duquel on considère le résultat non pertinent


# --- Bloc 1 : Chargement de l'index ---

def charger_index():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        print("ERREUR : Index introuvable. Lance d'abord 'python indexation.py'")
        exit(1)

    print("Chargement de l'index FAISS...")
    index = faiss.read_index(INDEX_PATH)

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        documents = json.load(f)

    print(f"  {index.ntotal} chunks chargés pour {len(set(d['metadata']['medicament'] for d in documents))} médicaments")
    return index, documents


# --- Bloc 2 : Recherche vectorielle ---

def rechercher(question: str, modele: SentenceTransformer, index: faiss.Index, documents: list) -> list:
    # On transforme la question en vecteur avec le même modèle qu'à l'indexation
    vecteur_question = modele.encode([question], convert_to_numpy=True).astype(np.float32)

    # FAISS retourne les distances (scores) et les indices des k chunks les plus proches
    scores, indices = index.search(vecteur_question, TOP_K)

    resultats = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        resultats.append({
            "contenu": documents[idx]["contenu"],
            "metadata": documents[idx]["metadata"],
            "score": float(score),
        })

    return resultats


# --- Bloc 3 : Génération de la réponse ---

def construire_prompt_systeme() -> str:
    return """Tu es un assistant d'information sur les médicaments. Tu réponds aux questions des utilisateurs en te basant UNIQUEMENT sur les extraits de notices officielles fournis dans le contexte.

Règles strictes :
- Si l'information n'est pas dans le contexte fourni, dis clairement "Je ne trouve pas cette information dans ma base de données."
- Ne jamais inventer ou supposer des informations médicales.
- Cite toujours le nom du médicament concerné dans ta réponse.
- Indique la section de la notice dont provient l'information (ex: "Selon la section Effets indésirables du Doliprane...").
- Termine TOUJOURS ta réponse par : "⚠️ Ces informations ne remplacent pas l'avis d'un professionnel de santé. En cas de doute, consultez votre médecin ou votre pharmacien."

Format de réponse : clair, structuré, en français."""


def construire_contexte(resultats: list) -> str:
    contexte = ""
    for i, r in enumerate(resultats, 1):
        meta = r["metadata"]
        contexte += f"\n--- Extrait {i} ---\n"
        contexte += f"Médicament : {meta['medicament']}\n"
        contexte += f"Section : {meta['section']}\n"
        contexte += f"Contenu : {r['contenu']}\n"
    return contexte


def generer_reponse(question: str, resultats: list, client: Groq) -> str:
    # Si aucun résultat pertinent, on répond directement sans appeler le LLM
    if not resultats or resultats[0]["score"] > SEUIL_SCORE:
        return (
            "Je ne trouve pas d'information pertinente sur ce sujet dans ma base de données.\n\n"
            "⚠️ Ces informations ne remplacent pas l'avis d'un professionnel de santé. "
            "En cas de doute, consultez votre médecin ou votre pharmacien."
        )

    contexte = construire_contexte(resultats)

    messages = [
        {"role": "system", "content": construire_prompt_systeme()},
        {"role": "user", "content": f"Contexte (extraits de notices officielles) :\n{contexte}\n\nQuestion : {question}"},
    ]

    reponse = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.2,  # faible température = réponses plus précises et moins créatives
    )

    return reponse.choices[0].message.content


# --- Bloc 4 : Interface interactive ---

def afficher_sources(resultats: list):
    print("\n  Sources :")
    medicaments_cites = set()
    for r in resultats:
        meta = r["metadata"]
        cle = f"{meta['medicament']} — {meta['section']}"
        if cle not in medicaments_cites:
            print(f"    • {cle}")
            medicaments_cites.add(cle)


def main():
    print("=" * 50)
    print("  Assistant Médicaments RAG")
    print("=" * 50)

    # Chargement unique au démarrage (pas à chaque question)
    index, documents = charger_index()

    print(f"Chargement du modèle d'embedding...")
    modele = SentenceTransformer(EMBEDDING_MODEL)

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    print("\nSystème prêt. Tapez 'quit' pour quitter.\n")

    while True:
        question = input("Votre question : ").strip()

        if question.lower() in ["quit", "exit", "q"]:
            print("Au revoir !")
            break

        if not question:
            continue

        # Étape 1 : recherche des chunks pertinents
        resultats = rechercher(question, modele, index, documents)

        # Étape 2 : génération de la réponse
        print("\nRecherche en cours...\n")
        reponse = generer_reponse(question, resultats, client)

        # Étape 3 : affichage
        print(f"Réponse :\n{reponse}")
        afficher_sources(resultats)
        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()
