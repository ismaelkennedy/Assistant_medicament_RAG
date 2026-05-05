import os
import json
import re
import pandas as pd
import numpy as np
import faiss
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
DATA_PATH = "data/CIS_RCP.csv"
INDEX_DIR = "index"
INDEX_PATH = os.path.join(INDEX_DIR, "medicaments.index")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.json")
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Les médicaments qu'on veut indexer
MEDICAMENTS_CIBLES = [
    "doliprane",
    "dafalgan",
    "efferalgan",
    "ibuprofene",
    "nurofen",
    "aspirine",
    "aspegic",
    "amoxicilline",
    "augmentin",
    "smecta",
    "imodium",
    "ventoline",
    "omeprazole",
    "metformine",
    "glucophage",
]

# Les sections de la notice qu'on veut garder (les plus utiles pour les questions)
SECTIONS_UTILES = {
    "RcpDenomination": "Dénomination",
    "RcpIndicTherap": "Indications thérapeutiques",
    "RcpPosoAdmin": "Posologie et administration",
    "RcpContreIndic": "Contre-indications",
    "RcpMisesEnGarde": "Mises en garde",
    "RcpInteractions": "Interactions médicamenteuses",
    "RcpEffetsIndesirables": "Effets indésirables",
    "RcpGrossAllait": "Grossesse et allaitement",
    "RcpSurdosage": "Surdosage",
}


# --- Bloc 2 : Chargement et filtrage des données ---

def reparer_encodage(texte) -> str:
    # Certaines cellules peuvent être vides (NaN)
    if not isinstance(texte, str):
        return ""
    # Le HTML dans le CSV est du UTF-8 stocké dans un fichier Latin-1
    # On re-encode en Latin-1 puis on décode en UTF-8 pour retrouver les vrais caractères
    try:
        return texte.encode("latin-1", errors="ignore").decode("utf-8", errors="replace")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return texte


def charger_donnees(chemin: str) -> pd.DataFrame:
    print("Chargement du fichier CSV...")
    df = pd.read_csv(chemin, sep="\t", encoding="latin-1")
    df["RCP_html"] = df["RCP_html"].apply(reparer_encodage)
    print(f"  {len(df)} notices chargées au total")
    return df


def extraire_denomination(html: str) -> str:
    # On cherche la position de la section dénomination puis on prend
    # les 600 caractères suivants — la classe CSS varie selon les notices
    if not isinstance(html, str):
        return ""
    idx = html.lower().find("rcpdenomination")
    if idx < 0:
        return ""
    extrait = html[idx: idx + 600]
    texte = re.sub(r"<[^>]+>", " ", extrait)
    return re.sub(r"\s+", " ", texte).strip().lower()


def filtrer_medicaments(df: pd.DataFrame, noms: list) -> pd.DataFrame:
    print("Filtrage des médicaments cibles...")

    # On extrait la dénomination une seule fois pour toutes les notices
    denominations = df["RCP_html"].apply(extraire_denomination)

    lignes_selectionnees = []
    for nom in noms:
        pattern = r"\b" + re.escape(nom) + r"\b"
        masque = denominations.str.contains(pattern, regex=True, na=False)
        correspondances = df[masque]
        if len(correspondances) == 0:
            print(f"  ATTENTION : '{nom}' introuvable dans la base")
            continue
        lignes_selectionnees.append(correspondances.iloc[0])
        print(f"  '{nom}' : notice trouvée (Code CIS: {correspondances.iloc[0]['Code_CIS']})")

    resultat = pd.DataFrame(lignes_selectionnees).reset_index(drop=True)
    print(f"  {len(resultat)} notices sélectionnées au total")
    return resultat


# --- Bloc 3 : Extraction du texte depuis le HTML ---

def extraire_nom_medicament(soup: BeautifulSoup) -> str:
    from bs4 import Tag
    ancre = soup.find("a", {"name": "RcpDenomination"})
    if not ancre:
        return "Inconnu"
    for element in ancre.parent.next_siblings:
        if not isinstance(element, Tag):
            continue
        # On s'arrête seulement sur les vraies sections (name="Rcp...")
        # Les ancres _Toc... sont des signets internes, pas des sections
        ancre_section = element.find("a", attrs={"name": lambda n: n and n.startswith("Rcp")})
        if ancre_section:
            break
        texte = element.get_text(separator=" ", strip=True)
        if texte:
            return texte
    return "Inconnu"


def extraire_section(soup: BeautifulSoup, nom_ancre: str) -> str:
    balise = soup.find("a", {"name": nom_ancre})
    if not balise:
        return ""

    textes = []
    # On parcourt les éléments HTML qui suivent cette ancre
    for element in balise.parent.next_siblings:
        # On s'arrête quand on tombe sur la prochaine section
        if element.name and element.find("a", href="#HautDePage"):
            break
        if hasattr(element, "get_text"):
            texte = element.get_text(separator=" ", strip=True)
            if texte:
                textes.append(texte)

    return " ".join(textes)


def html_vers_sections(html: str, code_cis: int) -> list:
    soup = BeautifulSoup(html, "html.parser")
    nom = extraire_nom_medicament(soup)
    sections = []

    for ancre, titre_section in SECTIONS_UTILES.items():
        texte = extraire_section(soup, ancre)
        if not texte or len(texte) < 20:
            continue

        # Nettoyage : suppression des espaces multiples
        texte = re.sub(r"\s+", " ", texte).strip()

        sections.append({
            "medicament": nom,
            "code_cis": code_cis,
            "section": titre_section,
            "texte": texte,
        })

    return sections


def extraire_toutes_sections(df: pd.DataFrame) -> list:
    print("Extraction du texte des notices...")
    toutes_sections = []

    for _, ligne in df.iterrows():
        sections = html_vers_sections(ligne["RCP_html"], ligne["Code_CIS"])
        toutes_sections.extend(sections)

    print(f"  {len(toutes_sections)} sections extraites")
    return toutes_sections


# --- Bloc 4 : Chunking ---

def chunker(texte: str, taille_max: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    # Si le texte est court, pas besoin de découper
    if len(texte) <= taille_max:
        return [texte]

    chunks = []
    debut = 0

    while debut < len(texte):
        fin = debut + taille_max

        # Si on n'est pas à la fin, on recule jusqu'au dernier espace
        # pour ne pas couper un mot en deux
        if fin < len(texte):
            dernier_espace = texte.rfind(" ", debut, fin)
            if dernier_espace > debut:
                fin = dernier_espace

        chunks.append(texte[debut:fin].strip())
        # On avance en tenant compte de l'overlap
        debut = fin - overlap

    return chunks


def creer_documents(sections: list) -> list:
    print("Découpage en chunks...")
    documents = []

    for i, section in enumerate(sections):
        chunks = chunker(section["texte"])

        for j, chunk in enumerate(chunks):
            # On préfixe le contenu avec le nom du médicament et la section
            # pour que l'embedding capture ces informations clés
            contenu_enrichi = f"{section['medicament']} — {section['section']} : {chunk}"
            documents.append({
                "id": f"doc_{i:04d}_chunk_{j:02d}",
                "contenu": contenu_enrichi,
                "metadata": {
                    "medicament": section["medicament"],
                    "code_cis": section["code_cis"],
                    "section": section["section"],
                    "chunk_index": j,
                    "total_chunks": len(chunks),
                }
            })

    print(f"  {len(documents)} chunks créés")
    return documents


# --- Bloc 5 : Embeddings et index FAISS ---

def creer_embeddings(documents: list, modele: SentenceTransformer) -> np.ndarray:
    print("Création des embeddings...")
    textes = [doc["contenu"] for doc in documents]

    # show_progress_bar=True affiche une barre de progression dans le terminal
    vecteurs = modele.encode(textes, show_progress_bar=True, convert_to_numpy=True)
    print(f"  Vecteurs créés : {vecteurs.shape[0]} vecteurs de dimension {vecteurs.shape[1]}")
    return vecteurs


def creer_index_faiss(vecteurs: np.ndarray) -> faiss.Index:
    dimension = vecteurs.shape[1]

    # IndexFlatL2 = recherche exacte par distance euclidienne
    index = faiss.IndexFlatL2(dimension)

    # FAISS exige des float32
    index.add(vecteurs.astype(np.float32))
    print(f"  Index FAISS créé avec {index.ntotal} vecteurs")
    return index


def sauvegarder(index: faiss.Index, documents: list):
    os.makedirs(INDEX_DIR, exist_ok=True)

    # Sauvegarde de l'index FAISS (les vecteurs)
    faiss.write_index(index, INDEX_PATH)

    # Sauvegarde des chunks et métadonnées (le texte)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    print(f"  Index sauvegardé dans '{INDEX_PATH}'")
    print(f"  Chunks sauvegardés dans '{CHUNKS_PATH}'")




# --- Bloc 6 : Main ---

def main():
    print("=" * 50)
    print("  PHASE 1 : INDEXATION")
    print("=" * 50)

    # Étape 1 : charger et filtrer les données
    df = charger_donnees(DATA_PATH)
    df_filtre = filtrer_medicaments(df, MEDICAMENTS_CIBLES)

    # Étape 2 : extraire le texte des notices HTML
    sections = extraire_toutes_sections(df_filtre)

    # Étape 3 : découper en chunks
    documents = creer_documents(sections)

    # Étape 4 : charger le modèle d'embedding
    print(f"Chargement du modèle d'embedding '{EMBEDDING_MODEL}'...")
    modele = SentenceTransformer(EMBEDDING_MODEL)

    # Étape 5 : créer les vecteurs
    vecteurs = creer_embeddings(documents, modele)

    # Étape 6 : créer et sauvegarder l'index FAISS
    index = creer_index_faiss(vecteurs)
    sauvegarder(index, documents)

    print("=" * 50)
    print("  Indexation terminée avec succès !")
    print(f"  {len(documents)} chunks indexés pour {len(df_filtre)} médicaments")
    print("=" * 50)


if __name__ == "__main__":
    main()
