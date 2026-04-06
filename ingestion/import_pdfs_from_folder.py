import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from import_pdf import import_pdf_to_supabase, PDFS_DIR


def get_pdf_files(folder_path: Path):
    if not folder_path.exists():
        raise FileNotFoundError(f"Dossier introuvable : {folder_path}")

    pdf_files = sorted(folder_path.rglob("*.pdf"))
    return pdf_files


def main():
    pdf_files = get_pdf_files(PDFS_DIR)

    print(f"Nombre de PDF trouvés dans le dossier : {len(pdf_files)}")

    inserted_count = 0
    already_exists_count = 0
    empty_count = 0
    error_count = 0

    for index, pdf_path in enumerate(pdf_files, start=1):
        print("-" * 60)
        print(f"[{index}/{len(pdf_files)}] Traitement de : {pdf_path.name}")

        try:
            result = import_pdf_to_supabase(pdf_path, category="bulk_pdf_import")
            print("Résultat :", result)

            status = result.get("status")

            if status == "inserted":
                inserted_count += 1
            elif status == "already_exists":
                already_exists_count += 1
            elif status == "empty":
                empty_count += 1
            else:
                error_count += 1

        except Exception as e:
            error_count += 1
            print("Erreur :", str(e))

    print("\n" + "=" * 60)
    print("Import PDF terminé")
    print(f"Nouveaux PDF importés : {inserted_count}")
    print(f"PDF déjà existants   : {already_exists_count}")
    print(f"PDF sans texte       : {empty_count}")
    print(f"Erreurs              : {error_count}")


if __name__ == "__main__":
    main()