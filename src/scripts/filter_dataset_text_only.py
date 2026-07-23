from src.utils.args import parse_args
from src.utils.process_config import load_config
from src.utils.db import DatabaseManager


def main():
    args = parse_args()
    config = load_config(args.config)

    dbm = DatabaseManager(config)

    request_count = len(list(dbm.requests.find()))
    text_count = len(list(dbm.requests.find({"modality": "text"})))

    if request_count != text_count:
        dbm.prune_to_modalities(keep_modalities=["text"], delete_files=False)
    else:
        print(f"Database already has {text_count} text-only requests; skipping DB prune.")

    dbm.compact_breps_to_referenced()
    dbm.cleanup_orphan_files()
    dbm.print_db_schema_counts()
    dbm.close_connection()


if __name__ == "__main__":
    main()
