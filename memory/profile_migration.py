"""Back up and remove legacy user-profile documents for one exact user."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import chromadb


def _client() -> chromadb.ClientAPI:
    return chromadb.HttpClient(
        host=os.getenv("CHROMA_HOST", "localhost"),
        port=int(os.getenv("CHROMA_PORT", "8011")),
        settings=chromadb.Settings(anonymized_telemetry=False),
    )


def inspect_user(user_id: str) -> Dict[str, Any]:
    collection = _client().get_or_create_collection("user_profile")
    result = collection.get(
        where={"user_id": user_id},
        include=["documents", "metadatas"],
    )
    return {
        "user_id": user_id,
        "count": len(result.get("ids") or []),
        "ids": result.get("ids") or [],
        "documents": result.get("documents") or [],
        "metadatas": result.get("metadatas") or [],
    }


def apply_migration(user_id: str, backup_dir: Path) -> Dict[str, Any]:
    snapshot = inspect_user(user_id)
    ids = list(snapshot["ids"])
    if not ids:
        return {
            "user_id": user_id,
            "deleted": 0,
            "backup_file": "",
            "status": "already_clean",
        }
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = backup_dir / f"user_profile_backup_{user_id}_{stamp}.json"
    backup_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    verified = json.loads(backup_path.read_text(encoding="utf-8"))
    if len(verified.get("ids") or []) != len(ids):
        raise RuntimeError(
            "backup verification failed; legacy profiles were not deleted"
        )
    collection = _client().get_or_create_collection("user_profile")
    collection.delete(ids=ids)
    remaining = inspect_user(user_id)["count"]
    if remaining:
        raise RuntimeError(f"profile cleanup incomplete: {remaining} documents remain")
    return {
        "user_id": user_id,
        "deleted": len(ids),
        "backup_file": str(backup_path),
        "status": "applied",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate legacy user profiles")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--backup-dir",
        default="data/migrations",
    )
    args = parser.parse_args()
    if args.dry_run == args.apply:
        parser.error("choose exactly one of --dry-run or --apply")
    result = (
        inspect_user(args.user_id)
        if args.dry_run
        else apply_migration(args.user_id, Path(args.backup_dir))
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
