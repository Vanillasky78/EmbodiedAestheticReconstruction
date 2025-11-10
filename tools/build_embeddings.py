# tools/portrait_csv_to_jsonl.py
import csv, json, argparse, os, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = ["id","artist_en","title_en","year","notes_pose","file_name","embedding_path"]
        miss = [c for c in required if c not in reader.fieldnames]
        if miss:
            print(f"[FATAL] CSV missing columns: {miss}")
            sys.exit(1)

        for r in reader:
            for k,v in r.items():
                if isinstance(v, str):
                    r[k] = v.strip()            # 去除首尾空格/换行
            if not r["file_name"]:
                raise ValueError(f"row id={r.get('id')} empty file_name")

            rows.append({
                "id": int(r["id"]) if r["id"] else None,
                "artist_en": r["artist_en"],
                "title_en": r["title_en"],
                "year": r["year"],
                "notes_pose": r["notes_pose"],
                "file_name": r["file_name"],           # 关键字段
                "embedding_path": r["embedding_path"],
            })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] wrote {len(rows)} rows -> {args.out}")

if __name__ == "__main__":
    main()
