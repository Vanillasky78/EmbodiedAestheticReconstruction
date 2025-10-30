# tools/portrait_csv_to_jsonl.py
# Convert portrait_works.csv → portrait_art_dataset.jsonl for index building

import csv, json, os, argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/portrait_works.csv")
    parser.add_argument("--out", default="data/interim/portrait_art_dataset.jsonl")
    parser.add_argument("--images_dir", default="data/images")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.csv, newline="", encoding="utf-8") as f, open(args.out, "w", encoding="utf-8") as out:
        reader = csv.DictReader(f)
        count = 0

        for row in reader:
            # 兼容你的列名结构
            artist = row.get("artist_en", "").strip()
            title = row.get("title_en", "").strip()
            notes = row.get("notes_pose", "").strip()
            period = row.get("period", "").strip()
            composition = row.get("composition", "").strip()
            image_path = os.path.join(args.images_dir, os.path.basename(row.get("image", "")))
            embedding_path = row.get("embedding_path", "").strip()

            record = {
                "artwork_id": str(count + 1),
                "artist_name_en": artist,
                "artwork_title_en": title,
                "notes_pose": notes,
                "period": period,
                "composition": composition,
                "museum": "Curated Portrait Dataset",
                "license": "CC0",
                "isPublicDomain": True,
                "image_path": image_path,
                "embedding_path": embedding_path
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"[OK] Wrote {count} records to {args.out}")

if __name__ == "__main__":
    main()
