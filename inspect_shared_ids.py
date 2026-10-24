
import csv
from pathlib import Path
from collections import defaultdict

PROJECT = Path(__file__).resolve().parent

JPS_FILE = PROJECT / "JPS2024_ALS_dataset.csv"
MINSK_FILE = (
    PROJECT / "Minsk2020"
    / "Minsk2020_ALS_database-main"
    / "Minsk2020_ALS_dataset.csv"
)


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        columns = reader.fieldnames or []
    return columns, rows


def group_by_id(rows):
    groups = defaultdict(list)

    for row in rows:
        participant_id = (row.get("ID") or "").strip()
        if participant_id:
            groups[participant_id].append(row)

    return groups


jps_columns, jps_rows = read_csv(JPS_FILE)
minsk_columns, minsk_rows = read_csv(MINSK_FILE)

jps_groups = group_by_id(jps_rows)
minsk_groups = group_by_id(minsk_rows)

print("=" * 60)
print("1. COLUMN DIFFERENCES")
print("=" * 60)

print("Only in JPS2024:")
print(sorted(set(jps_columns) - set(minsk_columns)))

print("\nOnly in Minsk2020:")
print(sorted(set(minsk_columns) - set(jps_columns)))

print("\n" + "=" * 60)
print("2. DUPLICATE PARTICIPANT IDS")
print("=" * 60)

for name, groups in [
    ("JPS2024", jps_groups),
    ("Minsk2020", minsk_groups)
]:
    print("\n" + name)

    duplicates = {
        participant_id: records
        for participant_id, records in groups.items()
        if len(records) > 1
    }

    if not duplicates:
        print("No repeated IDs.")
        continue

    for participant_id, records in sorted(duplicates.items()):
        print(
            "ID:", participant_id,
            "| Number of rows:", len(records)
        )

        for index, row in enumerate(records, start=1):
            diagnosis = row.get("Diagnosis (ALS)", "Not available")
            print(
                "  Row", index,
                "| Diagnosis:", diagnosis,
                "| Age:", row.get("Age"),
                "| Sex:", row.get("Sex"),
                "| J1_a:", row.get("J1_a"),
                "| HNR_a:", row.get("HNR_a")
            )

print("\n" + "=" * 60)
print("3. SHARED IDS AND FEATURE COMPARISON")
print("=" * 60)

shared_ids = sorted(set(jps_groups) & set(minsk_groups))
common_features = (
    set(jps_columns) & set(minsk_columns)
) - {"ID", "Diagnosis (ALS)"}

print("Shared ID values:", len(shared_ids))
print("Shared feature columns:", len(common_features))

if not shared_ids:
    print("No shared IDs found.")
else:
    print("\nID | JPS rows | Minsk rows | Matching feature pairs")

    for participant_id in shared_ids:
        jps_records = jps_groups[participant_id]
        minsk_records = minsk_groups[participant_id]

        matching = 0
        total = 0

        for jps_row in jps_records:
            for minsk_row in minsk_records:
                total += len(common_features)

                for feature in common_features:
                    jps_value = (jps_row.get(feature) or "").strip()
                    minsk_value = (minsk_row.get(feature) or "").strip()

                    if jps_value == minsk_value:
                        matching += 1

        print(
            participant_id,
            "|", len(jps_records),
            "|", len(minsk_records),
            "|", matching, "/", total
        )

print("\n" + "=" * 60)
print("4. MINSK DIAGNOSIS BY SHARED ID")
print("=" * 60)

for participant_id in shared_ids:
    records = minsk_groups[participant_id]
    diagnoses = [
        (row.get("Diagnosis (ALS)") or "").strip()
        for row in records
    ]

    print(
        "ID:", participant_id,
        "| Diagnoses:", diagnoses
    )

print("\nInspection complete. No files were modified.")
