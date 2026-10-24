
import csv
from pathlib import Path

PROJECT = Path(__file__).resolve().parent

JPS_FILE = PROJECT / "JPS2024_ALS_dataset.csv"
MINSK_FILE = (
    PROJECT / "Minsk2020"
    / "Minsk2020_ALS_database-main"
    / "Minsk2020_ALS_dataset.csv"
)


def load_csv(file_path):
    with open(file_path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        columns = reader.fieldnames or []
    return columns, rows


def normalize(text):
    return "".join(c.lower() for c in text if c.isalnum())


def find_column(columns, possible_names):
    for column in columns:
        if normalize(column) in possible_names:
            return column
    return None


def inspect_dataset(name, file_path):
    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

    if not file_path.exists():
        print("FILE NOT FOUND:", file_path)
        return None

    columns, rows = load_csv(file_path)

    print("File:", file_path)
    print("Total rows:", len(rows))
    print("Total columns:", len(columns))
    print("\nColumn names:")
    for column in columns:
        print("-", column)

    id_column = find_column(
        columns,
        {"id", "participantid", "subjectid", "patientid", "participant"}
    )

    diagnosis_column = find_column(
        columns,
        {"diagnosis", "diagnosisals", "diagnosislabel",
         "label", "class", "group", "status", "condition"}
    )

    print("\nID column:", id_column or "Not detected")
    print("Diagnosis column:", diagnosis_column or "Not detected")

    ids = []
    if id_column:
        ids = [
            row[id_column].strip()
            for row in rows
            if row.get(id_column) and row[id_column].strip()
        ]

        unique_ids = set(ids)
        duplicates = sorted({
            value for value in ids if ids.count(value) > 1
        })

        print("Rows with IDs:", len(ids))
        print("Unique IDs:", len(unique_ids))
        print("Duplicate IDs:", duplicates or "None")

    if diagnosis_column:
        counts = {}
        for row in rows:
            value = (row.get(diagnosis_column) or "").strip()
            counts[value or "(blank)"] = counts.get(
                value or "(blank)", 0
            ) + 1
        print("Diagnosis counts:", counts)

    return {
        "columns": columns,
        "rows": rows,
        "id_column": id_column,
        "diagnosis_column": diagnosis_column,
        "ids": set(ids)
    }


jps = inspect_dataset("JPS2024 DATASET", JPS_FILE)
minsk = inspect_dataset("MINSK2020 DATASET", MINSK_FILE)

print("\n" + "=" * 55)
print("COMPARISON RESULTS")
print("=" * 55)

if jps and minsk:
    jps_ids = jps["ids"]
    minsk_ids = minsk["ids"]

    print("JPS unique IDs:", len(jps_ids))
    print("Minsk unique IDs:", len(minsk_ids))
    print("Shared IDs:", len(jps_ids & minsk_ids))
    print("JPS-only IDs:", len(jps_ids - minsk_ids))
    print("Minsk-only IDs:", len(minsk_ids - jps_ids))

    common_columns = set(jps["columns"]) & set(minsk["columns"])
    print("\nCommon columns:", sorted(common_columns))

    if jps["id_column"] and minsk["id_column"]:
        jps_id_col = jps["id_column"]
        minsk_id_col = minsk["id_column"]

        jps_diagnosis = jps["diagnosis_column"]
        minsk_diagnosis = minsk["diagnosis_column"]

        if jps_diagnosis and minsk_diagnosis:
            print("\nDiagnosis values for shared IDs:")
            jps_map = {
                row[jps_id_col].strip(): row[jps_diagnosis]
                for row in jps["rows"]
                if row.get(jps_id_col, "").strip()
            }
            minsk_map = {
                row[minsk_id_col].strip(): row[minsk_diagnosis]
                for row in minsk["rows"]
                if row.get(minsk_id_col, "").strip()
            }

            for participant in sorted(jps_ids & minsk_ids):
                print(
                    participant,
                    "| JPS:", jps_map.get(participant),
                    "| Minsk:", minsk_map.get(participant)
                )
        else:
            print(
                "\nDiagnosis comparison unavailable: "
                "a diagnosis column was not detected in both files."
            )
else:
    print(
        "\nCheck the file paths above. "
        "Both datasets must load before comparison."
    )

print("\nInspection complete. No CSV files were modified.")
