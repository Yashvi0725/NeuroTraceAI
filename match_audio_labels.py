
import os
import pandas as pd

EXCEL_FILE = r"C:\Users\ypath\Downloads\VOC-ALS.xlsx"
AUDIO_FOLDER = r"C:\NeuroTraceAIFinal\audio"
OUTPUT_FOLDER = r"C:\NeuroTraceAIFinal"

folders = [
    "phonationA",
    "phonationE",
    "phonationI",
    "phonationO",
    "phonationU",
    "rhythmPA",
    "rhythmTA",
    "rhythmKA"
]

# Read participant labels from Excel
df = pd.read_excel(
    EXCEL_FILE,
    sheet_name="VOC-ALS_Data",
    header=1
)

labels = df[["ID", "Category"]].copy()
labels["ID"] = labels["ID"].astype(str).str.strip()
labels["Category"] = labels["Category"].astype(str).str.strip().str.upper()

# Check the label data
if labels["ID"].duplicated().any():
    raise ValueError("Duplicate participant IDs found in Excel.")

if labels["ID"].isna().any() or (labels["ID"] == "").any():
    raise ValueError("Missing participant IDs found in Excel.")

if labels["Category"].isna().any() or (labels["Category"] == "").any():
    raise ValueError("Missing diagnosis labels found in Excel.")

if not set(labels["Category"].unique()).issubset({"ALS", "HC"}):
    raise ValueError("Unexpected diagnosis labels found.")

# Save one verified label per participant
labels.to_csv(
    os.path.join(OUTPUT_FOLDER, "participant_labels.csv"),
    index=False
)

# Build a list of all audio recordings
audio_rows = []

for folder in folders:
    folder_path = os.path.join(AUDIO_FOLDER, folder)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".wav"):
            participant_id = filename.split("_")[0]

            audio_rows.append({
                "ParticipantID": participant_id,
                "Task": folder,
                "Filename": filename,
                "AudioPath": os.path.join(folder_path, filename)
            })

audio_df = pd.DataFrame(audio_rows)

# Match audio participants with Excel labels
audio_df = audio_df.merge(
    labels,
    left_on="ParticipantID",
    right_on="ID",
    how="left",
    validate="many_to_one"
)

# Check for unmatched recordings
unmatched = audio_df[audio_df["Category"].isna()]

print("Participants in Excel:", len(labels))
print("Unique audio participants:", audio_df["ParticipantID"].nunique())
print("Total audio recordings:", len(audio_df))
print("Matched recordings:", audio_df["Category"].notna().sum())
print("Unmatched recordings:", len(unmatched))

print("\nDiagnosis counts in Excel:")
print(labels["Category"].value_counts())

print("\nMatched recording counts by diagnosis:")
print(audio_df["Category"].value_counts(dropna=False))

print("\nRecording counts by task:")
print(audio_df["Task"].value_counts())

if len(unmatched) > 0:
    print("\nUnmatched participant IDs:")
    print(sorted(unmatched["ParticipantID"].unique()))
    raise ValueError("Some audio recordings do not have verified labels.")

# Check each participant has exactly eight recordings
recording_counts = audio_df.groupby("ParticipantID")["Task"].nunique()

if not (recording_counts == 8).all():
    print("\nParticipants without all eight tasks:")
    print(recording_counts[recording_counts != 8])
    raise ValueError("Some participants are missing audio tasks.")

# Save the complete matched audio inventory
output_file = os.path.join(
    OUTPUT_FOLDER,
    "audio_manifest_labeled.csv"
)

audio_df.to_csv(output_file, index=False)

print("\nSUCCESS! All recordings have verified labels.")
print("Saved participant_labels.csv")
print("Saved audio_manifest_labeled.csv")
