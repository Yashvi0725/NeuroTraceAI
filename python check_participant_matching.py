
import pandas as pd
from pathlib import Path

BASE = Path(r"C:\NeuroTraceAIFinal")

audio = pd.read_csv(BASE / "audio_acoustic_features.csv")

clinical = pd.read_excel(
    BASE / "VOC-ALS.xlsx",
    sheet_name="VOC-ALS_Data",
    header=1
)

audio_ids = set(audio["ParticipantID"].astype(str).str.strip())
clinical_ids = set(clinical["ID"].astype(str).str.strip())

print("Unique audio participants:", len(audio_ids))
print("Unique spreadsheet participants:", len(clinical_ids))
print("Participants in both:", len(audio_ids & clinical_ids))
print("Audio-only IDs:", sorted(audio_ids - clinical_ids))
print("Spreadsheet-only IDs:", sorted(clinical_ids - audio_ids))

audio_labels = audio.groupby("ParticipantID")["Category"].nunique()
print("Participants with conflicting audio labels:",
      int((audio_labels > 1).sum()))
