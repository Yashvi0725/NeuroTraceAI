
import pandas as pd

minsk = pd.read_csv("Minsk2020_ALS_dataset.csv")
jps = pd.read_csv("JPS2024_ALS_dataset.csv")

for df in [minsk, jps]:
    df["ID"] = df["ID"].astype(str).str.strip()

for participant_id in ["24", "28", "42"]:
    print("\n" + "=" * 50)
    print("PARTICIPANT ID:", participant_id)

    print("\nMINSK2020:")
    print(minsk[minsk["ID"] == participant_id][
        ["ID", "Diagnosis (ALS)"]
    ].to_string(index=False))

    print("\nJPS2024:")
    print(jps[jps["ID"] == participant_id][
        ["ID"]
    ].to_string(index=False))
