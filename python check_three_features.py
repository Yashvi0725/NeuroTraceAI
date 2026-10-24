
import pandas as pd

voc = pd.read_excel(
    "VOC-ALS.xlsx",
    sheet_name="VOC-ALS_Data",
    header=1
)

minsk = pd.read_csv("Minsk2020_ALS_clean_for_training.csv")

# Candidate mappings to investigate
mappings = [
    ("HNR", voc, "HNR_A", minsk, "HNR_a"),
    ("Local jitter", voc, "localJitter_A", minsk, "J1_a"),
    ("Local shimmer", voc, "localShimmer_A", minsk, "S1_a")
]

for name, vdf, vc, mdf, mc in mappings:
    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

    for dataset, df, col in [
        ("VOC-ALS", vdf, vc),
        ("Minsk2020", mdf, mc)
    ]:
        values = pd.to_numeric(df[col], errors="coerce")

        print(f"\n{dataset} - {col}")
        print("Valid values:", values.notna().sum())
        print("Missing:", values.isna().sum())
        print("Minimum:", values.min())
        print("Median:", values.median())
        print("Maximum:", values.max())
        print("Mean:", values.mean())

print("\nScale audit complete.")
print("Different scales may indicate different units or extraction methods.")
