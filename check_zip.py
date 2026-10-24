
import zipfile

path = r"C:\NeuroTraceAIFinal\datasets\raw\als\Mendeley-ALS\ALS Disease patient classification.zip"

with zipfile.ZipFile(path, "r") as z:
    print("Files inside the ZIP:")
    for name in z.namelist():
        print("-", name)
