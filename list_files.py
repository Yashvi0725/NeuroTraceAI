
import os

folder = r"C:\NeuroTraceAIFinal"

print("Files in NeuroTraceAIFinal:\n")

for name in os.listdir(folder):
    path = os.path.join(folder, name)

    if os.path.isfile(path):
        print(name, "-", os.path.getsize(path), "bytes")
    else:
        print(name, "[folder]")
