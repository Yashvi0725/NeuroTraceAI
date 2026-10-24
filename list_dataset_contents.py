
import os

folder = r"C:\NeuroTraceAIFinal\datasets"

for root, dirs, files in os.walk(folder):
    print("\nFolder:", root)

    for name in files:
        path = os.path.join(root, name)
        print("  ", name, "-", os.path.getsize(path), "bytes")

    if not dirs and not files:
        print("   (empty folder)")
