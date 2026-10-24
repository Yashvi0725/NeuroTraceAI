
import os
import synapseclient
import synapseutils

BASE_FOLDER = r"C:\NeuroTraceAIFinal\audio"

# Folder IDs in the same order you provided them

folders = {
    "phonationE": "syn53175169",
    "phonationI": "syn53175354",
    "phonationO": "syn53175548",
    "phonationU": "syn53175731",
    "rhythmPA": "syn53176264",
    "rhythmTA": "syn53176762",
    "rhythmKA": "syn53175914"
}


syn = synapseclient.Synapse()
syn.login()

print("\nSTEP 1: Verifying all folder IDs...\n")

verified_folders = {}

# Verify every folder before downloading anything
for expected_name, folder_id in folders.items():
    entity = syn.get(folder_id)

    print("Expected:", expected_name)
    print("Found:", entity.name)
    print("ID:", entity.id)
    print("Type:", entity.concreteType)
    print("-" * 35)

    if entity.name != expected_name:
        raise ValueError(
            f"Folder mismatch! Expected {expected_name}, "
            f"but found {entity.name}. No downloads started."
        )

    if entity.concreteType != "org.sagebionetworks.repo.model.Folder":
        raise ValueError(
            f"{folder_id} is not a folder. No downloads started."
        )

    verified_folders[expected_name] = folder_id

print("\nAll seven folders verified!")
print("\nSTEP 2: Downloading recordings...\n")

for folder_name, folder_id in verified_folders.items():
    destination = os.path.join(BASE_FOLDER, folder_name)
    os.makedirs(destination, exist_ok=True)

    print(f"\nDownloading {folder_name}...")

    files = synapseutils.syncFromSynapse(
        syn,
        entity=folder_id,
        path=destination
    )

    print(f"Entries downloaded/synced: {len(files)}")
    print(f"Location: {destination}")

print("\nAll seven folder downloads completed!")
