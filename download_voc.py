
import os
import synapseclient
import synapseutils

# Destination for phonationA recordings
destination = r"C:\NeuroTraceAIFinal\audio\phonationA"
os.makedirs(destination, exist_ok=True)

# Connect to Synapse
syn = synapseclient.Synapse()
syn.login()

# Verify the folder before downloading
folder_id = "syn53174987"
entity = syn.get(folder_id)

print("Folder:", entity.name)
print("ID:", entity.id)
print("Type:", entity.concreteType)

if entity.concreteType != "org.sagebionetworks.repo.model.Folder":
    raise ValueError("The selected entity is not a folder.")

print("\nStarting download...")

files = synapseutils.syncFromSynapse(
    syn,
    entity=folder_id,
    path=destination
)

print("\nDownload completed.")
print("Downloaded entries:", len(files))
print("Saved to:", destination)
