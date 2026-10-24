
import synapseclient

syn = synapseclient.Synapse()
syn.login()

ids = [
    "syn53175914",
    "syn53176264",
    "syn53176762"
]

for folder_id in ids:
    entity = syn.get(folder_id)
    print(folder_id, "->", entity.name, "|", entity.concreteType)
