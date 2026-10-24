
import os
import pandas as pd
import soundfile as sf

AUDIO_FOLDER = r"C:\NeuroTraceAIFinal\audio"

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

total_files = 0
valid_files = 0
failed_files = []

for folder in folders:
    folder_path = os.path.join(AUDIO_FOLDER, folder)
    files = [
        file for file in os.listdir(folder_path)
        if file.lower().endswith(".wav")
    ]

    folder_valid = 0

    for file in files:
        file_path = os.path.join(folder_path, file)

        try:
            audio, sample_rate = sf.read(file_path)

            if len(audio) > 0 and sample_rate > 0:
                valid_files += 1
                folder_valid += 1
            else:
                failed_files.append(file_path)

        except Exception as error:
            failed_files.append(file_path)
            print("Error:", file_path, error)

    total_files += len(files)
    print(folder, "- Files:", len(files), "| Valid:", folder_valid)

print("\nTOTAL WAV FILES:", total_files)
print("VALID WAV FILES:", valid_files)
print("FAILED WAV FILES:", len(failed_files))

if len(failed_files) == 0:
    print("All audio files passed the basic validation!")
else:
    print("Please check the failed files listed above.")
