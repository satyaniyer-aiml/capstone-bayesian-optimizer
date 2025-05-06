"""
Backup important project folders (data, models, logs)
into a timestamped ZIP file under 'backups/'.
"""

import os
import shutil
import zipfile
from datetime import datetime

# === Settings ===
backup_root = "backups"
folders_to_backup = ["data", "models", "logs"]

# === Create backup directory if not exists ===
os.makedirs(backup_root, exist_ok=True)

# === Create timestamped ZIP file name ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
zip_filename = os.path.join(backup_root, f"backup_{timestamp}.zip")

# === Create a ZIP file and add folders ===
with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as backup_zip:
    for folder in folders_to_backup:
        if os.path.exists(folder):
            for root, dirs, files in os.walk(folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, start=os.getcwd())
                    backup_zip.write(file_path, arcname)
            print(f"? Added '{folder}' folder to backup ZIP.")
        else:
            print(f"?? Warning: '{folder}' folder not found. Skipped.")

print(f"\n? Full backup ZIP created at: {zip_filename}")
