#!/usr/bin/env python3
"""
Script to split large model files into smaller parts for GitHub release upload.
Each part will be compressed into a ZIP file for better compression.
"""

import os
import zipfile
from pathlib import Path

def split_file_into_parts(file_path, part_size_mb=20):
    """
    Split a file into parts of specified size (in MB).
    Each part is compressed into a ZIP file.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return []

    file_size = file_path.stat().st_size
    part_size_bytes = part_size_mb * 1024 * 1024  # Convert MB to bytes

    # Calculate number of parts needed
    num_parts = (file_size + part_size_bytes - 1) // part_size_bytes  # Ceiling division

    print(f"📊 Splitting {file_path.name} ({file_size / (1024*1024):.1f} MB) into {num_parts} parts of ~{part_size_mb} MB each")

    part_files = []

    with open(file_path, 'rb') as f:
        for part_num in range(1, num_parts + 1):
            # Read part_size_bytes from file
            part_data = f.read(part_size_bytes)

            if not part_data:
                break

            # Create part filename
            part_filename = f"{file_path.name}.part{part_num:02d}.zip"
            part_path = file_path.parent / part_filename

            # Compress the part into a ZIP file
            with zipfile.ZipFile(part_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Store the part data with a simple name inside the ZIP
                zipf.writestr(f"{file_path.name}.part{part_num:02d}", part_data)

            actual_size = part_path.stat().st_size
            print(f"  ✅ Created {part_filename} ({actual_size / (1024*1024):.1f} MB)")

            part_files.append(part_path)

    return part_files

def main():
    """Split both coarse and line model files."""

    # Define model files to split
    models_to_split = [
        ("new_model/vul/checkpoint-best-f1/model.safetensors", "coarse"),
        ("new_model/line_vul/checkpoint-best-f1/model_2048.bin", "line")
    ]

    all_parts = []

    for model_path, model_type in models_to_split:
        model_file = Path(model_path)
        if model_file.exists():
            print(f"\n🔄 Splitting {model_type} model: {model_path}")
            parts = split_file_into_parts(model_file, part_size_mb=20)
            all_parts.extend(parts)
            print(f"✅ {model_type.capitalize()} model split into {len(parts)} parts")
        else:
            print(f"⚠️ {model_type.capitalize()} model not found: {model_path}")

    print("\n📋 Summary:")
    print(f"  Total parts created: {len(all_parts)}")
    for part in all_parts:
        size_mb = part.stat().st_size / (1024 * 1024)
        print(f"    {part.name}: {size_mb:.1f} MB")

    print("\n🎯 Next steps:")
    print("  1. Create GitHub release v1.0.0 in repository HackLoading/CVD-project-VII")
    print("  2. Upload all .zip part files to the release")
    print("  3. Test the download functionality in the app")

if __name__ == "__main__":
    main()