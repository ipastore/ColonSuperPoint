#!/bin/bash
#Assuming pwd: ./DAtaset_II
# ls : # Seq_01, Seq_02, Seq_03, ..., Seq_10
# This script extracts frames from .mov files in Seq_* directories at 3 FPS

# Loop through all Seq_* folders
for seq_dir in Seq_*; do
    if [[ -d "$seq_dir" ]]; then
        mov_file="$seq_dir/$seq_dir.mov"
        out_dir="$seq_dir/img_at_3_fps"
        mkdir -p "$out_dir"

        # Check if the .mov file exists
        if [[ -f "$mov_file" ]]; then
            echo "Processing $mov_file at 3 FPS..."

            # Extract frames using ffmpeg
            ffmpeg -i "$mov_file" -vf "fps=3" "$out_dir/out_%04d.png"

            echo "Finished processing $mov_file."
        else
            echo "⚠️ Missing file: $mov_file"
        fi
    fi
done

# Final summary
echo "✅ Frame extraction completed."
