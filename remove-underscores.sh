#!/bin/bash

# Loop through each file in the current directory
for file in *ipynb; do
    # Check if the file name contains an underscore
    if [[ $file == *"_"* ]]; then
        # Replace underscores with spaces
        mv "$file" "${file//_/ }"
    fi
done

