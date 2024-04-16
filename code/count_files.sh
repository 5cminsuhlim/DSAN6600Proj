#!/bin/bash

## CAN BE USED TO COUNT THE NUMBER OF FILES IN A DIRECTORY IF YOU ARE CHECKING ON MULTIPROCESSING
# Ex: ./count_files.sh /path/to/directory
# Be sure to chmod +x count_files.sh

# Check if the directory path is provided as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory_path>"
    exit 1
fi

directory="$1"

# Check if the provided directory exists
if [ ! -d "$directory" ]; then
    echo "Error: $directory does not exist."
    exit 1
fi

# Count the number of files in the directory
file_count=$(ls -1 "$directory" | wc -l)

echo "Number of files in $directory: $file_count"
