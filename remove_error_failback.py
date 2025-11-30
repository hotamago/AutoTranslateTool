#!/usr/bin/env python3
"""
Simple script to find and remove lines with key = value from a cache file.
"""

import json
import sys
from pathlib import Path


def find_error_failback_lines(file_path):
    """Find all lines that have key = value."""
    error_lines = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    # Check if any key in the JSON object equals value
                    for key, value in data.items():
                        if key == value:
                            error_lines.append(line_num)
                            break
                except json.JSONDecodeError:
                    # Skip invalid JSON lines
                    continue
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    return error_lines


def remove_lines(file_path, line_numbers):
    """Remove specified line numbers from file."""
    # Read all lines
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Create a set for faster lookup
    lines_to_remove = set(line_numbers)
    
    # Filter out lines to remove (line_numbers are 1-indexed, list is 0-indexed)
    filtered_lines = [
        line for idx, line in enumerate(lines, start=1)
        if idx not in lines_to_remove
    ]
    
    # Write back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(filtered_lines)


def main():
    # Get file path from user
    if len(sys.argv) > 1:
        cache_path = sys.argv[1]
    else:
        cache_path = input("Enter the path to the cache file: ").strip()
    
    # Validate path
    cache_file = Path(cache_path)
    if not cache_file.exists():
        print(f"Error: File '{cache_path}' does not exist.")
        sys.exit(1)
    
    if not cache_file.is_file():
        print(f"Error: '{cache_path}' is not a file.")
        sys.exit(1)
    
    print(f"Scanning file: {cache_path}")
    print("Looking for lines with key = value...")
    
    # Find error lines
    error_lines = find_error_failback_lines(cache_path)
    
    # Report results
    count = len(error_lines)
    if count == 0:
        print("\nNo lines found with key = value.")
        return
    
    print(f"\nFound {count} line(s) with key = value:")
    if count <= 20:
        print(f"Line numbers: {', '.join(map(str, error_lines))}")
    else:
        print(f"Line numbers: {', '.join(map(str, error_lines[:20]))} ... and {count - 20} more")
    
    # Ask user for confirmation
    response = input(f"\nDo you want to remove all {count} line(s)? (yes/no): ").strip().lower()
    
    if response in ('yes', 'y'):
        print("Removing lines...")
        remove_lines(cache_path, error_lines)
        print(f"Successfully removed {count} line(s) from '{cache_path}'.")
    else:
        print("Operation cancelled. No lines were removed.")


if __name__ == "__main__":
    main()

