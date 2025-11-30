uv run main.py -s ja -t vi -i "_temp/ManualTransFile.json" -o "_temp/ManualTransFile_out.json" -m google -c 200 -b 100 --cache_file "_temp\ManualTransFile_out.json.cache" --ignore_regex "^[0-9]+$"

uv run main.py -s ja -t vi -i "_temp/ManualTransFile.json" -o "_temp/ManualTransFile_out.json" -m cerebras -c 4 -b 500 --cache_file "_temp\ManualTransFile_out.json.cache" --ignore_regex "^[0-9]+$" -k 