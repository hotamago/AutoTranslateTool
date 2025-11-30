uv run main.py -s ja -t vi -i "_temp/ManualTransFile.json" -o "_temp/ManualTransFile_out.json" -m google -c 100 --cache_file "_temp\ManualTransFile_out.json.cache" --ignore_regex "^[0-9]+$"

uv run main.py -s ja -t vi -i "_temp/ManualTransFile.json" -o "_temp/ManualTransFile_out.json" -m cerebras -c 4 -b 200 --cache_file "_temp\ManualTransFile_out.json.cache" --ignore_regex "^[0-9]+$" -k 

uv run main.py -s ja -t vi -i "_temp/ManualTransFile.json" -o "_temp/ManualTransFile_out.json" -m lmstudio -c 1 -b 20 --cache_file "_temp\ManualTransFile_out.json.cache" --ignore_regex "^[0-9]+$"

uv run main.py -s ja -t vi -i "_temp/ManualTransFile.json" -o "_temp/ManualTransFile_out.json" -m google,cerebras -c 100,4 -b 1,200 --cache_file "_temp\ManualTransFile_out.json.cache" --ignore_regex "^[0-9]+$" -k 