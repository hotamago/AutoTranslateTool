uv run main.py -s ja -t vi -i "_temp/ManualTransFile.json" -o "_temp/ManualTransFile_out.json" -m google -c 100 --cache_file "_temp\ManualTransFile_out.json.cache" --ignore_regex "^[0-9]+$"

uv run main.py -s ja -t vi -i "_temp/ManualTransFile.json" -o "_temp/ManualTransFile_out.json" -m cerebras -c 4 -b 50 --cache_file "_temp\ManualTransFile_out.json.cache" --ignore_regex "^[0-9]+$" --api_key_cerebras ""

uv run main.py -s ja -t vi -i "_temp/ManualTransFile.json" -o "_temp/ManualTransFile_out.json" -m nvidia -c 4 -b 50 --cache_file "_temp\ManualTransFile_out.json.cache" --ignore_regex "^[0-9]+$" --api_key_nvidia ""

uv run main.py -s ja -t vi -i "_temp/ManualTransFile.json" -o "_temp/ManualTransFile_out.json" -m lmstudio -c 1 -b 20 --cache_file "_temp\ManualTransFile_out.json.cache" --ignore_regex "^[0-9]+$"

uv run main.py -s ja -t vi -i "_temp/ManualTransFile.json" -o "_temp/ManualTransFile_out.json" -m google,cerebras,nvidia -c "20,2,8" -b "1,50,50" --cache_file "_temp\ManualTransFile_out.json.cache" --ignore_regex "^[0-9]+$" --api_key_cerebras "" --api_key_nvidia ""

uv run remove_error_failback.py _temp\ManualTransFile_out.json.cache