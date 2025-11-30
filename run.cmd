uv run main.py -s ja -t vi -i "_temp/ManualTransFile.json" -o "_temp/ManualTransFile_out.json" -m nllb -b 8 --cache_file "_temp\ManualTransFile_out.json.cache" --ignore_regex "^[0-9]+$"

uv run main.py -s ja -t vi -i "_temp/ManualTransFile.json" -o "_temp/ManualTransFile_out.json" -m google -c 40 --cache_file "_temp\ManualTransFile_out.json.cache" --ignore_regex "^[0-9]+$"

uv run main.py -s ja -t vi -i "_temp/ManualTransFile.json" -o "_temp/ManualTransFile_out.json" -m opus-mt --model_url "Helsinki-NLP/opus-mt-ja-vi" -b 8 --cache_file "_temp\ManualTransFile_out.json.cache" --ignore_regex "^[0-9]+$"