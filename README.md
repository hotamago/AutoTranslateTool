# Simple File Translator

A simple command-line tool for translating entire files using Google Translate. Just specify input file, output file, source language, and target language.

## Features

- **Simple file-to-file translation** using Google Translate
- **Auto-detect source language** or specify manually
- **100+ supported languages**
- **UTF-8 encoding support** for international text
- **Command-line interface** for easy automation

## Installation

Install the required dependency:

```bash
pip install googletrans==4.0.0rc1
```

Or install from requirements.txt:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Translate file (auto-detect source, translate to Spanish)
python main.py input.txt output.txt

# Specify source and target languages
python main.py input.txt output.txt --from en --to es

# Auto-detect source, translate to French
python main.py input.txt output.txt --to fr

# English to German
python main.py input.txt output.txt --from en --to de
```

### List Supported Languages

```bash
# Show all supported language codes
python main.py --list-languages
```

### Common Language Codes

| Code | Language    | Code | Language    | Code | Language   |
|------|-------------|------|-------------|------|------------|
| auto | Auto-detect | en   | English     | es   | Spanish    |
| fr   | French      | de   | German      | it   | Italian    |
| pt   | Portuguese  | ru   | Russian     | ja   | Japanese   |
| ko   | Korean      | zh   | Chinese     | ar   | Arabic     |

## Examples

### Example 1: English to Spanish
```bash
echo "Hello world! How are you today?" > sample.txt
python main.py sample.txt translated.txt --from en --to es
cat translated.txt
# Output: ¡Hola mundo! ¿Cómo estás hoy?
```

### Example 2: Auto-detect to French
```bash
echo "Good morning! Have a nice day." > sample.txt
python main.py sample.txt translated.txt --to fr
cat translated.txt
# Output: Bonjour! Passe une bonne journée.
```

### Example 3: Japanese to English
```bash
echo "こんにちは世界" > japanese.txt
python main.py japanese.txt english.txt --from ja --to en
cat english.txt
# Output: Hello world
```

## Command Line Options

```
python main.py input_file output_file [options]

Arguments:
  input_file                Input file to translate
  output_file               Output file for translation

Options:
  --from SOURCE_LANG       Source language code (default: auto)
  --to TARGET_LANG         Target language code (default: es)
  --list-languages         List all supported language codes
  --help                   Show help message
```

## Features in Detail

### File Translation
- Reads entire file content and translates it as one unit
- Preserves text formatting and structure
- Handles large files efficiently
- UTF-8 encoding support for international characters

### Language Detection
- **Auto-detect**: Automatically detects source language
- **Manual**: Specify exact source language for better accuracy
- **100+ languages**: Supports all Google Translate languages

### Error Handling
- Validates input file exists
- Checks language codes are valid
- Handles network errors gracefully
- Creates output directories if needed

## Requirements

- Python 3.6+
- googletrans==4.0.0rc1
- Internet connection for Google Translate API

## Automation Examples

### Batch Processing
```bash
# Translate multiple files
for file in *.txt; do
    python main.py "$file" "translated_$file" --from en --to es
done
```

### Different Languages
```bash
# Create multiple translations
python main.py document.txt document_spanish.txt --to es
python main.py document.txt document_french.txt --to fr
python main.py document.txt document_german.txt --to de
```

## License

This project is open source and available under the MIT License.
