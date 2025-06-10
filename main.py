import argparse
import sys
from pathlib import Path

try:
    from googletrans import Translator, LANGUAGES
    GOOGLETRANS_AVAILABLE = True
except ImportError:
    GOOGLETRANS_AVAILABLE = False
    print("Error: googletrans not installed.")
    print("Install with: pip install googletrans==4.0.0rc1")
    sys.exit(1)


class SimpleTranslator:
    def __init__(self, source_lang='auto', target_lang='es'):
        """Initialize the simple translator."""
        self.translator = Translator()
        self.source_lang = source_lang
        self.target_lang = target_lang
        
        print(f"Translator initialized:")
        print(f"  Source language: {source_lang} ({LANGUAGES.get(source_lang, 'Auto-detect')})")
        print(f"  Target language: {target_lang} ({LANGUAGES.get(target_lang, 'Unknown')})")
    
    def translate_text(self, text: str) -> str:
        """Translate a single text string."""
        try:
            result = self.translator.translate(text, src=self.source_lang, dest=self.target_lang)
            return result.text
        except Exception as e:
            print(f"Translation error: {e}")
            return text
    
    def translate_file(self, input_file: str, output_file: str):
        """Translate entire file content."""
        try:
            # Read input file
            input_path = Path(input_file)
            if not input_path.exists():
                print(f"Error: Input file '{input_file}' not found.")
                return False
            
            print(f"Reading from: {input_file}")
            with open(input_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                print("Warning: Input file is empty.")
                return False
            
            print(f"Translating content ({len(content)} characters)...")
            
            # Translate the content
            translated_content = self.translate_text(content)
            
            # Write output file
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(translated_content)
            
            print(f"Translation completed!")
            print(f"Output saved to: {output_file}")
            return True
            
        except Exception as e:
            print(f"Error translating file: {e}")
            return False
    
    def list_languages(self):
        """List all supported languages."""
        print("Supported languages:")
        print("-" * 60)
        for code, name in sorted(LANGUAGES.items()):
            print(f"{code:4} : {name}")


def main():
    parser = argparse.ArgumentParser(
        description="Simple File Translator using Google Translate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py input.txt output.txt
  python main.py input.txt output.txt --from en --to es
  python main.py input.txt output.txt --from auto --to fr
  python main.py --list-languages
        """
    )
    
    parser.add_argument("input_file", nargs='?', help="Input file to translate")
    parser.add_argument("output_file", nargs='?', help="Output file for translation")
    parser.add_argument("--from", dest="source_lang", default="auto", 
                       help="Source language code (default: auto)")
    parser.add_argument("--to", dest="target_lang", default="es",
                       help="Target language code (default: es)")
    parser.add_argument("--list-languages", action="store_true",
                       help="List all supported language codes")
    
    args = parser.parse_args()
    
    # List languages and exit
    if args.list_languages:
        translator = SimpleTranslator()
        translator.list_languages()
        return
    
    # Validate required arguments
    if not args.input_file or not args.output_file:
        print("Error: Both input_file and output_file are required.")
        print("Use --help for usage information.")
        sys.exit(1)
    
    # Validate language codes
    if args.source_lang != 'auto' and args.source_lang not in LANGUAGES:
        print(f"Error: Invalid source language code '{args.source_lang}'")
        print("Use --list-languages to see supported codes.")
        sys.exit(1)
    
    if args.target_lang not in LANGUAGES:
        print(f"Error: Invalid target language code '{args.target_lang}'")
        print("Use --list-languages to see supported codes.")
        sys.exit(1)
    
    # Create translator and translate file
    translator = SimpleTranslator(args.source_lang, args.target_lang)
    success = translator.translate_file(args.input_file, args.output_file)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
