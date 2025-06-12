import unicodedata
import re
from typing import List

try:
    from janome.tokenizer import Tokenizer
    JANOME_AVAILABLE = True
except ImportError:
    print("Warning: Janome not installed. Japanese tokenization will be limited.")
    JANOME_AVAILABLE = False

class JapaneseTextProcessor:
    """A utility class for Japanese text processing."""
    
    def __init__(self):
        self.tokenizer = Tokenizer() if JANOME_AVAILABLE else None
        # Common Japanese stop words (can be expanded)
        self.stop_words = {
            'の', 'に', 'は', 'を', 'た', 'が', 'で', 'て', 'と', 'し', 'れ', 'さ',
            'ある', 'いる', 'も', 'する', 'から', 'な', 'こと', 'として', 'い', 'や',
            'れる', 'など', 'なっ', 'ない', 'この', 'ため', 'その', 'あっ', 'よう',
            'また', 'もの', 'という', 'あり', 'まで', 'られ', 'なる', 'へ', 'か',
            'だ', 'これ', 'によって', 'により', 'おり', 'より', 'による', 'ず', 'なり',
            'られる', 'において', 'ば', 'なかっ', 'なく', 'しかし', 'について', 'せ', 'だっ',
            'その後', 'できる', 'それ', 'う', 'ので', 'なお', 'のみ', 'でき', 'き',
            'つ', 'における', 'および', 'いう', 'さらに', 'でも', 'ら', 'たり', 'その他',
            'に関する', 'たち', 'ます', 'ん', 'なら', 'に対して', '特に', 'せる', '及び',
            'これら', 'とき', 'では', 'にて', 'ほか', 'ながら', 'うち', 'そして', 'とも',
            'ただし', 'かつて', 'それぞれ', 'または', 'お', 'ほど', 'ものの', 'に対する',
            'ほとんど', 'と共に', 'といった', 'です', 'ました', 'ません'
        }
    
    def is_japanese(self, text: str) -> bool:
        """Checks if the text contains Japanese characters."""
        for char in text:
            name = unicodedata.name(char, '')
            if 'CJK' in name or 'HIRAGANA' in name or 'KATAKANA' in name:
                return True
        return False
    
    def tokenize(self, text: str, remove_stop_words: bool = True) -> List[str]:
        """Tokenizes Japanese text."""
        if not self.tokenizer or not self.is_japanese(text):
            # Fallback to space-splitting for non-Japanese text
            return text.split()
        
        tokens = []
        for token in self.tokenizer.tokenize(text):
            # Extract nouns, verbs, adjectives (can be customized)
            if token.part_of_speech.split(',')[0] in ['名詞', '動詞', '形容詞', '形容動詞']:
                base_form = token.base_form if token.base_form != '*' else token.surface
                
                if remove_stop_words and base_form in self.stop_words:
                    continue
                    
                tokens.append(base_form)
        
        return tokens
    
    def normalize_text(self, text: str) -> str:
        """Normalizes text (e.g., full-width to half-width)."""
        # NFKC normalization (converts full-width chars to half-width)
        text = unicodedata.normalize('NFKC', text)
        # Replace multiple whitespaces with a single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
