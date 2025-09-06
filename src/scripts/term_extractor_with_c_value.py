#!/usr/bin/env python3
"""term_extractor_with_c_value.py
C値・NC値アルゴリズムを実装した専門用語抽出
------------------------------------------------
* C値（C-value）: 複合語の専門用語らしさを測る指標
* NC値（NC-value）: C値に文脈情報を加えた改良版
"""

from __future__ import annotations

import math
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
from sudachipy import tokenizer, dictionary
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CValueExtractor:
    """C値・NC値による専門用語抽出クラス"""
    
    def __init__(self):
        self.tokenizer_obj = dictionary.Dictionary().create()
        self.mode = tokenizer.Tokenizer.SplitMode.A
        
    def extract_noun_phrases(self, text: str) -> List[List[str]]:
        """テキストから名詞句を抽出"""
        morphemes = self.tokenizer_obj.tokenize(text, self.mode)
        noun_phrases = []
        current_phrase = []
        
        for morpheme in morphemes:
            pos = morpheme.part_of_speech()[0]
            
            if pos == '名詞':
                current_phrase.append(morpheme.normalized_form())
            else:
                if len(current_phrase) >= 2:  # 2語以上の名詞句のみ
                    noun_phrases.append(current_phrase[:])
                current_phrase = []
        
        # 最後のフレーズを追加
        if len(current_phrase) >= 2:
            noun_phrases.append(current_phrase)
            
        return noun_phrases
    
    def calculate_c_value(self, text: str, min_freq: int = 2) -> Dict[str, float]:
        """
        C値を計算
        C-value(a) = log2|a| × (freq(a) - (1/|Ta|) × Σb∈Ta freq(b))
        
        ここで:
        - |a|: 候補語aの長さ（単語数）
        - freq(a): 候補語aの出現頻度
        - Ta: aを含むより長い候補語の集合
        - freq(b): より長い候補語bの出現頻度
        """
        # 名詞句を抽出
        noun_phrases = self.extract_noun_phrases(text)
        
        # 候補語とその頻度を計算
        candidates = defaultdict(int)
        for phrase in noun_phrases:
            for length in range(2, len(phrase) + 1):
                for i in range(len(phrase) - length + 1):
                    candidate = ' '.join(phrase[i:i+length])
                    candidates[candidate] += 1
        
        # 最低頻度でフィルタリング
        candidates = {k: v for k, v in candidates.items() if v >= min_freq}
        
        # 各候補語に対してC値を計算
        c_values = {}
        
        for candidate in candidates:
            freq_a = candidates[candidate]
            words_a = candidate.split()
            length_a = len(words_a)
            
            # aを含むより長い候補語を探す
            longer_terms = []
            for other_candidate in candidates:
                if other_candidate != candidate and candidate in other_candidate:
                    longer_terms.append(other_candidate)
            
            # C値を計算
            if not longer_terms:
                # より長い候補語がない場合
                c_value = math.log2(length_a) * freq_a if length_a > 1 else freq_a
            else:
                # より長い候補語がある場合
                sum_freq = sum(candidates[term] for term in longer_terms)
                t_a = len(longer_terms)
                c_value = math.log2(length_a) * (freq_a - sum_freq / t_a) if length_a > 1 else (freq_a - sum_freq / t_a)
            
            c_values[candidate] = c_value
        
        return c_values
    
    def calculate_nc_value(self, text: str, c_values: Dict[str, float]) -> Dict[str, float]:
        """
        NC値を計算
        NC-value(a) = 0.8 × C-value(a) + 0.2 × Context(a)
        
        Context(a) = Σw∈Ca freq(w)
        ここで:
        - Ca: 候補語aと共起する文脈語の集合
        - freq(w): 文脈語wの頻度
        """
        # 名詞句を抽出
        noun_phrases = self.extract_noun_phrases(text)
        
        # 文脈語を収集（候補語の前後に出現する単語）
        context_words = defaultdict(set)
        all_morphemes = self.tokenizer_obj.tokenize(text, self.mode)
        
        # テキスト全体から文脈を抽出
        morpheme_strings = [m.normalized_form() for m in all_morphemes]
        
        for candidate in c_values:
            candidate_words = candidate.split()
            candidate_length = len(candidate_words)
            
            # 候補語の出現位置を探す
            for i in range(len(morpheme_strings) - candidate_length + 1):
                if morpheme_strings[i:i+candidate_length] == candidate_words:
                    # 前の単語を文脈語として追加
                    if i > 0:
                        prev_word = morpheme_strings[i-1]
                        if all_morphemes[i-1].part_of_speech()[0] in ['名詞', '動詞', '形容詞']:
                            context_words[candidate].add(prev_word)
                    
                    # 後の単語を文脈語として追加
                    if i + candidate_length < len(morpheme_strings):
                        next_word = morpheme_strings[i+candidate_length]
                        if all_morphemes[i+candidate_length].part_of_speech()[0] in ['名詞', '動詞', '形容詞']:
                            context_words[candidate].add(next_word)
        
        # NC値を計算
        nc_values = {}
        max_context = max(len(words) for words in context_words.values()) if context_words else 1
        
        for candidate, c_value in c_values.items():
            context_score = len(context_words[candidate]) / max_context if max_context > 0 else 0
            nc_value = 0.8 * c_value + 0.2 * context_score * abs(c_value)
            nc_values[candidate] = nc_value
        
        return nc_values
    
    def extract_terms(self, text: str, top_k: int = 20, use_nc_value: bool = True) -> List[Tuple[str, float]]:
        """
        専門用語を抽出
        
        Args:
            text: 入力テキスト
            top_k: 上位k件を返す
            use_nc_value: NC値を使用するか（FalseならC値のみ）
        
        Returns:
            [(用語, スコア), ...] の形式で上位k件
        """
        # C値を計算
        c_values = self.calculate_c_value(text)
        
        if not c_values:
            return []
        
        # NC値を使用するか選択
        if use_nc_value:
            scores = self.calculate_nc_value(text, c_values)
        else:
            scores = c_values
        
        # スコアでソートして上位k件を返す
        sorted_terms = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # 空白区切りを元に戻す
        result = []
        for term, score in sorted_terms[:top_k]:
            japanese_term = ''.join(term.split())  # 空白を除去して連結
            result.append((japanese_term, score))
        
        return result


def integrate_with_existing_extractor(text: str, existing_candidates: List[str]) -> List[Tuple[str, float]]:
    """
    既存のterm_extractor_embeding.pyの候補語リストにC値/NC値スコアを付与
    
    Args:
        text: 元のテキスト
        existing_candidates: generate_candidates_from_chunk()で生成された候補語リスト
    
    Returns:
        スコア付きの候補語リスト
    """
    extractor = CValueExtractor()
    
    # テキスト全体からC値/NC値を計算
    c_values = extractor.calculate_c_value(text)
    nc_values = extractor.calculate_nc_value(text, c_values) if c_values else {}
    
    # 既存の候補語にスコアを付与
    scored_candidates = []
    for candidate in existing_candidates:
        # 空白区切りバージョンも試す（C値計算は空白区切りで行われるため）
        spaced_candidate = ' '.join(extractor.tokenizer_obj.tokenize(candidate, extractor.mode))
        
        # NC値があればそれを、なければC値を、どちらもなければ長さベースのスコアを使用
        if candidate in nc_values:
            score = nc_values[candidate]
        elif spaced_candidate in nc_values:
            score = nc_values[spaced_candidate]
        elif candidate in c_values:
            score = c_values[candidate]
        elif spaced_candidate in c_values:
            score = c_values[spaced_candidate]
        else:
            # C値/NC値がない場合は、長さベースの簡易スコア
            score = math.log2(len(candidate)) if len(candidate) > 1 else 0.1
        
        scored_candidates.append((candidate, score))
    
    # スコアでソートして返す
    return sorted(scored_candidates, key=lambda x: x[1], reverse=True)


# 使用例
if __name__ == "__main__":
    sample_text = """
    医薬品製造管理及び品質管理に関する基準において、品質保証システムの構築は重要である。
    製造工程の管理と品質管理の両立が求められる。生物由来製品の製造には特別な注意が必要だ。
    安定性モニタリングと試験検査の実施により、医薬品の品質を確保する。
    """
    
    extractor = CValueExtractor()
    
    # C値のみで抽出
    print("=== C値による抽出 ===")
    c_terms = extractor.extract_terms(sample_text, top_k=10, use_nc_value=False)
    for term, score in c_terms:
        print(f"{term}: {score:.3f}")
    
    # NC値で抽出
    print("\n=== NC値による抽出 ===")
    nc_terms = extractor.extract_terms(sample_text, top_k=10, use_nc_value=True)
    for term, score in nc_terms:
        print(f"{term}: {score:.3f}")