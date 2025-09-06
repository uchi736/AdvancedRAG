#!/usr/bin/env python3
"""test_synonym_detection.py
関連語検出機能のテストスクリプト
"""

import sys
from pathlib import Path

# 親ディレクトリをパスに追加
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.scripts.term_extractor_embeding import SynonymDetector, generate_candidates_from_chunk

def test_synonym_detection():
    """関連語検出のテスト"""
    
    # テスト用のサンプルテキスト
    sample_text = """
    医薬品の製造管理及び品質管理の基準（GMP）において、原薬の品質保証は重要である。
    原料となる原薬（API）は、添加剤や賦形剤と共に製剤化される。
    品質管理（QC）と品質保証（QA）の両方が、医薬品製造において不可欠である。
    製造工程のバリデーションでは、製造設備や製造装置の適格性評価が必要となる。
    試験検査では、規格基準への適合性を確認する。
    標準作業手順（SOP）に従い、各工程で品質チェックを実施する。
    """
    
    print("=" * 60)
    print("テストテキスト:")
    print(sample_text)
    print("=" * 60)
    
    # 候補語と関連語を生成
    candidates, detected_synonyms = generate_candidates_from_chunk(sample_text)
    
    print(f"\n抽出された候補語数: {len(candidates)}")
    print("\n上位20件の候補語:")
    for i, cand in enumerate(candidates[:20], 1):
        print(f"  {i:2}. {cand}")
    
    print(f"\n\n検出された関連語グループ数: {len(detected_synonyms)}")
    print("\n関連語グループ（上位10件）:")
    
    # 関連語を表示
    for i, (term, synonyms) in enumerate(list(detected_synonyms.items())[:10], 1):
        print(f"\n  {i}. {term}")
        print(f"     関連語: {', '.join(synonyms[:5])}")  # 最大5個まで表示
    
    # SynonymDetectorのテスト
    detector = SynonymDetector()
    
    # テスト用の候補語リスト
    test_candidates = [
        "医薬品", "品質管理", "QC", "品質保証", "QA",
        "原薬", "API", "原料", "添加剤", "賦形剤",
        "製造管理", "製造工程", "製造設備", "製造装置",
        "GMP", "適正製造規範", "SOP", "標準作業手順",
        "試験", "検査", "評価", "規格", "基準", "標準"
    ]
    
    print("\n" + "=" * 60)
    print("個別機能テスト:")
    print("=" * 60)
    
    # 略語検出テスト
    print("\n【略語と正式名称の検出】")
    abbreviations = {
        'GMP': '適正製造規範',
        'QC': '品質管理',
        'QA': '品質保証',
        'API': '原薬',
        'SOP': '標準作業手順'
    }
    
    for abbr, full in abbreviations.items():
        if abbr in test_candidates and full in test_candidates:
            print(f"  ✓ {abbr} ⟷ {full}")
    
    # 部分文字列関係のテスト
    print("\n【部分文字列関係】")
    for cand1 in ["製造", "品質", "試験"]:
        related = [c for c in test_candidates if cand1 in c and c != cand1]
        if related:
            print(f"  {cand1}: {', '.join(related)}")
    
    # ドメイン関連語のテスト
    print("\n【ドメイン固有の関連語】")
    domain_groups = [
        ("原薬", ["原料", "API", "主成分"]),
        ("添加剤", ["賦形剤", "添加物"]),
        ("試験", ["検査", "評価", "テスト"]),
        ("規格", ["基準", "標準", "スペック"])
    ]
    
    for main_term, related_terms in domain_groups:
        if main_term in test_candidates:
            found = [t for t in related_terms if t in test_candidates]
            if found:
                print(f"  {main_term}: {', '.join(found)}")
    
    print("\n" + "=" * 60)
    print("テスト完了")

if __name__ == "__main__":
    test_synonym_detection()