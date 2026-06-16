from task5lib.metrics import char_bleu, keyword_hits


def test_char_bleu_perfect_match_is_high():
    assert char_bleu(["很热"], ["很热"]) > 99


def test_keyword_hits_counts_feature_terms():
    result = keyword_hits("覅出去，热显热，冇水")
    assert result["覅"] == 1
    assert result["显"] == 1
    assert result["冇"] == 1
