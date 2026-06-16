from task5lib.text import apply_corrections, normalize_text, stable_id


def test_normalize_text_keeps_wenzhou_marks():
    raw = "  Ａ１２  匄（hà）\r\n  覅   冇  "
    assert normalize_text(raw) == "A12 匄（hà） 覅 冇"


def test_apply_corrections_exact_string_only():
    text = "饭走归吃，不宿食堂吃。山头蚕窟"
    corrections = {
        "饭走归吃，不宿食堂吃": "饭走归吃，不宿食堂里吃",
        "山头蚕窟": "山头岙窟",
    }
    assert apply_corrections(text, corrections) == "饭走归吃，不宿食堂里吃。山头岙窟"


def test_stable_id_is_deterministic():
    assert stable_id("06", 12, "wz_to_zh") == stable_id("06", 12, "wz_to_zh")
    assert stable_id("06", 12, "wz_to_zh") != stable_id("06", 12, "zh_to_wz")
