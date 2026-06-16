import pandas as pd

from task5lib.splits import assign_splits, collect_groups


def test_collect_groups_uses_entry_for_lexicon():
    df = pd.DataFrame({
        "source_code": ["01", "01"],
        "row_id": [1, 2],
        "entry": ["阿爸", "阿爸"],
    })
    groups = collect_groups({"01": df})
    assert groups["01:entry:阿爸"] == ["01:1", "01:2"]


def test_assign_splits_has_no_group_overlap():
    groups = {f"g{i}": [f"06:{i}"] for i in range(100)}
    splits = assign_splits(groups, seed=42)
    group_by_item = {}
    for split, ids in splits.items():
        for item_id in ids:
            assert item_id not in group_by_item
            group_by_item[item_id] = split
    assert set(splits) == {"train", "val", "test"}
    assert len(splits["train"]) > len(splits["val"]) > 0
    assert len(splits["test"]) > 0
