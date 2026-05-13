from spreadsheet_rlm.bench.evaluation import _build_instruction


def test_sheet_level_instruction_frames_natural_language_as_workbook_edit():
    prompt = _build_instruction(
        "How can I use VBA code in Excel to automatically highlight non-matching pairs?",
        "D1:E22",
        "Sheet1",
        "Sheet-Level Manipulation",
    )

    assert "This is a workbook editing task" in prompt
    assert "Do not answer with explanatory prose" in prompt
    assert "preserve existing cell values" in prompt
    assert "formatting/style changes" in prompt
