"""DSR identity: a first name is not a person."""

import pandas as pd

from sndintel.identity import attach_dsr_identity, dsr_display_name, dsr_unit_id, parse_dsr_unit_id


def test_two_shahids_never_share_an_id():
    a = dsr_unit_id("Karachi", "Dist A", "Shahid")
    b = dsr_unit_id("Lahore", "Dist B", "Shahid")
    assert a != b
    assert dsr_display_name(a) == "Shahid"
    assert dsr_display_name(b) == "Shahid"
    assert parse_dsr_unit_id(a) == ("Shahid", "Karachi", "Dist A")


def test_attach_identity_sets_unique_grain_id():
    df = pd.DataFrame(
        [
            {"city": "Karachi", "distributor": "Eva", "dsr_name": "Amir"},
            {"city": "Lahore", "distributor": "Eva", "dsr_name": "Amir"},
        ]
    )
    out = attach_dsr_identity(df)
    assert out["grain_id"].nunique() == 2
    assert list(out["dsr_name"]) == ["Amir", "Amir"]
