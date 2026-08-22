from sndintel.ingest.shops import parse_shop_master
from sndintel.ingest.ssrs import parse_sales_file


def test_ssrs_parser_strips_chrome_and_totals(demo):
    df, report = parse_sales_file(demo["sales"])
    assert report.strategy == "ssrs_field_ids"
    assert set(["distributor", "dsr_name", "section", "store_id", "store_name", "sku", "year", "month", "period", "volume_mt"]).issubset(df.columns)
    assert df["store_id"].str.contains("Total").sum() == 0
    assert (df["volume_mt"] > 0).all()
    assert df["month"].between(1, 12).all()
    assert "T0001601407" in set(df["store_id"])
    hameed = df[df["store_id"] == "T0001601407"]
    assert not hameed.empty
    assert (hameed["store_name"] == "Hameed GS").all()
    assert "Maan" in " ".join(hameed["sku"].unique())


def test_ssrs_csv_roundtrip(demo):
    df, report = parse_sales_file(demo["sales_csv"])
    assert len(df) > 100
    assert report.n_clean_rows == len(df)


def test_shop_master_layout(demo):
    shops, report = parse_shop_master(demo["shops"])
    assert report.strategy == "headers"
    assert "T0001601407" in set(shops["store_id"])
    row = shops.set_index("store_id").loc["T0001601407"]
    assert row["city"] == "Quetta"
    assert row["zone"] == "West"
    assert row["section"] == "Alamdar Road"
    assert row["dsr_name"] == "ASHRAF KHAN"
    # unused trailing columns must not become store_id
    assert shops["store_id"].str.startswith("T").all()
