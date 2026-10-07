from indicators.oi import classify_price_oi, compute_oi_metrics


def test_classify_price_oi_matrix():
    # Price falling + OI increasing
    c1, _ = classify_price_oi(price_change_pct=-2.5, oi_change_pct=+6.0)
    assert c1 == "OI BUILD + PRICE WEAKNESS"

    # Price rising + OI increasing
    c2, _ = classify_price_oi(price_change_pct=+3.5, oi_change_pct=+8.0)
    assert c2 == "OI BUILD + PRICE STRENGTH"

    # Price falling + OI falling
    c3, _ = classify_price_oi(price_change_pct=-1.5, oi_change_pct=-4.0)
    assert c3 == "OI CONTRACTION + PRICE WEAKNESS"

    # Price rising + OI falling
    c4, _ = classify_price_oi(price_change_pct=+2.0, oi_change_pct=-3.0)
    assert c4 == "OI CONTRACTION + PRICE STRENGTH"


def test_compute_oi_metrics():
    # Current: 110M, 1H ago: 100M (+10%), 4H ago: 90M (+22.2%)
    metrics = compute_oi_metrics(
        current_oi=110_000_000.0,
        oi_1h_ago=100_000_000.0,
        oi_4h_ago=90_000_000.0,
        price_1h_change_pct=-1.5,
    )
    assert metrics.change_1h_pct == 10.0
    assert metrics.change_4h_pct == 22.22
    assert metrics.classification == "OI BUILD + PRICE WEAKNESS"
