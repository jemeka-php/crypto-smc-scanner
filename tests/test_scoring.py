from config import ScoringWeights
from indicators.oi import compute_oi_metrics
from smc.structure import MarketStructure
from smc.displacement import Displacement
from smc.order_blocks import OrderBlock
from scanner.scoring import calculate_candidate_score


def test_calculate_candidate_score():
    weights = ScoringWeights()
    oi_metrics = compute_oi_metrics(110.0, 100.0, 95.0, price_1h_change_pct=-2.0)
    structure_1h = MarketStructure("1H", "BULLISH", [], None, None, [], None, None)
    disp = [
        Displacement(
            start_index=0,
            end_index=2,
            start_timestamp=1,
            end_timestamp=3,
            direction="BULLISH",
            total_displacement=15.0,
            atr=5.0,
            displacement_ratio=3.0,
        )
    ]
    obs = [
        OrderBlock(
            index=0,
            timestamp=1,
            ob_type="BULLISH",
            high=105.0,
            low=100.0,
            open=104.0,
            close=101.0,
            state="ACTIVE",
            displacement_ratio=3.0,
        )
    ]

    breakdown = calculate_candidate_score(
        quote_volume_24h=120_000_000.0,
        min_volume=50_000_000.0,
        rvol=2.4,
        min_rvol=1.5,
        oi_metrics=oi_metrics,
        min_oi_pct=5.0,
        price_change_24h_pct=-4.2,
        min_volatility_pct=3.0,
        htf_location="NEAR DEMAND",
        sweeps=[],
        nearby_liquidity=[],
        displacements=disp,
        structure_1h=structure_1h,
        active_obs=obs,
        active_fvgs=[],
        weights=weights,
    )

    assert breakdown.total_score > 60.0
    assert breakdown.liquidity_score == 10.0
    assert breakdown.volatility_score == 7.5
    assert breakdown.htf_location_score == 15.0
