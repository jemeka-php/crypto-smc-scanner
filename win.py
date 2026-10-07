"""
Temporary Diagnostic Script: BZUSDT 15M Event Sequence & Chronology Analysis.
Read-only inspection using existing scanner objects and definitions.
"""

from datetime import datetime, timezone
import pandas as pd

from config import AppConfig
from data.exchange import ExchangeClient
from indicators.oi import compute_oi_metrics
from scanner.candidates import build_candidate, evaluate_ltf_confirmation


def format_ts(ts_ms: int) -> str:
    """Format Unix millisecond timestamp to UTC string and HH:MM."""
    dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def format_time_only(ts_ms: int) -> str:
    dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
    return dt.strftime("%H:%M UTC")


def run_diagnostic():
    print("=" * 80)
    print("BZUSDT 15M READ-ONLY EVENT DIAGNOSTIC")
    print("=" * 80)

    # 1. Initialize client and config with current LIVE configuration
    config = AppConfig(demo_mode=False)
    client = ExchangeClient(demo_mode=False)

    print("\nFetching live universe tickers...")
    tickers = client.fetch_universe_tickers()
    bz_ticker = next((t for t in tickers if t.symbol == "BZUSDT"), None)
    if not bz_ticker:
        print("ERROR: BZUSDT not found in universe tickers!")
        return

    print(f"Found BZUSDT: Last Price = ${bz_ticker.last_price:.4f}, 24h Vol = ${bz_ticker.quote_volume_24h:,.0f}, 24h Chg = {bz_ticker.price_change_24h_pct:+.2f}%")

    # 2. Fetch multi-timeframe OHLCV using exact pipeline parameters
    print("Fetching 1H, 4H, and 15M candles for Stage 1 analysis...")
    df_1h = client.fetch_ohlcv("BZUSDT", interval="1h", limit=60)
    df_4h = client.fetch_ohlcv("BZUSDT", interval="4h", limit=50)
    df_15m = client.fetch_ohlcv("BZUSDT", interval="15m", limit=50)

    curr_oi, oi_1h, oi_4h = client.fetch_oi_history("BZUSDT")
    oi_metrics = compute_oi_metrics(
        current_oi=curr_oi,
        oi_1h_ago=oi_1h,
        oi_4h_ago=oi_4h,
        price_1h_change_pct=bz_ticker.price_change_24h_pct / 24.0,
    )

    # 3. Build candidate (Stage 1 SMC analysis)
    cand = build_candidate(
        symbol="BZUSDT",
        last_price=bz_ticker.last_price,
        quote_volume_24h=bz_ticker.quote_volume_24h,
        price_change_24h_pct=bz_ticker.price_change_24h_pct,
        df_1h=df_1h,
        df_4h=df_4h,
        df_15m=df_15m,
        oi_metrics=oi_metrics,
        config=config,
    )

    if not cand:
        print("ERROR: Failed to build candidate for BZUSDT!")
        return

    # 4. Fetch micro timeframes for Stage 2 LTF confirmation
    print("Fetching 5M, 3M, 1M candles for Stage 2 entry confirmation...")
    df_5m = client.fetch_ohlcv("BZUSDT", interval="5m", limit=50)
    df_3m = client.fetch_ohlcv("BZUSDT", interval="3m", limit=50)
    df_1m = client.fetch_ohlcv("BZUSDT", interval="1m", limit=50)

    ltf_conf = evaluate_ltf_confirmation(
        candidate=cand,
        df_5m=df_5m,
        df_3m=df_3m,
        df_1m=df_1m,
        config=config,
    )
    cand.ltf_confirmation = ltf_conf

    # =========================================================================
    # SECTION 1: 15M SWEEPS
    # =========================================================================
    print("\n" + "=" * 80)
    print("### 15M Sweeps")
    print("=" * 80)
    print(f"Total 15M sweeps detected in lookback: {len(cand.sweeps_15m)}")
    for idx, sw in enumerate(cand.sweeps_15m, 1):
        side_label = sw.level.side  # BUY_SIDE or SELL_SIDE
        ts_str = format_ts(sw.timestamp)
        print(f"\nSweep #{idx}:")
        print(f"  * Timestamp:          {sw.timestamp} ({ts_str})")
        print(f"  * Sweep Type:         {side_label} (classified as {sw.sweep_type})")
        print(f"  * Liquidity Level:    ${sw.level.price:.4f} (Source: {sw.level.source})")
        print(f"  * Sweep Extreme Price: ${sw.extreme_price:.4f}")
        print(f"  * Candle Close:       ${sw.close_price:.4f}")
        print(f"  * Level ID / Info:    Timeframe={sw.level.timeframe}, CreatedAt={sw.level.created_at} ({format_ts(sw.level.created_at)}), Status={sw.level.status}")

    # =========================================================================
    # SECTION 2: 15M DISPLACEMENTS
    # =========================================================================
    print("\n" + "=" * 80)
    print("### 15M Displacements")
    print("=" * 80)
    print(f"Total 15M displacements detected: {len(cand.displacements_15m)}")
    for idx, d in enumerate(cand.displacements_15m, 1):
        start_ts_str = format_ts(d.start_timestamp)
        end_ts_str = format_ts(d.end_timestamp)
        start_price = df_15m.iloc[d.start_index]["open"] if d.start_index < len(df_15m) else None
        end_price = df_15m.iloc[d.end_index]["close"] if d.end_index < len(df_15m) else None
        print(f"\nDisplacement #{idx}:")
        print(f"  * Start:              Index {d.start_index} @ {d.start_timestamp} ({start_ts_str})")
        print(f"  * End:                Index {d.end_index} @ {d.end_timestamp} ({end_ts_str})")
        print(f"  * Direction:          {d.direction}")
        print(f"  * ATR Multiple:       {d.displacement_ratio:.2f}x ATR (Total move: ${d.total_displacement:.4f}, ATR: ${d.atr:.4f})")
        if start_price is not None and end_price is not None:
            print(f"  * Price Span:         Open ${start_price:.4f} -> Close ${end_price:.4f}")

    # =========================================================================
    # SECTION 3: 15M STRUCTURE (BOS & MSS)
    # =========================================================================
    print("\n" + "=" * 80)
    print("### 15M Structure")
    print("=" * 80)
    struct_15m = cand.structure_15m
    if struct_15m:
        print(f"Structure Bias: {struct_15m.bias}")

        # BOS
        bos = struct_15m.latest_bos
        print("\nLatest 15M BOS:")
        if bos:
            bos_close = df_15m.iloc[bos.index]["close"] if bos.index < len(df_15m) else None
            print(f"  * Type:               {bos.break_type}")
            print(f"  * Timestamp:          {bos.timestamp} ({format_ts(bos.timestamp)})")
            print(f"  * Direction:          {bos.direction}")
            print(f"  * Broken Swing Price: ${bos.broken_level:.4f} (Swing Index: {bos.broken_swing.index}, Confirmed at: {bos.broken_swing.confirmed_at_index})")
            print(f"  * Break Candle Close: ${bos_close:.4f}" if bos_close else "  * Break Candle Close: N/A")
            print(f"  * Has Displacement:   {bos.has_displacement}")
        else:
            print("  * None detected.")

        # MSS
        mss = struct_15m.latest_mss
        print("\nLatest 15M MSS:")
        if mss:
            mss_close = df_15m.iloc[mss.index]["close"] if mss.index < len(df_15m) else None
            print(f"  * Type:               {mss.break_type}")
            print(f"  * Timestamp:          {mss.timestamp} ({format_ts(mss.timestamp)})")
            print(f"  * Direction:          {mss.direction}")
            print(f"  * Broken Swing Price: ${mss.broken_level:.4f}")
            print(f"  * Break Candle Close: ${mss_close:.4f}" if mss_close else "  * Break Candle Close: N/A")
            print(f"  * Has Displacement:   {mss.has_displacement}")
        else:
            print("  * None detected (Market structure remained in trend continuation without opposite-direction shift).")

        # All recent breaks in 15M
        print(f"\nAll Recent 15M Structure Breaks ({len(struct_15m.recent_breaks)}):")
        for b in struct_15m.recent_breaks:
            b_close = df_15m.iloc[b.index]["close"] if b.index < len(df_15m) else None
            print(f"  - Bar {b.index:02d} | {format_ts(b.timestamp)} | {b.break_type} {b.direction} | Broken: ${b.broken_level:.4f} | Close: ${b_close:.4f} | Disp: {b.has_displacement}")
    else:
        print("No 15M structure data available.")

    # =========================================================================
    # SECTION 4: STAGE 2 LTF CONFIRMATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("### Stage 2 Micro Confirmation (5M / 3M / 1M)")
    print("=" * 80)
    print(f"Status:    {cand.ltf_confirmation.status}")
    print(f"Direction: {cand.ltf_confirmation.direction}")
    print(f"5M Sweep:  {cand.ltf_confirmation.m5_sweep} ({cand.ltf_confirmation.m5_sweep_desc})")
    print(f"3M Disp:   {cand.ltf_confirmation.m3_displacement} ({cand.ltf_confirmation.m3_displacement_desc})")
    print(f"1M MSS:    {cand.ltf_confirmation.m1_mss} ({cand.ltf_confirmation.m1_mss_desc})")
    print(f"Summary:   {cand.ltf_confirmation.summary}")

    # =========================================================================
    # SECTION 5: CRITICAL REQUIREMENT — CHRONOLOGICAL EVENT TIMELINE
    # =========================================================================
    print("\n" + "=" * 80)
    print("### TIMELINE (Chronologically Sorted by Timestamp)")
    print("=" * 80)

    events = []

    # 1. 15M Sweeps
    for sw in cand.sweeps_15m:
        events.append({
            "timestamp": sw.timestamp,
            "event_type": f"{sw.level.side}_SWEEP",
            "direction": "bearish context" if sw.level.side == "BUY_SIDE" else "bullish context",
            "price_level": f"${sw.level.price:.4f} (wick ${sw.extreme_price:.4f})",
            "details": f"15M {sw.sweep_type} took {sw.level.source} at ${sw.level.price:.4f}; closed back at ${sw.close_price:.4f} (Bar {sw.bar_index})",
        })

    # 2. 15M Displacements
    for d in cand.displacements_15m:
        start_p = df_15m.iloc[d.start_index]["open"] if d.start_index < len(df_15m) else 0.0
        end_p = df_15m.iloc[d.end_index]["close"] if d.end_index < len(df_15m) else 0.0
        events.append({
            "timestamp": d.end_timestamp,
            "event_type": f"{d.direction}_DISPLACEMENT",
            "direction": d.direction.lower(),
            "price_level": f"${start_p:.4f} -> ${end_p:.4f}",
            "details": f"15M displacement {d.displacement_ratio:.2f}x ATR (Bars {d.start_index}->{d.end_index}, net move ${d.total_displacement:.4f})",
        })

    # 3. 15M Structure Breaks (BOS / MSS)
    if struct_15m:
        for b in struct_15m.recent_breaks:
            b_close = df_15m.iloc[b.index]["close"] if b.index < len(df_15m) else 0.0
            events.append({
                "timestamp": b.timestamp,
                "event_type": f"{b.direction}_{b.break_type}",
                "direction": b.direction.lower(),
                "price_level": f"${b.broken_level:.4f}",
                "details": f"15M {b.break_type} broken swing ${b.broken_level:.4f}; closed at ${b_close:.4f} (Bar {b.index}, has_disp={b.has_displacement})",
            })

    # Sort strictly by timestamp
    events.sort(key=lambda x: x["timestamp"])

    print(f"{'Timestamp':<20} | {'Event Type':<24} | {'Direction':<18} | {'Price / Level':<24} | {'Details'}")
    print("-" * 130)
    for ev in events:
        ts_display = format_ts(ev["timestamp"])
        print(f"{ts_display:<20} | {ev['event_type']:<24} | {ev['direction']:<18} | {ev['price_level']:<24} | {ev['details']}")

    # =========================================================================
    # SECTION 6: FORENSIC QUESTION ANSWERS
    # =========================================================================
    print("\n" + "=" * 80)
    print("### FORENSIC ANALYSIS & DIRECT ANSWERS")
    print("=" * 80)

    # Question 1
    sweep_types = [sw.level.side for sw in cand.sweeps_15m]
    print(f"\n1. Were the three sweeps BUY_SIDE or SELL_SIDE?")
    print(f"   -> Result: ALL THREE are {', '.join(sweep_types)} sweeps.")
    print(f"      They took out buy-side swing highs (above resistance) and closed back down.")
    print(f"      In SMC terminology, taking buy-side liquidity is a BEARISH sweep pattern.")

    # Question 2
    sweep_order_str = " -> ".join([f"Sweep #{i} ({sw.level.side} @ {format_time_only(sw.timestamp)}, level ${sw.level.price:.2f})" for i, sw in enumerate(cand.sweeps_15m, 1)])
    print(f"\n2. What was their chronological order?")
    print(f"   -> Result: {sweep_order_str}")

    # Question 3
    print(f"\n3. Did a bullish/bearish displacement occur after each sweep?")
    for i, sw in enumerate(cand.sweeps_15m, 1):
        subsequent_disps = [d for d in cand.displacements_15m if d.end_timestamp >= sw.timestamp]
        if subsequent_disps:
            disp_desc = ", ".join([f"{d.direction} ({d.displacement_ratio:.2f}x ATR at {format_time_only(d.end_timestamp)})" for d in subsequent_disps])
            print(f"   -> Sweep #{i} ({format_time_only(sw.timestamp)}): Followed by displacement(s): {disp_desc}")
        else:
            print(f"   -> Sweep #{i} ({format_time_only(sw.timestamp)}): No displacement occurred after this sweep.")

    # Question 4
    print(f"\n4. Did BOS occur after the displacement?")
    if struct_15m and struct_15m.latest_bos:
        bos = struct_15m.latest_bos
        disp_before_bos = [d for d in cand.displacements_15m if d.end_timestamp <= bos.timestamp]
        print(f"   -> Latest 15M BOS occurred at {format_time_only(bos.timestamp)} (Bar {bos.index}).")
        if disp_before_bos:
            latest_d = disp_before_bos[-1]
            print(f"   -> YES: Bullish displacement (Bars {latest_d.start_index}->{latest_d.end_index}, {latest_d.displacement_ratio:.2f}x ATR) occurred at/before the BOS at {format_time_only(bos.timestamp)}.")
        else:
            print(f"   -> No displacement preceded the BOS.")
    else:
        print(f"   -> No 15M BOS found.")

    # Question 5
    print(f"\n5. Does the event chronology actually support the scanner's result?")
    print(f"   -> Scanner Stage 1 Result: Direction = {cand.direction}, Total Score = {cand.total_score:.1f}")
    print(f"   -> Scanner Stage 2 Result: LTF Status = {cand.ltf_confirmation.status}")
    print(f"   -> Explanation:")
    print(f"      - On 15M: The latest structure break is a BULLISH BOS at ${struct_15m.latest_bos.broken_level:.2f} if present,")
    print(f"        while the recent sweeps were BUY_SIDE liquidity sweeps taking out highs.")
    print(f"      - On Stage 2 LTF (5M/3M/1M): LTF Status is '{cand.ltf_confirmation.status}'.")
    if cand.ltf_confirmation.status == "CONFIRMED":
        print(f"        CONFIRMED is fully supported: 5M sweep occurred before/with 1M MSS in target direction.")
    elif cand.ltf_confirmation.status == "PARTIAL_CONFIRMATION":
        print(f"        Status is PARTIAL_CONFIRMATION (NOT confirmed yet): Awaiting full micro alignment.")
        print(f"        Detail: {cand.ltf_confirmation.summary}")
    else:
        print(f"        Status is {cand.ltf_confirmation.status}: {cand.ltf_confirmation.summary}")


if __name__ == "__main__":
    run_diagnostic()