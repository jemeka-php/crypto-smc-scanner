"""
End-to-End Two-Stage Market Scanner Pipeline.
Stage 1: Filters 300+ markets via Liquidity/Volatility/RVOL/OI and scores with 1H/15M SMC setup engine.
Stage 2: Evaluates 5M/3M/1M entry confirmation exclusively on shortlisted (top 5-15) candidates.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging
import time
from typing import Dict, List, Optional, Tuple
import pandas as pd

from config import AppConfig, FilterConfig
from data.exchange import ExchangeClient, MarketTicker
from indicators.oi import compute_oi_metrics
from indicators.volume import calculate_rvol
from scanner.candidates import (
    Candidate,
    FundingData,
    classify_funding_rate,
    build_candidate,
    evaluate_ltf_confirmation,
)

logger = logging.getLogger(__name__)


@dataclass
class ScanStats:
    markets_scanned: int
    passed_liquidity: int
    passed_volatility: int
    passed_rvol: int
    passed_oi: int
    smc_candidates: int
    top_candidates: int
    scan_timestamp: float
    is_stale: bool = False
    stage1_candidates: int = 0
    stage2_candidates: int = 0
    stage1_duration_s: float = 0.0
    stage2_duration_s: float = 0.0
    total_duration_s: float = 0.0
    api_requests_count: int = 0
    symbols_processed: int = 0
    error_message: Optional[str] = None


class MarketScanner:
    """
    Orchestrates the two-stage filtering, data fetching, and SMC evaluation pipeline.
    """

    def __init__(self, config: AppConfig, exchange_client: Optional[ExchangeClient] = None):
        self.config = config
        self.client = exchange_client or ExchangeClient(
            demo_mode=config.demo_mode,
            proxy_url=getattr(config, "proxy_url", None),
        )
        self.last_candidates: List[Candidate] = []
        self.last_stats: Optional[ScanStats] = None

    def run_scan(
        self,
        progress_callback=None,
    ) -> Tuple[List[Candidate], ScanStats]:
        """
        Execute two-stage scan:
        STAGE 1: 300+ universe -> Liquidity/Volatility -> RVOL/OI -> 1H/15M SMC -> 100-pt Score -> Top 5-15 shortlist
        STAGE 2: Top 5-15 shortlist ONLY -> Fetch 5M/3M/1M -> Micro liquidity sweep, displacement, MSS
        """
        t_total_start = time.time()
        if hasattr(self.client, "reset_request_count"):
            self.client.reset_request_count()

        # ================= STAGE 1: Universe to Candidate Shortlist =================
        t_stage1_start = time.time()

        # 1. Fetch universe tickers
        tickers = self.client.fetch_universe_tickers()
        total_markets = len(tickers)
        if total_markets == 0:
            err = getattr(self.client, "last_error_message", None) or "Unable to fetch universe tickers from exchange."
            stats = ScanStats(
                markets_scanned=0,
                passed_liquidity=0,
                passed_volatility=0,
                passed_rvol=0,
                passed_oi=0,
                smc_candidates=0,
                top_candidates=0,
                scan_timestamp=time.time(),
                is_stale=True,
                error_message=err,
            )
            return [], stats

        if progress_callback:
            progress_callback(10, f"Stage 1: Filtered {total_markets} perpetuals by liquidity and volatility...")

        # 2. Liquidity & Volatility filtering
        min_vol = self.config.filters.min_24h_quote_volume
        min_volat = self.config.filters.min_abs_price_change_pct

        passed_liq_tickers = [t for t in tickers if t.quote_volume_24h >= min_vol]
        passed_liq_count = len(passed_liq_tickers)

        passed_volat_tickers = [t for t in passed_liq_tickers if abs(t.price_change_24h_pct) >= min_volat]
        passed_volat_count = len(passed_volat_tickers)

        # Work with top liquid & volatile markets (cap at 25 for rate limit safety)
        shortlist = sorted(passed_volat_tickers, key=lambda t: t.quote_volume_24h, reverse=True)[:25]
        if len(shortlist) < 5:
            shortlist = sorted(passed_liq_tickers, key=lambda t: t.quote_volume_24h, reverse=True)[:15]

        passed_rvol_count = 0
        passed_oi_count = 0
        candidates: List[Candidate] = []

        def process_ticker(ticker: MarketTicker) -> Optional[Tuple[Candidate, bool, bool]]:
            sym = ticker.symbol
            try:
                # Fetch 1H OHLCV
                df_1h = self.client.fetch_ohlcv(sym, interval="1h", limit=60)
                if df_1h.empty or len(df_1h) < 25:
                    return None

                # Calculate RVOL on completed candles (drop forming candle)
                completed_1h_vol = df_1h["volume"].iloc[:-1]
                rvol, _, _ = calculate_rvol(completed_1h_vol, lookback=self.config.smc.volume_lookback)
                is_rvol_pass = rvol >= self.config.filters.min_rvol

                # Fetch OI history
                curr_oi, oi_1h, oi_4h = self.client.fetch_oi_history(sym)
                oi_metrics = compute_oi_metrics(
                    current_oi=curr_oi,
                    oi_1h_ago=oi_1h,
                    oi_4h_ago=oi_4h,
                    price_1h_change_pct=ticker.price_change_24h_pct / 24.0,
                )
                is_oi_pass = oi_metrics.change_1h_pct >= self.config.filters.min_1h_oi_change_pct

                # Fetch 4H (Macro Context) and 15M (Primary Setup Refinement) OHLCV
                df_4h = self.client.fetch_ohlcv(sym, interval="4h", limit=50)
                df_15m = self.client.fetch_ohlcv(sym, interval="15m", limit=50)

                # Fetch display-only funding info (completely isolated from scoring/filtering)
                funding_data = FundingData()
                try:
                    f_rate, f_ts, f_next = self.client.fetch_funding_info(sym)
                    if f_rate is not None:
                        f_label = classify_funding_rate(
                            f_rate,
                            pos_threshold=self.config.funding.positive_threshold,
                            neg_threshold=self.config.funding.negative_threshold,
                        )
                        funding_data = FundingData(
                            rate=f_rate,
                            rate_pct=round(f_rate * 100.0, 4),
                            label=f_label,
                            timestamp=f_ts,
                            next_funding_time=f_next,
                        )
                except Exception as fe:
                    logger.warning(f"Error fetching funding for {sym}: {fe}")
                    funding_data = FundingData()

                # Build candidate with 1H/15M setup analysis and 4H macro context
                cand = build_candidate(
                    symbol=sym,
                    last_price=ticker.last_price,
                    quote_volume_24h=ticker.quote_volume_24h,
                    price_change_24h_pct=ticker.price_change_24h_pct,
                    df_1h=df_1h,
                    df_4h=df_4h,
                    df_15m=df_15m,
                    oi_metrics=oi_metrics,
                    config=self.config,
                    funding=funding_data,
                )
                if cand:
                    return cand, is_rvol_pass, is_oi_pass
            except Exception as e:
                logger.warning(f"Error processing {sym}: {e}")
            return None

        total_shortlist = len(shortlist)
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {executor.submit(process_ticker, t): t for t in shortlist}
            completed_count = 0
            for fut in as_completed(futures):
                completed_count += 1
                if progress_callback:
                    pct = 15 + int((completed_count / max(total_shortlist, 1)) * 60)
                    t = futures[fut]
                    progress_callback(pct, f"Stage 1: Analyzed {t.symbol} 1H/15M setup ({completed_count}/{total_shortlist})...")
                res = fut.result()
                if res:
                    cand, r_pass, o_pass = res
                    candidates.append(cand)
                    if r_pass:
                        passed_rvol_count += 1
                    if o_pass:
                        passed_oi_count += 1

        # Sort candidates descending by total score (deterministic HTF + quantitative rubric)
        candidates.sort(key=lambda c: c.total_score, reverse=True)
        stage1_duration = time.time() - t_stage1_start
        stage1_candidate_count = len(candidates)

        # ================= STAGE 2: 5M/3M/1M Entry Confirmation on Top Candidates =================
        t_stage2_start = time.time()

        # Select top 5-15 candidates for Stage 2 micro analysis
        max_stage2_count = min(max(len(candidates), 0), 15)
        stage2_shortlist = candidates[:max_stage2_count]
        stage2_candidate_count = len(stage2_shortlist)

        if progress_callback:
            progress_callback(78, f"Stage 2: Evaluating 5M/3M/1M micro confirmation for top {stage2_candidate_count} candidates...")

        def process_ltf_candidate(cand: Candidate) -> None:
            sym = cand.symbol
            try:
                # Fetch 5M, 3M, 1M OHLCV strictly for shortlisted candidate
                df_5m = self.client.fetch_ohlcv(sym, interval="5m", limit=50)
                df_3m = self.client.fetch_ohlcv(sym, interval="3m", limit=50)
                df_1m = self.client.fetch_ohlcv(sym, interval="1m", limit=50)

                # Evaluate lower timeframe confirmation
                ltf_conf = evaluate_ltf_confirmation(
                    candidate=cand,
                    df_5m=df_5m,
                    df_3m=df_3m,
                    df_1m=df_1m,
                    config=self.config,
                )
                cand.ltf_confirmation = ltf_conf
            except Exception as e:
                logger.warning(f"Error evaluating LTF confirmation for {sym}: {e}")

        if stage2_shortlist:
            with ThreadPoolExecutor(max_workers=min(len(stage2_shortlist), 6)) as ltf_executor:
                ltf_futures = [ltf_executor.submit(process_ltf_candidate, c) for c in stage2_shortlist]
                for idx, fut in enumerate(as_completed(ltf_futures), 1):
                    fut.result()
                    if progress_callback:
                        pct = 78 + int((idx / max(len(stage2_shortlist), 1)) * 20)
                        progress_callback(pct, f"Stage 2: Micro confirmation checked ({idx}/{len(stage2_shortlist)})...")

        stage2_duration = time.time() - t_stage2_start
        total_duration = time.time() - t_total_start

        smc_candidates_count = len([c for c in candidates if c.stage in ("CONFIRMED", "SETUP_FORMING")])
        top_candidates_count = min(len(candidates), 15)

        req_count = getattr(self.client, "request_count", 0)

        stats = ScanStats(
            markets_scanned=total_markets,
            passed_liquidity=passed_liq_count,
            passed_volatility=passed_volat_count,
            passed_rvol=passed_rvol_count,
            passed_oi=passed_oi_count,
            smc_candidates=smc_candidates_count,
            top_candidates=top_candidates_count,
            scan_timestamp=time.time(),
            is_stale=self.client.is_stale,
            stage1_candidates=stage1_candidate_count,
            stage2_candidates=stage2_candidate_count,
            stage1_duration_s=round(stage1_duration, 2),
            stage2_duration_s=round(stage2_duration, 2),
            total_duration_s=round(total_duration, 2),
            api_requests_count=req_count,
            symbols_processed=total_shortlist,
            error_message=getattr(self.client, "last_error_message", None) if self.client.is_stale else None,
        )

        self.last_candidates = candidates
        self.last_stats = stats

        if progress_callback:
            progress_callback(100, f"Scan complete! {len(candidates)} candidates ranked. Top {stage2_candidate_count} LTF analyzed.")

        return candidates, stats
