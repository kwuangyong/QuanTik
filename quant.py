import numpy as np, pandas as pd, logging, warnings, glob, json, sys, hashlib, math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

# Windows may default redirected output to cp1252, while this CLI prints
# Vietnamese text, box-drawing characters, and emoji.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        try:
            _stream.reconfigure(
                encoding='utf-8', errors='replace',
                line_buffering=True, write_through=True,
            )
        except (OSError, ValueError):
            pass

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger("quant_pipeline")

# === CONFIG ===
@dataclass
class QuantConfig:
    LOOKBACK_DAYS: int = 252; MIN_HISTORY: int = 120; RISK_FREE_RATE: float = 0.045
    VOL_WINDOW_SHORT: int = 10; VOL_WINDOW_MED: int = 20; VOL_WINDOW_LONG: int = 60
    GARCH_P: int = 1; GARCH_Q: int = 1
    ARIMA_MAX_ORDER: int = 4; HMM_N_STATES: int = 3; HMM_N_ITER: int = 100
    ATR_PERIOD: int = 14; MAX_POSITION_RISK: float = 0.02; ACCOUNT_SIZE: float = 500_000_000
    FORECAST_HORIZON: int = 10; MONTE_CARLO_SIMS: int = 1000; CONFIDENCE_LEVEL: float = 0.95
    NORMALITY_ALPHA: float = 0.05
    # Settlement configuration is defined once in the V6 market section below.
    SWING_HORIZON_MIN: int = 7
    SWING_HORIZON_MAX: int = 20
    SWING_DEFAULT: int = 10
    TICK_SIZE_RULES: dict = field(default_factory=lambda: {
        10000: 10, 50000: 50, 999999999: 100
    })
    # === V4 NEW CONFIG ===
    HMM_N_INITS: int = 10          # Multiple random inits for robustness
    HMM_VOL_WINDOW: int = 10       # Rolling vol window for HMM features
    HMM_MOM_WINDOW: int = 5        # Short momentum window for HMM features
    SR_EXTREMA_ORDER: int = 3      # argrelextrema order for swing detection
    # === V4.1 BUG FIXES ===
    VNI_CRASH_1D_PCT: float = -0.02     # 1-day drop to flag crash
    VNI_CRASH_5D_PCT: float = -0.05     # 5-day drop to flag crash
    CRISIS_VOL_MULT: float = 2.0        # Multiply GARCH vol in crisis
    FORECAST_CAP_CRISIS: float = 2.0    # Max forecast % in crisis regime
    ABSOLUTE_SCORE_ANCHOR: float = 0.5  # Blend: 50% cross-sectional + 50% absolute
    MC_SEED_PER_SYMBOL: bool = True     # Deterministic per-symbol seed

    # === V6 VIETNAM MARKET / PRODUCTION SAFETY ===
    PRICE_INPUT_UNIT: str = 'THOUSAND_VND'  # KBS/VCI via vnstock_data: giá thường ở nghìn VND
    PRICE_OUTPUT_UNIT: str = 'VND'          # Chuẩn nội bộ duy nhất của pipeline
    DEFAULT_EXCHANGE: str = 'HOSE'
    SETTLEMENT_DAYS: int = 2                # Cổ phiếu/CCQ/CW: T+2
    MIN_HOLD_SESSIONS: int = 2              # Daily-bar approximation: sớm nhất bán chiều T+2
    BOARD_LOT: int = 100

    # Data quality
    MAX_STALE_BUSINESS_DAYS: int = 5
    MAX_ZERO_VOLUME_20D_PCT: float = 20.0
    MAX_MISSING_BUSINESS_DAY_PCT: float = 15.0
    PRICE_LIMIT_TOLERANCE: float = 0.015

    # Liquidity / capacity
    MIN_ADV20_VALUE_VND: float = 5_000_000_000
    MAX_ADV20_PARTICIPATION: float = 0.05
    MAX_POSITION_PCT: float = 0.15

    # Execution-cost assumptions (broker commission is configurable)
    COMMISSION_BPS_PER_SIDE: float = 15.0
    SELL_TAX_BPS: float = 10.0
    BASE_SLIPPAGE_BPS_PER_SIDE: float = 8.0
    MIN_NET_FORECAST_PCT: float = 1.0

    # Forecast robustness
    FORECAST_DRIFT_SHRINK: float = 0.25
    ENABLE_INTERNAL_STOP_OPTIMIZATION: bool = False  # Backtest/tuning thuộc file riêng
    GENERATE_VISUALS: bool = False
    VISUAL_TOP_N: int = 10
    VISUAL_OUTPUT_DIR: str = 'output'

CFG = QuantConfig()


# ==============================================================
# V6-A: VIETNAM MARKET RULES / UNITS / COSTS
# ==============================================================

class VNMarketRules:
    """Quy tắc giao dịch cơ sở Việt Nam. Tất cả giá đầu vào của class này là VND."""
    PROFILES = {
        'HOSE': {'aliases': {'HOSE','HSX','HOCHIMINH'}, 'limit_pct': 0.07, 'lot_size': 100},
        'HNX': {'aliases': {'HNX','HANOI'}, 'limit_pct': 0.10, 'lot_size': 100},
        'UPCOM': {'aliases': {'UPCOM','UPCO','UP'}, 'limit_pct': 0.15, 'lot_size': 100},
    }

    @classmethod
    def normalize_exchange(cls, exchange):
        x = str(exchange or CFG.DEFAULT_EXCHANGE).upper().replace(' ', '')
        for name, profile in cls.PROFILES.items():
            if x == name or x in profile['aliases']:
                return name
        return CFG.DEFAULT_EXCHANGE

    @classmethod
    def profile(cls, exchange):
        return cls.PROFILES[cls.normalize_exchange(exchange)]

    @classmethod
    def tick_size(cls, price_vnd, exchange='HOSE'):
        ex = cls.normalize_exchange(exchange)
        p = float(price_vnd)
        if ex == 'HOSE':
            if p < 10_000: return 10.0
            if p < 50_000: return 50.0
            return 100.0
        return 100.0

    @classmethod
    def round_price(cls, price_vnd, direction='nearest', exchange='HOSE'):
        if price_vnd is None or not np.isfinite(price_vnd):
            return None
        tick = cls.tick_size(price_vnd, exchange)
        q = float(price_vnd) / tick
        if direction == 'down': q = math.floor(q + 1e-12)
        elif direction == 'up': q = math.ceil(q - 1e-12)
        else: q = round(q)
        return float(q * tick)

    @classmethod
    def price_band(cls, reference_price_vnd, exchange='HOSE'):
        limit_pct = cls.profile(exchange)['limit_pct']
        floor_p = cls.round_price(reference_price_vnd * (1 - limit_pct), 'up', exchange)
        ceil_p = cls.round_price(reference_price_vnd * (1 + limit_pct), 'down', exchange)
        return {'floor': floor_p, 'ceiling': ceil_p, 'limit_pct': limit_pct}


class CostEngine:
    """Chi phí ước tính; commission/slippage là giả định cấu hình, không phải phí cố định toàn thị trường."""
    TIER_SLIPPAGE_BPS = {'A': 4.0, 'B': 7.0, 'C': 12.0, 'D': 25.0}

    @classmethod
    def analyze(cls, forecast=None, liquidity=None):
        forecast = forecast or {}; liquidity = liquidity or {}
        tier = liquidity.get('tier', 'D')
        slip = max(CFG.BASE_SLIPPAGE_BPS_PER_SIDE, cls.TIER_SLIPPAGE_BPS.get(tier, 25.0))
        roundtrip_bps = 2 * CFG.COMMISSION_BPS_PER_SIDE + CFG.SELL_TAX_BPS + 2 * slip
        gross = float(forecast.get('ensemble_ret_pct', 0) or 0)
        cost_pct = roundtrip_bps / 100.0
        return {
            'commission_bps_per_side': CFG.COMMISSION_BPS_PER_SIDE,
            'sell_tax_bps': CFG.SELL_TAX_BPS,
            'slippage_bps_per_side': slip,
            'roundtrip_cost_bps': round(roundtrip_bps, 1),
            'roundtrip_cost_pct': round(cost_pct, 3),
            'gross_forecast_pct': round(gross, 3),
            'net_forecast_pct': round(gross - cost_pct, 3),
            'breakeven_move_pct': round(cost_pct, 3),
        }


class DataQualityEngine:
    @staticmethod
    def audit(df, exchange='HOSE', as_of=None):
        if df is None or df.empty:
            return {'status': 'FAIL', 'score': 0, 'flags': ['NO_DATA']}
        flags = []; score = 100
        required = ['open','high','low','close','volume']
        missing_cols = [c for c in required if c not in df.columns]
        if missing_cols:
            return {'status':'FAIL','score':0,'flags':[f'MISSING_COLS:{missing_cols}']}

        numeric = df[required].apply(pd.to_numeric, errors='coerce')
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            return {'status': 'FAIL', 'score': 0, 'flags': ['NONFINITE_OHLCV']}
        if (numeric['volume'] < 0).any():
            return {'status': 'FAIL', 'score': 0, 'flags': ['NEGATIVE_VOLUME']}
        if not isinstance(df.index, pd.DatetimeIndex) or df.index.hasnans:
            return {'status': 'FAIL', 'score': 0, 'flags': ['INVALID_DATE_INDEX']}
        if not df.index.is_monotonic_increasing:
            return {'status': 'FAIL', 'score': 0, 'flags': ['UNSORTED_DATES']}
        df = numeric
        audit_day = pd.Timestamp(as_of if as_of is not None else datetime.now()).date()
        if df.index.max().date() > audit_day:
            return {'status': 'FAIL', 'score': 0, 'flags': ['FUTURE_DATA']}

        n = len(df)
        if n < CFG.MIN_HISTORY:
            flags.append(f'SHORT_HISTORY:{n}<{CFG.MIN_HISTORY}'); score -= 35
        dup = int(df.index.duplicated().sum())
        if dup:
            flags.append(f'DUPLICATE_DATES:{dup}'); score -= min(20, dup * 2)

        bad_ohlc = int(((df['high'] < df[['open','close']].max(axis=1)) |
                        (df['low'] > df[['open','close']].min(axis=1)) |
                        (df['high'] < df['low'])).sum())
        nonpos = int((df[['open','high','low','close']] <= 0).any(axis=1).sum())
        if bad_ohlc: flags.append(f'INVALID_OHLC:{bad_ohlc}'); score -= min(35, bad_ohlc * 5)
        if nonpos: flags.append(f'NONPOSITIVE_PRICE:{nonpos}'); score -= min(40, nonpos * 10)

        zero_vol_pct = float((df['volume'].tail(20) <= 0).mean() * 100)
        if zero_vol_pct > CFG.MAX_ZERO_VOLUME_20D_PCT:
            flags.append(f'ZERO_VOLUME_20D:{zero_vol_pct:.1f}%'); score -= 20

        try:
            first = pd.Timestamp(df.index.min()).date(); last = pd.Timestamp(df.index.max()).date()
            expected = max(int(np.busday_count(first, last)) + 1, 1)
            missing_pct = max(0.0, (expected - n) / expected * 100)
        except Exception:
            missing_pct = 0.0
        if missing_pct > CFG.MAX_MISSING_BUSINESS_DAY_PCT:
            flags.append(f'MISSING_SESSIONS_APPROX:{missing_pct:.1f}%'); score -= 15

        try:
            last_date = np.datetime64(pd.Timestamp(df.index.max()).date())
            today = np.datetime64(audit_day)
            stale_bd = max(int(np.busday_count(last_date, today)), 0)
        except Exception:
            stale_bd = 0
        if stale_bd > CFG.MAX_STALE_BUSINESS_DAYS:
            flags.append(f'STALE_DATA:{stale_bd}BD'); score -= 35

        limit_pct = VNMarketRules.profile(exchange)['limit_pct']
        ret = df['close'].pct_change().abs()
        jump_count = int((ret > limit_pct + CFG.PRICE_LIMIT_TOLERANCE).sum())
        if jump_count:
            flags.append(f'PRICE_JUMP_OR_CORP_ACTION:{jump_count}'); score -= min(15, jump_count * 3)

        score = int(np.clip(score, 0, 100))
        fatal = any(x.startswith(('INVALID', 'NONPOSITIVE', 'STALE', 'DUPLICATE')) for x in flags)
        status = 'FAIL' if fatal or score < 55 else ('PASS' if score >= 80 else 'WARN')
        return {
            'status': status, 'score': score, 'flags': flags, 'n_rows': n,
            'zero_volume_20d_pct': round(zero_vol_pct, 1),
            'missing_sessions_approx_pct': round(missing_pct, 1),
            'stale_business_days': stale_bd, 'price_jump_count': jump_count,
            'price_unit': 'VND',
        }


class LiquidityEngine:
    @staticmethod
    def analyze(df, exchange='HOSE'):
        if df is None or len(df) < 20:
            return {'tier':'D','pass':False,'reason':'INSUFFICIENT_DATA'}
        close = df['close'].astype(float); vol = df['volume'].astype(float)
        value = close * vol
        adv20_sh = float(vol.tail(20).mean())
        adv20_val = float(value.tail(20).mean())
        med20_val = float(value.tail(20).median())
        value_ratio_5_20 = float(value.tail(5).mean() / adv20_val) if adv20_val > 0 else 0
        zero_vol_pct = float((vol.tail(20) <= 0).mean() * 100)
        ret = close.pct_change()
        prev = close.shift(1)
        gap = (df['open'] / prev - 1).replace([np.inf,-np.inf], np.nan).dropna()
        gap_p95 = float(gap.abs().quantile(0.95) * 100) if len(gap) else 0
        limit_pct = VNMarketRules.profile(exchange)['limit_pct']
        near_ceiling_60d = int((ret.tail(60) >= limit_pct * 0.97).sum())
        near_floor_60d = int((ret.tail(60) <= -limit_pct * 0.97).sum())
        amihud = (ret.abs() / (value / 1e9).replace(0, np.nan)).tail(60).mean()
        amihud = float(amihud) if pd.notna(amihud) else None

        if adv20_val >= 50e9: tier = 'A'
        elif adv20_val >= 20e9: tier = 'B'
        elif adv20_val >= CFG.MIN_ADV20_VALUE_VND: tier = 'C'
        else: tier = 'D'
        liq_pass = bool(adv20_val >= CFG.MIN_ADV20_VALUE_VND and zero_vol_pct <= CFG.MAX_ZERO_VOLUME_20D_PCT)
        lot = VNMarketRules.profile(exchange)['lot_size']
        capacity_shares = int((adv20_sh * CFG.MAX_ADV20_PARTICIPATION) // lot) * lot
        return {
            'tier': tier, 'pass': liq_pass, 'adv20_shares': round(adv20_sh, 0),
            'adv20_value_vnd': round(adv20_val, 0), 'median20_value_vnd': round(med20_val, 0),
            'value_ratio_5_20': round(value_ratio_5_20, 2), 'zero_volume_20d_pct': round(zero_vol_pct,1),
            'capacity_shares': max(capacity_shares, 0), 'max_adv_participation_pct': CFG.MAX_ADV20_PARTICIPATION * 100,
            'gap_abs_p95_pct': round(gap_p95, 2), 'near_ceiling_60d': near_ceiling_60d,
            'near_floor_60d': near_floor_60d, 'amihud_60d_per_bn': round(amihud, 6) if amihud is not None else None,
        }


class ActionEngine:
    """Tách xếp hạng (score) khỏi hành động. Hard gates luôn thắng score."""
    @staticmethod
    def decide(report):
        rc = report.get('rec',{}); fc = report.get('fcast',{}); dq = report.get('data_quality',{})
        liq = report.get('liquidity',{}); costs = report.get('costs',{}); sl = report.get('sl',{})
        score = int(rc.get('score',0)); vni = rc.get('vni_regime','NEUTRAL')
        timing = fc.get('timing',''); net_fc = costs.get('net_forecast_pct', fc.get('ensemble_ret_pct',0))
        reasons = []
        if dq.get('status') not in ('PASS', 'WARN'): reasons.append('DATA_QUALITY_FAIL')
        if not liq.get('pass', False): reasons.append('LIQUIDITY_FAIL')
        if vni == 'CRISIS': reasons.append('VNI_CRISIS')
        if str(timing).startswith('🔴'): reasons.append('TIMING_BLOCKED')
        if sl.get('lock_risk_flag'): reasons.append('SETTLEMENT_LOCK_RISK')
        if net_fc < CFG.MIN_NET_FORECAST_PCT: reasons.append('NET_FORECAST_TOO_LOW')

        hard_block = any(x in reasons for x in ('DATA_QUALITY_FAIL','LIQUIDITY_FAIL','VNI_CRISIS','TIMING_BLOCKED'))
        if hard_block:
            action = 'AVOID'
        elif score >= 75 and vni not in ('BEAR','WEAK') and net_fc >= CFG.MIN_NET_FORECAST_PCT:
            action = 'BUY_NOW'
        elif score >= 65 and net_fc > 0 and vni != 'BEAR':
            action = 'BUY_SETUP'
        elif score >= 50 and net_fc > 0:
            action = 'WATCH'
        else:
            action = 'AVOID'

        gate_pass = action in ('BUY_NOW','BUY_SETUP')
        return {'action': action, 'gate_pass': gate_pass, 'reason_codes': reasons,
                'net_forecast_pct': round(float(net_fc),3)}


class DashboardArchitecture:
    MODULES = [
        ('01','Market Regime','VNINDEX regime, crash circuit breaker, breadth/sector context'),
        ('02','Data Quality','freshness, OHLC consistency, missing sessions, corporate-action jumps'),
        ('03','Liquidity & Tradability','ADV20, capacity, zero volume, gap/ceiling/floor risk'),
        ('04','Signal & Relative Strength','trend, momentum, CMF, RS vs VNINDEX, sector'),
        ('05','Forecast & Uncertainty','GARCH-t Monte Carlo, HMM regime, model agreement'),
        ('06','Costs & Risk','round-trip costs, VaR/CVaR, ATR stop, settlement lock'),
        ('07','Position & Exposure','risk budget, allocation cap, ADV capacity, board lot'),
        ('08','Action Gate','BUY_NOW / BUY_SETUP / WATCH / AVOID + reason codes'),
        ('09','System Health','source success, stale data, model failures, export status'),
    ]

    @classmethod
    def as_dataframe(cls):
        return pd.DataFrame(cls.MODULES, columns=['ID','Dashboard','Purpose'])

# ==============================================================
# BRIDGE: Doc Excel tu anhson.py + Lay OHLCV data
# V4.2 — Multi-source fallback: KBS (primary) → VCI (backup)
# VCI & TCBS deprecated — removed
# ==============================================================

class ScreenerBridge:
    """Cau noi giua anhson.py (Screener) va Quant Pipeline.

    V4.3: Unified UI — vnstock_data 4.0 (Market layer).
      - Primary: KBS  (KB Securities — ổn định cho HOSE daily)
      - Fallback: VCI (dự phòng khi KBS fail)
      - TCBS: deprecated, đã remove hoàn toàn.

    Design:
      - Dùng `from vnstock_data import Market` (Unified UI 4.0).
      - Gọi `Market().equity(symbol=...).ohlcv(start, end,
        interval='1D', source=src)` thay vì Quote(symbol, source).history().
      - Với mỗi symbol: thử SOURCE_CHAIN[0] trước; nếu data không
        hợp lệ (empty / < MIN_BARS phiên / exception), fallback tiếp theo.
      - Schema sau normalize: index=DatetimeIndex, cols=[open,high,low,close,volume].
    """

    # Thứ tự ưu tiên — swap phần tử để đổi primary/fallback
    SOURCE_CHAIN = ['KBS', 'VCI']
    MIN_BARS = 30         # Số phiên tối thiểu để coi là hợp lệ
    RETRY_SLEEP = 0.8     # Sleep giữa các lần fallback (giây)

    def __init__(self, source=None):
        """
        Args:
            source: None  -> dùng full fallback chain (khuyến nghị).
                    'KBS' -> ép chỉ dùng KBS, không fallback.
                    'VCI' -> ép chỉ dùng VCI, không fallback.
        """
        if source is None:
            self.chain = list(self.SOURCE_CHAIN)
        else:
            self.chain = [source.upper()]
        self._market = None  # Lazy-init Market instance

    @property
    def market(self):
        """Lazy import Market — tránh crash nếu vnstock_data chưa cài."""
        if self._market is None:
            from vnstock_data import Market
            self._market = Market()
        return self._market

    # ---------- Excel screener I/O (không đổi) ----------
    def read_screener_excel(self, filepath=None):
        if filepath is None:
            filepath = self._find_latest()
        if filepath is None:
            log.error("Khong tim thay file Excel tu screener!")
            return [], pd.DataFrame()
        log.info(f"Doc screener Excel: {filepath}")
        try:
            df = pd.read_excel(filepath)
            sym_col = None
            for col in df.columns:
                if any(kw in str(col).lower() for kw in ['ma', 'symbol', 'ticker', 'ma ck']):
                    sym_col = col; break
            if sym_col is None: sym_col = df.columns[0]
            symbols = [s for s in df[sym_col].astype(str).str.strip().str.upper().tolist()
                       if s and len(s) == 3 and s.isalpha()]
            log.info(f"Doc duoc {len(symbols)} ma: {symbols[:10]}...")
            return symbols, df
        except Exception as e:
            log.error(f"Loi doc Excel: {e}"); return [], pd.DataFrame()

    def _find_latest(self):
        patterns = ['quant *.xlsx', 'Bo_Loc_Co_Phieu*.xlsx', 'bo_loc*.xlsx', '*screener*.xlsx', '*Bo_Loc*.xlsx']
        files = []
        for p in patterns:
            files.extend(glob.glob(p)); files.extend(glob.glob(f'**/{p}', recursive=True))
        # Loại bỏ file lock tạm của Excel (~$...) — không phải file dữ liệu thật
        files = [f for f in files if not Path(f).name.startswith('~$')]
        return max(files, key=lambda f: Path(f).stat().st_mtime) if files else None

    # ---------- OHLCV fetching — core logic mới ----------
    def _normalize_ohlcv(self, df, source=None):
        """Chuẩn hóa schema output về [open, high, low, close, volume] với DatetimeIndex.

        Xử lý mọi biến thể column naming giữa KBS / VCI.
        Trả về None nếu schema không thể normalize.
        """
        if df is None or df.empty:
            return None

        # Map column names (case-insensitive, handle cả 'time'/'date'/'trading_date')
        cm = {}
        for c in df.columns:
            cl = str(c).lower().strip()
            if 'time' in cl or 'date' in cl:
                cm[c] = 'time'
            elif cl in ('open', 'o'):
                cm[c] = 'open'
            elif cl in ('high', 'h'):
                cm[c] = 'high'
            elif cl in ('low', 'l'):
                cm[c] = 'low'
            elif cl in ('close', 'c'):
                cm[c] = 'close'
            elif 'volume' in cl or cl == 'v':
                cm[c] = 'volume'
        df = df.rename(columns=cm)

        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(c in df.columns for c in required):
            return None

        # Set DatetimeIndex
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time')
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        df = df.sort_index()

        # Numeric coercion + drop bad rows
        for c in required:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=['close'])

        # Chuẩn hóa đơn vị giá về VND ngay tại data boundary.
        unit = str(CFG.PRICE_INPUT_UNIT).upper()
        if unit == 'THOUSAND_VND':
            scale = 1000.0
        elif unit == 'VND':
            scale = 1.0
        elif unit == 'AUTO':
            median_close = float(df['close'].median())
            scale = 1000.0 if median_close < 1000 else 1.0
        else:
            raise ValueError(f'PRICE_INPUT_UNIT không hợp lệ: {CFG.PRICE_INPUT_UNIT}')
        for c in ['open','high','low','close']:
            df[c] = df[c] * scale

        # Deduplicate + basic structural cleanup before downstream models.
        df = df[~df.index.duplicated(keep='last')].sort_index()
        df = df[(df['close'] > 0) & (df['high'] >= df['low'])]
        out = df[required] if not df.empty else None
        if out is not None:
            out.attrs['price_unit'] = 'VND'
            out.attrs['source'] = source
        return out

    def _fetch_single(self, symbol, start, end, source):
        """Fetch 1 symbol từ 1 source cụ thể. Raise exception nếu fail.

        Unified UI 4.0: Market().equity(symbol=...).ohlcv(start, end,
        interval='1D', source=src)
        """
        df = self.market.equity(symbol=symbol).ohlcv(
            start=start,
            end=end,
            interval='1D',
            source=source,
        )
        return self._normalize_ohlcv(df, source=source)

    def _fetch_with_fallback(self, symbol, start, end):
        """Thử lần lượt từng source trong chain. Trả về (df, source_used) hoặc (None, None)."""
        import time
        last_err = None
        for src in self.chain:
            try:
                df = self._fetch_single(symbol, start, end, src)
                if df is not None and len(df) >= self.MIN_BARS:
                    return df, src
                # Data không đủ — thử source tiếp theo
                last_err = f"insufficient bars ({0 if df is None else len(df)} < {self.MIN_BARS})"
            except Exception as e:
                last_err = str(e)
            time.sleep(self.RETRY_SLEEP)  # backoff giữa các source
        log.warning(f"  {symbol}: all sources failed — last: {last_err}")
        return None, None

    def fetch_ohlcv(self, symbols, days=252, delay=0.3):
        """Batch fetch với fallback chain cho từng symbol."""
        import time
        data = {}
        end = datetime.now().strftime('%Y-%m-%d')
        calendar_days = max(int(days * 1.7), days + 90)
        start = (datetime.now() - timedelta(days=calendar_days)).strftime('%Y-%m-%d')
        src_stats = {s: 0 for s in self.chain}  # Track source nào đã dùng

        for i, sym in enumerate(symbols):
            log.info(f"[{i+1}/{len(symbols)}] Fetching {sym}...")
            df, src_used = self._fetch_with_fallback(sym, start, end)
            if df is not None:
                data[sym] = df.tail(days).copy()
                src_stats[src_used] = src_stats.get(src_used, 0) + 1
                log.info(f"  OK {sym}: {len(data[sym])} phien [src={src_used}]")
            time.sleep(delay)

        log.info(f"Data: {len(data)}/{len(symbols)} symbols loaded")
        log.info(f"Source breakdown: {src_stats}")
        return data

    def fetch_index(self, idx='VNINDEX', days=252):
        """Fetch index với cùng fallback chain.

        Lưu ý: VCI dùng symbol format khác cho index. Nếu KBS fail và VCI không
        hiểu 'VNINDEX', pipeline vẫn chạy được vì cfg cho phép idx_df=None.
        """
        import time
        end = datetime.now().strftime('%Y-%m-%d')
        calendar_days = max(int(days * 1.7), days + 90)
        start = (datetime.now() - timedelta(days=calendar_days)).strftime('%Y-%m-%d')
        last_err = None
        for src in self.chain:
            try:
                try:
                    raw = self.market.index(symbol=idx).ohlcv(start=start, end=end, interval='1D', source=src)
                except Exception:
                    # Compatibility fallback for older vnstock_data versions.
                    raw = self.market.equity(symbol=idx).ohlcv(start=start, end=end, interval='1D', source=src)
                df = self._normalize_ohlcv(raw, source=src)
                if df is not None and len(df) >= self.MIN_BARS:
                    df = df.tail(days).copy()
                    log.info(f"Index {idx}: {len(df)} phien [src={src}]")
                    return df
                last_err = f'insufficient bars ({0 if df is None else len(df)})'
            except Exception as e:
                last_err = str(e)
            time.sleep(self.RETRY_SLEEP)
        log.warning(f"Index {idx}: fetch failed on all sources — last: {last_err}")
        return None

    def fetch_exchange_map(self, symbols):
        """Tra cứu sàn bằng Reference().equity.list_by_exchange(); fallback DEFAULT_EXCHANGE."""
        out = {str(s).upper(): CFG.DEFAULT_EXCHANGE for s in symbols}
        try:
            from vnstock_data import Reference
            df = Reference().equity.list_by_exchange()
            if df is None or df.empty:
                return out
            sym_col = next((c for c in df.columns if str(c).lower() in ('symbol','ticker','code')), None)
            ex_col = next((c for c in df.columns if any(k in str(c).lower() for k in ('exchange','market','board'))), None)
            if sym_col is None or ex_col is None:
                return out
            for _, row in df[[sym_col, ex_col]].dropna().iterrows():
                sym = str(row[sym_col]).upper().strip()
                if sym in out:
                    out[sym] = VNMarketRules.normalize_exchange(row[ex_col])
        except Exception as e:
            log.warning(f'Không load được exchange map: {e}; fallback {CFG.DEFAULT_EXCHANGE}')
        return out

# ==============================================================
# M0-B: SECTOR ENGINE — Nhóm ngành (ICB) + Xu hướng nhóm ngành
# ==============================================================
#
# Nguồn: vnstock_data.Reference().equity.list_by_industry() (chuẩn ICB,
# 4 cấp). Dùng ICB_LEVEL=2 (ngành cấp 2) — đủ khái quát để nhóm cổ phiếu,
# không vụn như cấp 3/4.
#
# Xu hướng ngành = trung bình LOG-RETURN của các mã CÙNG NGÀNH trong danh
# sách đang phân tích (không phải toàn thị trường). Dùng log-return vì nó
# quy các mã về cùng "tốc độ tăng trưởng" (%), so sánh công bằng giữa các
# mã có thị giá khác nhau, và cộng dồn được qua nhiều phiên (sum log-ret).
# ==============================================================

class SectorEngine:
    _icb_cache = None  # {symbol: icb_name} — cache 1 lần / process
    _watchlist_cache = None  # {symbol: ten_nhom_watchlist} — fallback thủ công

    ICB_LEVEL = 2
    TREND_LOOKBACKS = {'5D': 5, '20D': 20, '60D': 60}  # phiên

    # Watchlist phân nhóm ngành thủ công (Watchlist.xlsx) — dùng làm FALLBACK
    # khi ICB map không có mã (hoặc load ICB lỗi).
    #
    # PURE_SECTORS: các ngành ICB-kiểu chuẩn, gần như KHÔNG trùng mã nhau
    # (chỉ còn vài mã trùng nhẹ do doanh nghiệp đa ngành thật sự, ví dụ PVI
    # là bảo hiểm thuộc PVN nên nằm cả Dầu khí lẫn Bảo hiểm) — dùng để tính
    # Tỷ trọng (%) GTGD toàn thị trường, KHÔNG bị double-count.
    #
    # THEMATIC_GROUPS: các nhóm chuyên đề cắt ngang (theo hệ sinh thái/quỹ
    # theo dõi riêng), CỐ Ý trùng mã với PURE_SECTORS và với nhau (vd FPT
    # vừa ở Công nghệ vừa ở FPT nhóm vừa ở Bán lẻ) — chỉ để theo dõi riêng,
    # KHÔNG cộng vào tổng GTGD thị trường vì sẽ đếm trùng.
    PURE_SECTORS = {
        'Ngân hàng (BANK)': ['LPB','VIB','NVB','BID','STB','HDB','TPB','MBB','ACB','TCB','VPB','VCB','CTG','NAB','EVF','ABB','MSB','EIB','SSB','BVB','SHB','VAB','BAB','OCB','KLB'],
        'Bất động sản (BĐS)': ['DPG','CII','DIG','KBC','IDC','CEO','HHS','VGC','DXG','HDG','TCH','HQC','NHA','SGR','DXS','QCG','PDR','HDC','KDH','KHG','VPI','NTL','NVL','VC7','TIG','SCR','CKG','NLG','AGG','L14','HPX','DC4','TAL'],
        'Chứng khoán': ['BVS','SBS','MBS','BSI','VCI','AAS','SHS','PSI','CTS','VFS','SSI','ABW','AGR','VIG','IPA','FTS','VIX','VCK','VDS','VPX','ORS','TCX','DSE','APG'],
        'Thép': ['NKG','VGS','HSG','TVN','TLH','HPG','HSV'],
        'XD-VLXD-Đầu tư công': ['VCG','FCN','G36','HHV','LCG','C4G','DPG','CII','HBC','CTI','VLB','KSB','MSR','SZC','BMP','CTD','IJC','VCS','BCC','C32','GDA','EVG','DC4','HT1','NTP'],
        'Dầu khí': ['PVD','PLC','PVS','PVC','PVT','LAS','PVB','GAS','OIL','BSR','PVI','PXL'],
        'Phân bón - Hoá chất': ['DGC','PHR','GVR','DPM','LAS','DCM','CSV','VFG','BFC','DHB','DDV'],
        'Cảng biển': ['PVP','PVT','VSC','SGP','MVN','GMD','VOS','TCL','DVP','PHP','HAH'],
        'Dệt may': ['VGT','TNG','MSH','HDM','GIL','TCM'],
        'Không thiết yếu': ['STH','DHC','VEA','PAN','DLG','YEG'],
        'Công nghệ': ['ELC','VTP','CTR','CMG','FPT','VGI'],
        'Hàng không': ['HVN','SCS','ACV','VJC','ATS'],
        'Điện': ['PC1','GEE','TV2','HDG','REE','TV1','GEX','GEG','NT2','POW','QTP','DRL','PPC'],
        'Bảo hiểm': ['BVH','BIC','PVI','MIG'],
        'Dược phẩm': ['DVM','DCL'],
        'Bán lẻ': ['PET','DGW','MSN','FRT','MWG','TLG','VNM','PNJ','FPT','HAX','SAB'],
        'Thuỷ sản': ['ASM','FMC','IDI','ACL','VHC','CMX','ANV'],
    }

    THEMATIC_GROUPS = {
        'BĐS Khu công nghiệp': ['PHR','KBC','CTI','GVR','IDC','VGC','SZC','CTD','SIP','IJC','LHG','BCM','TIP','NTC'],
        'Viettel': ['VTK','VTP','CTR','VGI'],
        'FPT nhóm': ['FRT','FOC','FTS','FPT','FOX'],
        'Vingroup': ['VPL','VRE','VHM','VIC'],
        'Tự doanh Tuấn Mượt': ['PC1','PET','GEE','VSC','VGC','MHC','GEX','EVF','BSR','EIB','GEL','VIX','PXL','HAH'],
        'List CMSC': ['PVD','PVS','HVN','PGS','PVT','SFG','PVB','GVR','DPM','LAS','MBB','DCM','PLX','GAS','OIL','ACV','BSR','PAC','CSV','POW','DPR','TRC','DHG','VNM','FPT','FOX','DDV','RTB','BRR'],
    }

    # Giữ WATCHLIST_SECTORS gộp cả 2 để tương thích code cũ (đọc mã 227 mã
    # phục vụ fetch_ohlcv) — CHỈ dùng compute_sector_cashflow() riêng với
    # PURE_SECTORS / THEMATIC_GROUPS, không dùng dict gộp này để tính GTGD.
    WATCHLIST_SECTORS = {**PURE_SECTORS, **THEMATIC_GROUPS}

    @classmethod
    def _load_icb_map(cls):
        if cls._icb_cache is not None:
            return cls._icb_cache
        try:
            from vnstock_data import Reference
            ref = Reference()
            df = ref.equity.list_by_industry()
            df = df[df['icb_level'] == cls.ICB_LEVEL][['symbol', 'icb_name']].dropna()
            df = df.drop_duplicates(subset='symbol')
            cls._icb_cache = dict(zip(df['symbol'], df['icb_name']))
            log.info(f"  [SECTOR] Đã load ICB map (level {cls.ICB_LEVEL}): {len(cls._icb_cache)} mã")
        except Exception as e:
            log.warning(f"  [SECTOR] Không load được ICB map: {e} — dùng 'Không xác định'")
            cls._icb_cache = {}
        return cls._icb_cache

    @classmethod
    def _load_watchlist_map(cls):
        """{symbol: ten_nhom} từ WATCHLIST_SECTORS — mã xuất hiện ở nhóm liệt kê
        trước sẽ được giữ nếu trùng nhiều nhóm."""
        if cls._watchlist_cache is not None:
            return cls._watchlist_cache
        m = {}
        for sector_name, syms in cls.WATCHLIST_SECTORS.items():
            for s in syms:
                m.setdefault(s, sector_name)
        cls._watchlist_cache = m
        return m

    @classmethod
    def map_symbols(cls, symbols):
        """{symbol: ten_nganh} cho danh sách mã.

        Ưu tiên ICB (vnstock_data); nếu mã không có trong ICB map (hoặc ICB
        load lỗi), fallback sang phân nhóm thủ công từ Watchlist.
        """
        icb = cls._load_icb_map()
        wl = cls._load_watchlist_map()
        return {s: icb.get(s) or wl.get(s, 'Không xác định') for s in symbols}

    @staticmethod
    def log_returns(price_series):
        """ln(P_t / P_t-1) — quy về tốc độ tăng trưởng, cộng dồn được qua thời gian."""
        return np.log(price_series / price_series.shift(1)).dropna()

    @classmethod
    def sector_trend(cls, data: dict, sector_map: dict):
        """Xu hướng từng nhóm ngành dựa trên log-return trung bình của các mã cùng ngành.

        data: {symbol: ohlcv_df}; sector_map: {symbol: ten_nganh}
        Trả về {ten_nganh: {...}}
        """
        groups = {}
        for sym, sector in sector_map.items():
            df = data.get(sym)
            if df is None or len(df) < 30:
                continue
            groups.setdefault(sector, []).append(sym)

        result = {}
        for sector, syms in groups.items():
            rets = {s: cls.log_returns(data[s]['close']) for s in syms}
            ret_df = pd.DataFrame(rets).dropna(how='all')
            if ret_df.empty:
                continue
            avg_ret = ret_df.mean(axis=1)  # log-return trung bình ngành / phiên

            out = {'symbols': syms, 'n_symbols': len(syms)}
            for label, w in cls.TREND_LOOKBACKS.items():
                if len(avg_ret) >= w:
                    cum = avg_ret.tail(w).sum()
                    out[f'ret_{label}_pct'] = round(float((np.exp(cum) - 1) * 100), 2)
                else:
                    out[f'ret_{label}_pct'] = None

            m = out.get('ret_20D_pct')
            if m is None:
                m = out.get('ret_5D_pct') or 0
            if m > 3: tl = '🟢 TĂNG MẠNH'
            elif m > 0.5: tl = '🟢 TĂNG'
            elif m > -0.5: tl = '⚪ ĐI NGANG'
            elif m > -3: tl = '🔴 GIẢM'
            else: tl = '🔴 GIẢM MẠNH'
            out['trend_label'] = tl
            out['avg_daily_log_ret'] = round(float(avg_ret.mean()), 6)
            result[sector] = out
        return result

# ==============================================================
# M0-C: CROSS-CORRELATION ENGINE — Tương quan chéo + VAR theo ngành
# ==============================================================
#
# 1) Log-return matrix theo ngành (quy về tốc độ tăng trưởng để so sánh).
# 2) Lagged CCF (lag -2..+2 phiên) cho từng cặp mã -> xác định mã DẪN DẮT
#    (leading) và mã ĐI THEO (lagging): corr(x_t, y_{t+lag}).
#    lag>0: x dẫn dắt y (y đi sau x); lag<0: y dẫn dắt x.
# 3) VAR (Vector AutoRegression) riêng cho từng ngành + Granger causality
#    để trả lời "mã X tăng thì các mã còn lại có tăng theo không, và có ý
#    nghĩa thống kê hay không".
# ==============================================================

class CrossCorrelationEngine:
    MAX_LAG = 2
    MIN_OBS = 40          # tối thiểu số quan sát chung để tính CCF/VAR đáng tin
    MAX_SYMS_VAR = 8      # giới hạn số mã/VAR để tránh overfit (bậc tự do)
    VAR_MAXLAGS = 5

    @staticmethod
    def _build_return_matrix(data: dict, symbols: list):
        rets = {}
        for s in symbols:
            df = data.get(s)
            if df is not None and len(df) >= 30:
                rets[s] = SectorEngine.log_returns(df['close'])
        if len(rets) < 2:
            return None
        mat = pd.DataFrame(rets).dropna()
        return mat if len(mat) >= CrossCorrelationEngine.MIN_OBS else None

    @classmethod
    def lagged_ccf(cls, ret_mat: pd.DataFrame):
        """CCF lag -2..+2 cho mọi cặp mã. Trả về (pair_ccf, lead_lag_summary)."""
        cols = ret_mat.columns.tolist()
        pair_ccf, lead_lag_summary = {}, []
        for i, x in enumerate(cols):
            for y in cols[i+1:]:
                lags = {}
                for lag in range(-cls.MAX_LAG, cls.MAX_LAG + 1):
                    joined = pd.concat([ret_mat[x], ret_mat[y].shift(-lag)], axis=1).dropna()
                    if len(joined) < 20:
                        lags[lag] = None; continue
                    c = joined.iloc[:, 0].corr(joined.iloc[:, 1])
                    lags[lag] = round(float(c), 4) if pd.notna(c) else None
                pair_ccf[(x, y)] = lags
                valid = {k: v for k, v in lags.items() if v is not None}
                if valid:
                    best_lag = max(valid, key=lambda k: abs(valid[k]))
                    best_corr = valid[best_lag]
                    if best_lag > 0: relation = f'{x} dẫn dắt {y} (trễ {best_lag} phiên)'
                    elif best_lag < 0: relation = f'{y} dẫn dắt {x} (trễ {abs(best_lag)} phiên)'
                    else: relation = f'{x} và {y} biến động ĐỒNG THỜI'
                    lead_lag_summary.append({'pair': f'{x}-{y}', 'best_lag': best_lag,
                                              'best_corr': best_corr, 'relation': relation})
        return pair_ccf, lead_lag_summary

    @staticmethod
    def correlation_matrix(ret_mat: pd.DataFrame):
        return ret_mat.corr().round(4)

    @classmethod
    def fit_var(cls, ret_mat: pd.DataFrame):
        """VAR theo ngành (giới hạn MAX_SYMS_VAR mã có variance cao nhất) + Granger causality."""
        try:
            from statsmodels.tsa.api import VAR
        except ImportError:
            return {'error': 'pip install statsmodels'}

        cols = ret_mat.columns.tolist()
        if len(cols) > cls.MAX_SYMS_VAR:
            cols = ret_mat.var().sort_values(ascending=False).head(cls.MAX_SYMS_VAR).index.tolist()
        sub = ret_mat[cols].dropna()
        if len(sub) < cls.MIN_OBS or len(cols) < 2:
            return {'error': 'Không đủ dữ liệu cho VAR'}
        try:
            model = VAR(sub)
            maxlags = max(min(cls.VAR_MAXLAGS, len(sub)//10 or 1), 1)
            sel = model.select_order(maxlags=maxlags)
            order = sel.aic if sel.aic and sel.aic > 0 else 1
            res = model.fit(order)
            granger = {}
            for y in cols:
                for x in cols:
                    if x == y: continue
                    try:
                        gc = res.test_causality(y, [x], kind='f')
                        granger[f'{x}->{y}'] = {'p_value': round(float(gc.pvalue), 4),
                                                 'significant': bool(gc.pvalue < 0.05)}
                    except Exception:
                        continue
            return {'symbols': cols, 'order': int(order), 'aic': round(float(res.aic), 4),
                     'granger_causality': granger}
        except Exception as e:
            return {'error': str(e)}

    @classmethod
    def analyze_sector(cls, data: dict, symbols: list):
        """Chạy full: corr matrix + lagged CCF + VAR cho 1 nhóm ngành."""
        ret_mat = cls._build_return_matrix(data, symbols)
        if ret_mat is None or ret_mat.shape[1] < 2:
            return {'error': 'Không đủ mã/dữ liệu chung trong ngành để tính tương quan',
                    'avg_pairwise_corr': None}
        corr = cls.correlation_matrix(ret_mat)
        _, lead_lag = cls.lagged_ccf(ret_mat)
        var_res = cls.fit_var(ret_mat)
        vals = corr.values; n = vals.shape[0]
        off_diag = vals[~np.eye(n, dtype=bool)]
        avg_corr = round(float(np.nanmean(off_diag)), 4) if len(off_diag) else None
        return {
            'symbols': ret_mat.columns.tolist(), 'n_obs': len(ret_mat),
            'avg_pairwise_corr': avg_corr, 'corr_matrix': corr.to_dict(),
            'lead_lag': sorted(lead_lag, key=lambda d: abs(d['best_corr'] or 0), reverse=True),
            'var': var_res,
        }

# ==============================================================
# M1: DISTRIBUTION ANALYSIS
# ==============================================================

class DistributionAnalyzer:
    @staticmethod
    def full_test(prices):
        from scipy import stats
        ret = prices.pct_change().dropna().values
        if len(ret) < 30: return {'error': 'Need 30+ obs'}
        r = {'n': len(ret), 'mean': round(float(np.mean(ret)),6), 'std': round(float(np.std(ret)),6),
             'skewness': round(float(stats.skew(ret)),4), 'excess_kurtosis': round(float(stats.kurtosis(ret)),4)}
        jb_s, jb_p = stats.jarque_bera(ret)
        sw_s, sw_p = stats.shapiro(ret[:5000])
        ad = stats.anderson(ret, dist='norm')
        dp_s, dp_p = stats.normaltest(ret) if len(ret) >= 20 else (0, 1)
        mu, sig = np.mean(ret), np.std(ret)
        ks_s, ks_p = stats.kstest(ret, 'norm', args=(mu, sig))
        tests = {
            'jarque_bera': {'stat': round(float(jb_s),4), 'p': round(float(jb_p),6), 'reject': bool(jb_p < CFG.NORMALITY_ALPHA)},
            'shapiro_wilk': {'stat': round(float(sw_s),6), 'p': round(float(sw_p),6), 'reject': bool(sw_p < CFG.NORMALITY_ALPHA)},
            'anderson_darling': {'stat': round(float(ad.statistic),4), 'crit_5pct': round(float(ad.critical_values[2]),4),
                                 'reject': bool(ad.statistic > ad.critical_values[2])},
            'dagostino': {'stat': round(float(dp_s),4), 'p': round(float(dp_p),6), 'reject': bool(dp_p < CFG.NORMALITY_ALPHA)},
            'ks_test': {'stat': round(float(ks_s),4), 'p': round(float(ks_p),6), 'reject': bool(ks_p < CFG.NORMALITY_ALPHA)},
        }
        r['tests'] = tests
        ek = stats.kurtosis(ret)
        p2 = np.mean(np.abs(ret) > 2*sig); n2 = 2*(1-stats.norm.cdf(2))
        p3 = np.mean(np.abs(ret) > 3*sig); n3 = 2*(1-stats.norm.cdf(3))
        try:
            ar = np.abs(ret); th = np.quantile(ar, 0.95); exc = ar[ar > th]
            hill = len(exc)/np.sum(np.log(exc/th)) if len(exc)>=5 and np.sum(np.log(exc/th))>0 else None
        except: hill = None
        r['fat_tail'] = {
            'excess_kurtosis': round(float(ek),4),
            'severity': 'EXTREME' if ek>10 else 'STRONG' if ek>5 else 'MODERATE' if ek>1 else 'MILD' if ek>0 else 'THIN',
            'tail_ratio_2sigma': round(float(p2/n2),2) if n2>0 else 1,
            'tail_ratio_3sigma': round(float(p3/n3),2) if n3>0 else 1,
            'prob_beyond_3sigma_pct': round(float(p3)*100,3),
            'hill_index': round(float(hill),3) if hill else None,
        }
        norm_p = stats.norm.fit(ret); norm_aic = 4 - 2*np.sum(stats.norm.logpdf(ret, *norm_p))
        t_p = stats.t.fit(ret); t_aic = 6 - 2*np.sum(stats.t.logpdf(ret, *t_p))
        lap_p = stats.laplace.fit(ret); lap_aic = 4 - 2*np.sum(stats.laplace.logpdf(ret, *lap_p))
        fits = {'normal': norm_aic, 'student_t': t_aic, 'laplace': lap_aic}
        best = min(fits, key=fits.get)
        r['best_fit'] = {'winner': best, 'aic': {k: round(v,2) for k,v in fits.items()}, 't_df': round(float(t_p[0]),2)}
        reject_n = sum(t['reject'] for t in tests.values())
        r['verdict'] = {
            'reject_count': reject_n, 'is_gaussian': reject_n <= 1,
            'conclusion': ('Returns GAN Gaussian' if reject_n <= 1 else
                          'Returns KHONG Gaussian - can models robust' if reject_n <= 3 else
                          'Returns RAT KHAC Gaussian - fat-tail manh'),
            'model_rec': ('Standard OK' if reject_n <= 1 else
                         'Student-t VaR, GARCH-t, Historical Sim' if reject_n <= 3 else
                         'EVT, GPD tail, non-parametric'),
        }
        return r

# ==============================================================
# M2: RETURN STATISTICS
# ==============================================================

class StatEngine:
    @staticmethod
    def returns(prices):
        if len(prices) < 20: return {}
        prices = prices.astype(float).dropna()
        ret = prices.pct_change().dropna()
        if ret.empty: return {}
        mr, sr = ret.mean(), ret.std(ddof=1)
        ann_mean = mr * 252
        av = sr * np.sqrt(252)
        years = max(len(ret) / 252.0, 1/252)
        total_ret = prices.iloc[-1] / prices.iloc[0] - 1 if prices.iloc[0] > 0 else 0
        cagr = (prices.iloc[-1] / prices.iloc[0]) ** (1/years) - 1 if prices.iloc[0] > 0 else 0
        conf = CFG.CONFIDENCE_LEVEL
        var_p = np.percentile(ret, (1-conf)*100)
        cvar_p = ret[ret <= var_p].mean() if len(ret[ret <= var_p]) else var_p
        up, dn = ret[ret > 0], ret[ret < 0]
        up_day_ratio = len(up) / len(ret)
        avg_up = up.mean() if len(up) else 0
        avg_down = abs(dn.mean()) if len(dn) else 0
        gross_up = up.sum(); gross_down = abs(dn.sum())
        daily_profit_factor = gross_up / gross_down if gross_down > 0 else None
        ex = ann_mean - CFG.RISK_FREE_RATE
        sharpe = ex / av if av > 0 else 0
        downside = np.sqrt(np.mean(np.minimum(ret.values, 0.0) ** 2)) * np.sqrt(252)
        sortino = ex / downside if downside > 0 else 0
        wealth = (1 + ret).cumprod(); dd = wealth / wealth.cummax() - 1
        mdd = dd.min()
        calmar = cagr / abs(mdd) if mdd < 0 else 0
        kurt = ret.kurtosis()  # excess kurtosis
        return {
            'mean_daily_pct': round(mr*100,4), 'ann_mean_return_pct': round(ann_mean*100,2),
            'ann_return_pct': round(cagr*100,2), 'total_return_pct': round(total_ret*100,2),
            'ann_vol_pct': round(av*100,2), 'skewness': round(ret.skew(),3),
            'kurtosis': round(kurt,3), 'is_fat_tail': bool(kurt > 1),
            f'VaR_{int(conf*100)}': round(var_p*100,3),
            f'CVaR_{int(conf*100)}': round(cvar_p*100,3) if not pd.isna(cvar_p) else None,
            'max_dd_pct': round(mdd*100,2), 'sharpe': round(sharpe,3),
            'sortino': round(sortino,3), 'calmar': round(calmar,3),
            'win_rate_pct': round(up_day_ratio*100,1), 'up_day_ratio_pct': round(up_day_ratio*100,1),
            'profit_factor': round(daily_profit_factor,3) if daily_profit_factor is not None else None,
            'daily_profit_factor': round(daily_profit_factor,3) if daily_profit_factor is not None else None,
            'avg_win_pct': round(avg_up*100,3), 'avg_loss_pct': round(avg_down*100,3),
            'metric_scope': 'DAILY_PRICE_RETURNS_NOT_TRADES', 'n': len(ret)}

    @staticmethod
    def vol_regime(prices):
        ret = prices.pct_change().dropna()
        v10 = ret.rolling(CFG.VOL_WINDOW_SHORT).std()*np.sqrt(252)
        v20 = ret.rolling(CFG.VOL_WINDOW_MED).std()*np.sqrt(252)
        v60 = ret.rolling(CFG.VOL_WINDOW_LONG).std()*np.sqrt(252)
        c10,c20,c60 = v10.iloc[-1],v20.iloc[-1],v60.iloc[-1]
        vr = c10/c60 if c60>0 else 1
        vp = (v20<c20).mean()*100 if len(v20)>=60 else 50
        rg = 'CONTRACTION' if vr<0.7 else ('EXPANSION' if vr>1.3 else 'NORMAL')
        return {'vol_10d': round(c10*100,2) if not pd.isna(c10) else None,
                'vol_20d': round(c20*100,2) if not pd.isna(c20) else None,
                'vol_60d': round(c60*100,2) if not pd.isna(c60) else None,
                'vol_ratio': round(vr,3), 'vol_pctile': round(vp,1), 'regime': rg}

    @staticmethod
    def autocorr(prices, max_lag=10):
        ret = prices.pct_change().dropna()
        if len(ret)<max_lag+10: return {'behavior': 'INSUFFICIENT'}
        ac = {f'lag_{i}': round(ret.autocorr(lag=i),4) for i in range(1, max_lag+1)}
        avg = np.mean([ac.get(f'lag_{i}',0) for i in range(1,4)])
        th = 2/np.sqrt(len(ret))
        beh = 'MOMENTUM' if avg>th else ('MEAN_REVERSION' if avg<-th else 'RANDOM_WALK')
        return {'ac': ac, 'avg_short': round(avg,4), 'threshold': round(th,4), 'behavior': beh}

# ==============================================================
# M3: ARIMA
# ==============================================================

class ARIMAEngine:
    @staticmethod
    def fit(prices, max_p=4, max_q=4):
        try:
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.stattools import adfuller
        except ImportError: return {'error': 'pip install statsmodels'}
        ret = prices.pct_change().dropna()
        if len(ret)<60: return {'error': 'Need 60+ obs'}
        adf_s, adf_p = adfuller(ret.values, maxlag=20)[:2]
        is_stat = adf_p < 0.05
        r = {'stationarity': {'adf_stat': round(adf_s,4), 'adf_p': round(adf_p,6), 'stationary': is_stat}}
        best_aic, best_order, best_m = np.inf, (0,0,0), None
        for p in range(0, max_p+1):
            for q in range(0, max_q+1):
                if p==0 and q==0: continue
                try:
                    m = ARIMA(ret.values, order=(p,0,q)).fit()
                    if m.aic < best_aic: best_aic, best_order, best_m = m.aic, (p,0,q), m
                except: pass
        if best_m is None: r['error'] = 'No ARIMA fit'; return r
        r['best'] = {'order': f'ARIMA{best_order}', 'aic': round(best_aic,2), 'bic': round(best_m.bic,2)}
        coefs = {}
        for nm, val, pv in zip(best_m.param_names, best_m.params, best_m.pvalues):
            coefs[nm] = {'val': round(float(val),6), 'p': round(float(pv),4), 'sig': bool(pv<0.05)}
        r['coefficients'] = coefs
        fc = best_m.forecast(steps=CFG.FORECAST_HORIZON)
        lp = prices.iloc[-1]; fp = [lp]
        for rt in fc: fp.append(fp[-1]*(1+rt))
        r['forecast'] = {'prices': [round(p,0) for p in fp[1:]], 'returns_pct': [round(x*100,3) for x in fc]}
        return r

# ==============================================================
# M4: GARCH / EGARCH
# ==============================================================

class GARCHEngine:
    @staticmethod
    def fit(prices, p=1, q=1):
        try:
            from arch import arch_model
        except ImportError: return {'error': 'pip install arch'}
        ret = prices.pct_change().dropna()*100
        if len(ret)<100: return {'error': 'Need 100+ obs'}
        r = {}
        try:
            m = arch_model(ret, vol='Garch', p=p, q=q, dist='t').fit(disp='off')
            alpha = float(m.params.get('alpha[1]',0)); beta = float(m.params.get('beta[1]',0))
            persist = alpha+beta
            fc = m.forecast(horizon=CFG.FORECAST_HORIZON)
            cvol = np.sqrt(fc.variance.iloc[-1].values)
            r['garch'] = {
                'model': f'GARCH({p},{q})-t', 'aic': round(float(m.aic),2),
                'alpha': round(alpha,4), 'beta': round(beta,4), 'persistence': round(persist,4),
                'half_life': round(np.log(0.5)/np.log(persist),1) if 0<persist<1 else None,
                'persist_label': ('IGARCH-like' if persist>0.97 else 'High' if persist>0.9 else 'Moderate' if persist>0.7 else 'Low'),
                'current_vol_pct': round(float(m.conditional_volatility.iloc[-1]),3),
                'forecast_vol': [round(float(v),3) for v in cvol],
                'shock_sensitivity': 'HIGH' if alpha>0.15 else ('MODERATE' if alpha>0.08 else 'LOW'),
                # Unrounded values for path-wise Monte Carlo recursion. The
                # model is fitted to percentage returns, so omega is in pct^2.
                'simulation_params': {
                    'omega_pct2': float(m.params.get('omega', np.nan)),
                    'alpha': alpha,
                    'beta': beta,
                },
            }
            params = {}
            for nm in m.params.index:
                params[nm] = {'val': round(float(m.params[nm]),6), 'p': round(float(m.pvalues[nm]),4)}
            r['garch']['params'] = params
        except Exception as e: r['garch'] = {'error': str(e)}
        try:
            em = arch_model(ret, vol='EGARCH', p=1, o=1, q=1, dist='t').fit(disp='off')
            gamma = float(em.params.get('gamma[1]',0))
            r['egarch'] = {'aic': round(float(em.aic),2), 'gamma': round(gamma,4),
                           'leverage': bool(gamma<0),
                           'note': 'Tin xau tang vol MANH hon tin tot' if gamma<-0.05 else 'Leverage effect yeu'}
        except Exception as e: r['egarch'] = {'error': str(e)}
        return r

# ==============================================================
# M5: HMM REGIME — V4: MULTIVARIATE + MULTI-INIT + BIC SELECT
# ==============================================================
#
# FIX #2: Single-feature HMM too noisy for position trading.
#   -> 4 features: [return, rolling_vol, volume_ratio, short_momentum]
#   -> N random inits, select best model by BIC
#   -> Robust state labeling via return-dimension ranking
# ==============================================================

class HMMEngine:
    @staticmethod
    def _build_features(df):
        """Build multivariate feature matrix for HMM.
        4 features standardized to prevent scale dominance:
          1. Daily return         — directional
          2. Rolling vol (10d)    — vol regime
          3. Volume ratio (5/20)  — participation
          4. Short momentum (5d)  — trend smoothing
        """
        close = df['close']; volume = df['volume']
        ret = close.pct_change()
        rolling_vol = ret.rolling(CFG.HMM_VOL_WINDOW).std()
        vol_ma5 = volume.rolling(5).mean()
        vol_ma20 = volume.rolling(20).mean()
        vol_ratio = vol_ma5 / vol_ma20.replace(0, np.nan)
        short_mom = ret.rolling(CFG.HMM_MOM_WINDOW).mean()
        feat_df = pd.DataFrame({
            'return': ret, 'rolling_vol': rolling_vol,
            'vol_ratio': vol_ratio, 'short_mom': short_mom,
        }).dropna()
        if len(feat_df) < 100: return None, None
        values = feat_df.values.copy()
        means = values.mean(axis=0); stds = values.std(axis=0)
        stds[stds == 0] = 1.0
        values_z = (values - means) / stds
        return values_z, feat_df.index

    @staticmethod
    def fit(df, n=3, symbol=''):
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError: return HMMEngine._fallback(df)
        features, feat_idx = HMMEngine._build_features(df)
        if features is None: return HMMEngine._fallback(df)
        try:
            best_model, best_bic = None, np.inf
            n_samples, n_features = features.shape
            # V4.1: Symbol-based seed for stability across runs
            sym_seed = int.from_bytes(hashlib.blake2b(str(symbol).encode('utf-8'), digest_size=4).digest(), 'little') if symbol else 0
            for init_i in range(CFG.HMM_N_INITS):
                try:
                    m = GaussianHMM(n_components=n, covariance_type='full',
                                    n_iter=CFG.HMM_N_ITER, random_state=sym_seed + init_i, tol=1e-4)
                    m.fit(features)
                    ll = m.score(features)
                    n_params = (n*(n-1) + n*n_features + n*n_features*(n_features+1)//2 + (n-1))
                    bic = -2*ll + n_params * np.log(n_samples)
                    if bic < best_bic: best_bic, best_model = bic, m
                except: continue
            if best_model is None: return HMMEngine._fallback(df)
            m = best_model; states = m.predict(features)
            means = m.means_; ret_means = means[:, 0]
            order = np.argsort(ret_means)
            names = {order[0]: 'BEAR', order[1]: 'SIDEWAY', order[2]: 'BULL'}
            emj = {'BEAR': '🔴', 'SIDEWAY': '🟡', 'BULL': '🟢'}
            info = {}
            for i in range(n):
                nm = names.get(i, '?')
                state_mask = states == i
                state_ret = df['close'].pct_change().reindex(feat_idx).values[state_mask]
                state_ret = state_ret[~np.isnan(state_ret)]
                info[nm] = {
                    'mean_daily_pct': round(float(np.mean(state_ret))*100, 4) if len(state_ret)>0 else 0,
                    'ann_ret_pct': round(float(np.mean(state_ret))*252*100, 2) if len(state_ret)>0 else 0,
                    'ann_vol_pct': round(float(np.std(state_ret))*np.sqrt(252)*100, 2) if len(state_ret)>1 else 0,
                    'pct_time': round(float(np.mean(state_mask))*100, 1),
                    'avg_vol_regime': round(float(means[i, 1]), 2),
                    'avg_volume_activity': round(float(means[i, 2]), 2),
                }
            cs = states[-1]; cn = names.get(cs, '?')
            probs = m.predict_proba(features)[-1]
            np_d = {names.get(i,'?'): round(float(probs[i])*100,1) for i in range(n)}
            trans = {}
            for i in range(n):
                f = names.get(i,'?')
                trans[f] = {names.get(j,'?'): round(float(m.transmat_[i,j])*100,1) for j in range(n)}
            return {
                'current': f"{emj.get(cn,'')} {cn}", 'prob_pct': round(float(max(probs))*100,1),
                'state_probs': np_d, 'states': info, 'transitions': trans,
                'recent_20': [names.get(s,'?') for s in states[-20:]],
                '_state_dates': feat_idx.tolist(),
                '_state_labels': [names.get(s, 'SIDEWAY') for s in states],
                'warning': '⚠️ Regime co the doi' if max(probs)<0.6 else 'On dinh',
                'method': 'multivariate_hmm', 'n_features': features.shape[1],
                'n_inits': CFG.HMM_N_INITS, 'bic': round(best_bic, 2),
            }
        except Exception as e:
            log.warning(f"HMM fit error: {e}"); return HMMEngine._fallback(df)

    @staticmethod
    def _fallback(df):
        prices = df['close'] if isinstance(df, pd.DataFrame) else df
        ma20, ma50 = prices.rolling(20).mean(), prices.rolling(50).mean()
        p, m2, m5 = prices.iloc[-1], ma20.iloc[-1], ma50.iloc[-1]
        r20 = prices.iloc[-1]/prices.iloc[-20]-1 if len(prices)>=20 else 0
        st = '🟢 BULL' if p>m2>m5 and r20>0.02 else ('🔴 BEAR' if p<m2<m5 and r20<-0.02 else '🟡 SIDEWAY')
        return {'method': 'fallback', 'current': st, 'note': 'pip install hmmlearn cho HMM'}

# ==============================================================
# M6: ALPHA SIGNALS
# ==============================================================

class AlphaEngine:
    @staticmethod
    def extract(df, index_df=None):
        if len(df)<30: return {}
        c = df['close']; v = df['volume']
        sig = {}
        delta = c.diff(); gain = delta.where(delta>0,0).rolling(14).mean()
        loss = -delta.where(delta<0,0).rolling(14).mean()
        rsi = 100 - 100/(1+gain/loss.replace(0,1e-9)); crsi = rsi.iloc[-1]
        ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
        macd_h = (ema12-ema26) - (ema12-ema26).ewm(span=9).mean()
        roc10 = (c.iloc[-1]/c.iloc[-10]-1)*100 if len(c)>=10 else 0
        ms = 0
        if crsi>70: ms-=0.3
        elif crsi<30: ms+=0.3
        elif crsi>50: ms+=0.1
        else: ms-=0.1
        if roc10>5: ms+=0.3
        elif roc10<-5: ms-=0.3
        if macd_h.iloc[-1]>0: ms+=0.2
        else: ms-=0.2
        sig['momentum'] = {'rsi': round(float(crsi),1), 'roc_10d': round(roc10,2),
                          'macd_hist': round(float(macd_h.iloc[-1]),2), 'score': round(np.clip(ms,-1,1),3)}
        ma20 = c.rolling(20).mean(); std20 = c.rolling(20).std()
        z = (c.iloc[-1]-ma20.iloc[-1])/std20.iloc[-1] if std20.iloc[-1]>0 else 0
        bb_u = ma20.iloc[-1]+2*std20.iloc[-1]; bb_l = ma20.iloc[-1]-2*std20.iloc[-1]
        mrs = 0
        if z>2: mrs-=0.5
        elif z<-2: mrs+=0.5
        elif abs(z)>1: mrs-=0.2*np.sign(z)
        sig['mean_reversion'] = {'z_score': round(float(z),3), 'bb_upper': round(float(bb_u),0),
                                 'bb_lower': round(float(bb_l),0), 'score': round(np.clip(mrs,-1,1),3)}
        obv = (np.sign(c.diff())*v).cumsum()
        obv_up = obv.iloc[-1]>obv.iloc[-20] if len(obv)>=20 else True
        vr = v.rolling(5).mean().iloc[-1] / v.rolling(20).mean().iloc[-1] if v.rolling(20).mean().iloc[-1]>0 else 1
        vs = (0.3 if obv_up else -0.3) + (0.2 if vr>1.5 else (-0.2 if vr<0.5 else 0))
        sig['volume'] = {'obv_trend': 'UP' if obv_up else 'DOWN', 'vol_ratio_5_20': round(float(vr),2),
                        'score': round(np.clip(vs,-1,1),3)}
        if index_df is not None and 'close' in index_df.columns:
            # Deduplicate index BEFORE reindex to avoid ValueError
            index_df = index_df[~index_df.index.duplicated(keep='last')]
            c = c[~c.index.duplicated(keep='last')]
            ic = index_df['close'].reindex(c.index).ffill()
            if len(c)>=20 and len(ic)>=20:
                sr20 = c.pct_change(20).iloc[-1] - ic.pct_change(20).iloc[-1]
                sr, ir = c.pct_change().dropna(), ic.pct_change().reindex(c.pct_change().dropna().index).dropna()
                common = sr.index.intersection(ir.index)
                if len(common)>=20:
                    cov = np.cov(sr[common].values, ir[common].values)
                    beta = cov[0,1]/cov[1,1] if cov[1,1]>0 else 1
                    alpha_ann = (sr[common].mean()-beta*ir[common].mean())*252
                else: beta, alpha_ann = 1, 0
                sig['cross_sectional'] = {'rs_20d_pct': round(float(sr20)*100,2), 'beta': round(float(beta),3),
                                          'alpha_ann_pct': round(float(alpha_ann)*100,2)}
        # Composite — weights = 0.4 + 0.3 + 0.3 = 1.0
        cs = (0.4*sig.get('momentum',{}).get('score',0) +
              0.3*sig.get('mean_reversion',{}).get('score',0) +
              0.3*sig.get('volume',{}).get('score',0))
        sig['composite'] = {'alpha': round(cs,3),
                           'label': '🟢 BULLISH' if cs>0.3 else ('🔴 BEARISH' if cs<-0.3 else '⚪ NEUTRAL')}
        return sig

# ==============================================================
# M7: MARKET STRUCTURE — V4: SWING HIGH/LOW S/R
# ==============================================================
#
# FIX #4: Histogram S/R finds price frequency, not turning points.
#   -> argrelextrema for real swing high/low detection
#   -> Cluster nearby levels within ATR distance
#   -> Rank by touches + recency
# ==============================================================

class StructureEngine:
    @staticmethod
    def sr_levels(df, n=3, lb=120):
        from scipy.signal import argrelextrema
        if len(df) < 30: return {'supports': [], 'resistances': []}
        tail = df.tail(lb)
        highs = tail['high'].values; lows = tail['low'].values
        close_now = tail['close'].iloc[-1]
        order = CFG.SR_EXTREMA_ORDER
        high_idx = argrelextrema(highs, np.greater_equal, order=order)[0]
        low_idx = argrelextrema(lows, np.less_equal, order=order)[0]
        atr = RiskEng.atr(df)
        cluster_dist = max(atr * 0.5, close_now * 0.005)

        def cluster_levels(prices, indices, n_levels):
            if len(indices) == 0: return []
            n_bars = len(prices)
            levels = [(prices[idx], 1.0 + (idx / n_bars)) for idx in indices]
            levels.sort(key=lambda x: x[0])
            clusters = []; current = [levels[0]]
            for i in range(1, len(levels)):
                if levels[i][0] - current[-1][0] <= cluster_dist:
                    current.append(levels[i])
                else:
                    clusters.append(current); current = [levels[i]]
            clusters.append(current)
            result = []
            for cl in clusters:
                avg_p = np.mean([c[0] for c in cl])
                touches = len(cl)
                avg_rec = np.mean([c[1] for c in cl])
                result.append({'price': round(float(avg_p),0), 'str': round(float(touches*avg_rec),2), 'touches': touches})
            result.sort(key=lambda x: x['str'], reverse=True)
            return result[:n_levels]

        all_res = cluster_levels(highs, high_idx, n*2)
        all_sup = cluster_levels(lows, low_idx, n*2)
        supports = sorted([s for s in all_sup if s['price']<close_now], key=lambda x:x['price'], reverse=True)[:n]
        resistances = sorted([r for r in all_res if r['price']>=close_now], key=lambda x:x['price'])[:n]
        return {'price': round(close_now,0), 'supports': supports, 'resistances': resistances, 'method': 'swing_extrema'}

    @staticmethod
    def trend(df, period=20):
        if len(df)<period+1: return {}
        p = df.tail(period+1)['close'].values
        d = abs(p[-1]-p[0]); vol = np.sum(np.abs(np.diff(p)))
        er = d/vol if vol>0 else 0; slope = (p[-1]-p[0])/p[0]
        x = np.arange(len(p)); co = np.polyfit(x,p,1); tl = np.polyval(co,x)
        ss_r = np.sum((p-tl)**2); ss_t = np.sum((p-np.mean(p))**2)
        r2 = 1-ss_r/ss_t if ss_t>0 else 0
        lb = ('📈 UPTREND' if er>0.6 and slope>0 else '📉 DOWNTREND' if er>0.6 and slope<0 else
              '↗️ UP NHE' if er>0.3 and slope>0 else '↘️ DOWN NHE' if er>0.3 else '↔️ SIDEWAY')
        return {'er': round(er,4), 'r2': round(r2,4), 'slope_pct': round(slope*100,2), 'label': lb}

    @staticmethod
    def flow(df, lb=20):
        r = df.tail(lb)
        if len(r)<5: return {}
        h,l,c,v = r['high'].values, r['low'].values, r['close'].values, r['volume'].values.astype(float)
        rng = np.where((h-l)==0, 1e-9, h-l)
        mf = ((c-l)-(h-c))/rng
        cmf = np.sum(mf*v)/np.sum(v) if np.sum(v)>0 else 0
        bp = np.mean((c-l)/rng)
        lb2 = ('🟢 MUA' if cmf>0.1 and bp>0.6 else '🔴 BAN' if cmf<-0.1 else '🟡 THIEN MANH' if cmf>0 else '🟡 THIEN YEU')
        return {'cmf': round(cmf,4), 'buy_p': round(bp,3), 'label': lb2}

# ==============================================================
# M8: FORECASTING — V4: GARCH-INFORMED MONTE CARLO
# ==============================================================
#
# FIX #3: MC uses constant mu/sigma, ignoring GARCH forecast.
#   -> Accept garch_result, extract h-step forecast vols
#   -> Time-varying vol per simulation step
#   -> Fallback to historical sigma if GARCH unavailable
# ==============================================================

class FcastEngine:
    @staticmethod
    def ensemble(prices, garch_result=None, symbol='', vni_regime='NEUTRAL'):
        if len(prices)<30: return {}
        lr = np.log(prices/prices.shift(1)).dropna().values
        if len(lr) < 20: return {}
        q_lo, q_hi = np.quantile(lr, [0.02, 0.98])
        lr_robust = np.clip(lr, q_lo, q_hi)
        mu = np.mean(lr_robust) * CFG.FORECAST_DRIFT_SHRINK
        hist_sigma = np.std(lr_robust, ddof=1); lp = prices.iloc[-1]

        # V4.1 FIX #1: Per-symbol deterministic RNG instead of global seed
        sym_seed = int.from_bytes(hashlib.blake2b(str(symbol).encode('utf-8'), digest_size=4).digest(), 'little') if symbol else 42
        rng = np.random.RandomState(sym_seed)

        H = CFG.SWING_DEFAULT; LOCK = CFG.MIN_HOLD_SESSIONS; N = CFG.MONTE_CARLO_SIMS

        # V4.1 FIX #2: Extract GARCH forecast vols with sanity bounds
        garch_vols = None
        if garch_result is not None:
            gv = garch_result.get('garch', {}).get('forecast_vol', [])
            if gv and len(gv) >= H:
                garch_vols = np.array(gv[:H]) / 100.0
            elif gv:
                extended = list(gv) + [gv[-1]] * (H - len(gv))
                garch_vols = np.array(extended[:H]) / 100.0
            if garch_vols is not None:
                # Sanity: cap vol at 3x historical, floor at 0.3x historical
                vol_floor = hist_sigma * 0.3
                vol_cap = hist_sigma * 3.0
                garch_vols = np.clip(garch_vols, vol_floor, vol_cap)
                # V4.1: Crisis vol multiplier
                if vni_regime == 'CRISIS':
                    garch_vols = garch_vols * CFG.CRISIS_VOL_MULT

        # Monte Carlo — GARCH-aware; dùng Student-t innovations khi model có nu.
        nu = None
        try:
            nu = float(garch_result.get('garch',{}).get('params',{}).get('nu',{}).get('val'))
            if not np.isfinite(nu) or nu <= 2.1: nu = None
        except Exception:
            nu = None
        def draw_z(size):
            if nu is not None:
                return rng.standard_t(nu, size=size) / np.sqrt(nu/(nu-2))
            return rng.standard_normal(size)

        sims = np.zeros((N, H))
        recursive_garch = False
        sim_params = (garch_result or {}).get('garch', {}).get('simulation_params', {})
        try:
            omega = float(sim_params.get('omega_pct2')) / 10000.0
            alpha = float(sim_params.get('alpha'))
            beta = float(sim_params.get('beta'))
            recursive_garch = (
                garch_vols is not None and nu is not None
                and np.isfinite([omega, alpha, beta]).all()
                and omega >= 0 and alpha >= 0 and beta >= 0
                and alpha + beta < 1.0
            )
        except (TypeError, ValueError):
            recursive_garch = False

        if recursive_garch:
            crisis_mult = CFG.CRISIS_VOL_MULT if vni_regime == 'CRISIS' else 1.0
            omega *= crisis_mult ** 2
            sigma2 = np.full(N, garch_vols[0] ** 2)
            variance_floor = (hist_sigma * 0.3) ** 2
            variance_cap = (hist_sigma * 3.0 * crisis_mult) ** 2
            prev_price = np.full(N, lp, dtype=float)
            for step in range(H):
                sigma2 = np.clip(sigma2, variance_floor, variance_cap)
                innovation = np.sqrt(sigma2) * draw_z(N)
                next_price = prev_price * np.exp(mu + innovation)
                sims[:, step] = next_price
                sigma2 = omega + alpha * innovation ** 2 + beta * sigma2
                prev_price = next_price
        elif garch_vols is not None:
            # Compatibility fallback for cached/legacy results that do not
            # contain the parameters required for recursive simulation.
            for step in range(H):
                z = draw_z(N)
                if step == 0:
                    sims[:, step] = lp * np.exp(mu + garch_vols[step] * z)
                else:
                    sims[:, step] = sims[:, step-1] * np.exp(mu + garch_vols[step] * z)
        else:
            sigma_use = hist_sigma
            if vni_regime == 'CRISIS': sigma_use *= CFG.CRISIS_VOL_MULT
            for i in range(N):
                sims[i] = lp * np.exp(np.cumsum(mu + sigma_use * draw_z(H)))

        fin = sims[:,-1]
        mc = {'price': round(np.mean(fin),0), 'prob_up': round(np.mean(fin>lp)*100,1),
              'ci_lo': round(np.percentile(fin,2.5),0), 'ci_hi': round(np.percentile(fin,97.5),0),
              'vol_source': 'recursive GARCH-t' if recursive_garch else ('GARCH-t' if garch_vols is not None and nu is not None else ('GARCH-normal' if garch_vols is not None else 'historical')),
              'innovation_df': round(nu,2) if nu else None,
              'recursive_vol': recursive_garch}

        lock_prices = sims[:, :LOCK]; lock_mins = lock_prices.min(axis=1)
        lock_dd = (lock_mins - lp) / lp * 100
        lock_risk = {
            'max_dd_lock_pct': round(np.percentile(lock_dd, 5), 2),
            'avg_dd_lock_pct': round(np.mean(lock_dd[lock_dd<0]), 2) if np.any(lock_dd<0) else 0,
            'prob_loss_in_lock_pct': round(np.mean(lock_mins < lp)*100, 1),
            'prob_loss_gt_3pct': round(np.mean(lock_dd < -3)*100, 1),
            'lock_sessions': LOCK,
        }
        unlock_prices = sims[:, LOCK-1]
        unlock = {
            'price_at_unlock': round(np.mean(unlock_prices), 0),
            'prob_up_at_unlock': round(np.mean(unlock_prices > lp)*100, 1),
            'ci_lo_unlock': round(np.percentile(unlock_prices, 10), 0),
            'ci_hi_unlock': round(np.percentile(unlock_prices, 90), 0),
        }
        ma20 = prices.rolling(20).mean(); std20 = prices.rolling(20).std()
        z = (lp-ma20.iloc[-1])/std20.iloc[-1] if std20.iloc[-1]>0 else 0
        mr_target = ma20.iloc[-1]
        mr_sessions = min(H, max(LOCK+1, int(abs(z)*3)))
        mr = {'z': round(z,3), 'target': round(mr_target,0), 'dir': 'DOWN' if z>0 else 'UP', 'est_sessions': mr_sessions}
        rd = prices.pct_change().tail(H); avg = rd.mean()
        ms = avg/rd.std() if rd.std()>0 else 0
        mom = {'strength': round(ms,3), 'proj_pct': round(avg*H*100,2),
               'signal': 'BULLISH' if ms>0.3 else ('BEARISH' if ms<-0.3 else 'NEUTRAL')}

        # V4.1 FIX #7: Balanced weights — prevent auto-switch to MR after crash
        # In crisis/bear: momentum gets MORE weight (trend-following, not mean-reverting)
        msa = abs(ms); za = abs(z)
        if vni_regime in ('CRISIS', 'BEAR'):
            # In downtrend: trust momentum > MR, MC as base
            w = (0.35, 0.15, 0.50)
        elif msa > 0.5:
            w = (0.30, 0.20, 0.50)
        elif za > 1.5:
            # Only trust MR in non-crisis environments
            w = (0.30, 0.45, 0.25)
        else:
            w = (0.50, 0.25, 0.25)

        mc_r = (np.mean(fin)/lp-1)*100; mr_r = (mr_target-lp)/lp*100; mom_r = avg*H*100
        er = w[0]*mc_r + w[1]*mr_r + w[2]*mom_r
        uncapped_er = er

        # V4.1: Cap forecast in crisis regime
        if vni_regime == 'CRISIS':
            er = np.clip(er, -CFG.FORECAST_CAP_CRISIS * 3, CFG.FORECAST_CAP_CRISIS)
        elif vni_regime == 'BEAR':
            er = np.clip(er, -20, CFG.FORECAST_CAP_CRISIS * 2)

        cons = '📈 TANG' if er > 0.5 else ('📉 GIAM' if er < -0.5 else '↔️ TRUNG LAP')
        dirs = []
        if mc['prob_up']>55: dirs.append('UP')
        elif mc['prob_up']<45: dirs.append('DOWN')
        if mr['dir']=='UP': dirs.append('UP')
        else: dirs.append('DOWN')
        if mom['signal']=='BULLISH': dirs.append('UP')
        elif mom['signal']=='BEARISH': dirs.append('DOWN')
        er_dir = 'UP' if er > 0 else 'DOWN'
        agree = sum(1 for d in dirs if d == er_dir)
        conf = agree/len(dirs)*100 if dirs else 33
        lock_safe = lock_risk['prob_loss_gt_3pct'] < 20
        signal_strong = abs(er) > 2 and conf >= 66

        # V4.1 FIX #3: Timing MUST respect VNI regime — circuit breaker
        if vni_regime == 'CRISIS':
            timing = '🔴 KHÔNG VÀO — CRISIS: thị trường đang trong trạng thái khẩn cấp'
        elif vni_regime == 'BEAR':
            # In BEAR: never VÀO NGAY, at most CHỜ PULLBACK
            if signal_strong and lock_safe:
                timing = '🟡 CHỜ PULLBACK — tín hiệu tốt nhưng VNI đang BEAR'
            elif signal_strong:
                timing = '🟡 CHỜ PULLBACK — tín hiệu tốt nhưng lock risk cao + VNI BEAR'
            elif lock_safe:
                timing = '🟡 THEO DÕI — lock an toàn nhưng tín hiệu yếu + VNI BEAR'
            else:
                timing = '🔴 KHÔNG VÀO — lock risk cao + tín hiệu yếu + VNI BEAR'
        else:
            # Original logic for NEUTRAL/BULL/WEAK
            if signal_strong and lock_safe:
                timing = '🟢 VÀO NGAY — lock risk thấp, tín hiệu mạnh'
            elif signal_strong and not lock_safe:
                timing = '🟡 CHỜ PULLBACK — tín hiệu tốt nhưng lock risk cao'
            elif not signal_strong and lock_safe:
                timing = '🟡 THEO DÕI — lock an toàn nhưng tín hiệu yếu'
            else:
                timing = '🔴 KHÔNG VÀO — lock risk cao + tín hiệu yếu'

        # V4.1 FIX #10: Hold plan with regime awareness
        if vni_regime in ('CRISIS', 'BEAR'):
            hold_plan = CFG.SWING_HORIZON_MIN  # Minimize exposure
        elif 'TANG' in cons and ms > 0.3:
            hold_plan = CFG.SWING_HORIZON_MAX
        elif 'TANG' in cons:
            hold_plan = CFG.SWING_DEFAULT
        else:
            hold_plan = CFG.SWING_HORIZON_MIN

        # ── MC path summary for Forecast chart export ──
        mc_median_path = np.median(sims, axis=0).tolist()
        mc_ci_upper = np.percentile(sims, 97.5, axis=0).tolist()
        mc_ci_lower = np.percentile(sims, 2.5, axis=0).tolist()

        return {
            'weights': dict(zip(('mc', 'mean_reversion', 'momentum'), w)),
            'component_returns': {'mc_ret_pct': float(mc_r), 'mr_ret_pct': float(mr_r), 'mom_ret_pct': float(mom_r)},
            'ensemble_uncapped_ret_pct': float(uncapped_er),
            'ensemble_ret_pct': round(er,2), 'ensemble_price': round(lp*(1+er/100),0),
            'consensus': cons, 'confidence': round(conf,0), 'agreement_pct': round(conf,0), 'calibrated_prob_up': None, 'horizon': H,
            'mc': mc, 'mr': mr, 'mom': mom, 'lock_risk': lock_risk, 'unlock': unlock,
            'timing': timing, 'hold_plan_sessions': hold_plan,
            'hold_plan_label': f'{hold_plan} phien (~{round(hold_plan*7/5)} ngay)',
            'mc_path': {'median': mc_median_path, 'upper': mc_ci_upper, 'lower': mc_ci_lower},
            '_mc_paths': sims,
        }

# ==============================================================
# M9: RISK ENGINE
# ==============================================================

class RiskEng:
    @staticmethod
    def atr(df, per=14):
        if len(df)<per+1: return 0
        h,l,c = df['high'].values, df['low'].values, df['close'].values
        tr = np.maximum(h[1:]-l[1:], np.maximum(abs(h[1:]-c[:-1]), abs(l[1:]-c[:-1])))
        a = pd.Series(tr).rolling(per).mean().iloc[-1]
        return a if not pd.isna(a) else 0

    @staticmethod
    def tick_round(price, direction='down', exchange='HOSE'):
        return VNMarketRules.round_price(price, direction=direction, exchange=exchange)

    # Grid search bounds cho ATR-multiplier cua SL va 2 chot loi TP1 (chot mot phan)/TP2 (chot phan con lai)
    # Quy uoc: SL = entry - k_sl*ATR ; TP1 = entry + k_tp1*ATR ; TP2 = entry + k_tp2*ATR (k_tp2 > k_tp1)
    SL_GRID  = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5)
    TP1_GRID = (0.8, 1.0, 1.5, 2.0, 2.5)
    TP2_GRID = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
    BACKTEST_LOOKBACK = 120     # so phien lich su dung de backtest expectancy
    MAX_HOLD_SESSIONS = 20      # so phien toi da nam giu khi backtest 1 trade (label horizon)
    TP1_EXIT_FRACTION = 0.5     # ty trong chot tai TP1; phan con lai chay tiep toi TP2/SL

    # --- Sanity floors/caps cho SL: chan cac truong hop SL vo ly (0, am, hoac qua sat entry) ---
    MIN_SL_PCT = 0.015   # SL toi thieu phai cach entry >= 1.5% (du xa hon phi giao dich + noise 1 tick)
    MAX_SL_PCT = 0.12    # SL toi da 12% (chan ATR qua lon o co phieu bien dong cuc manh lam SL phi thuc te)
    VAR_SL_MULT = 1.1    # SL (theo %) nen rong hon toi thieu 1.1x |VaR95| 1-ngay, tranh bi quet boi bien dong thuong nhat

    @staticmethod
    def _simulate_trade_2tp(highs, lows, closes, i, k_sl, k_tp1, k_tp2, atr_i, max_hold,
                             tp1_frac=None, move_to_be_after_tp1=True):
        """
        Mo phong 1 lenh mo tai phien i, chot mot phan (tp1_frac) tai TP1, phan con lai chay toi TP2/SL/timeout.
        Sau khi cham TP1, SL cua phan con lai duoc doi ve breakeven (move_to_be_after_tp1) -- ky thuat quan tri
        rui ro chuan giup bao toan loi nhuan da thuc hien va khong de mot lenh thang bien thanh lenh lo.
        Tra ve R-multiple thuc hien duoc (blended theo tp1_frac), dung intrabar high/low de xac dinh cham truoc.
        Quy uoc: neu trong cung phien ca high cham TP va low cham SL thi coi la SL cham truoc (gia dinh bao thu).
        """
        tp1_frac = RiskEng.TP1_EXIT_FRACTION if tp1_frac is None else tp1_frac
        entry = closes[i]
        risk = k_sl * atr_i
        if risk <= 0:
            return None
        sl_price = entry - risk
        tp1_price = entry + k_tp1 * atr_i
        tp2_price = entry + k_tp2 * atr_i
        n = len(closes)
        end = min(i + max_hold, n - 1)

        tp1_hit = False
        r1 = None
        active_sl = sl_price
        for j in range(i + 1, end + 1):
            hit_sl = lows[j] <= active_sl
            hit_tp1 = (not tp1_hit) and highs[j] >= tp1_price
            hit_tp2 = tp1_hit and highs[j] >= tp2_price

            if not tp1_hit:
                if hit_sl and hit_tp1:
                    return -1.0  # bao thu: SL cham truoc khi kip chot TP1 -> toan bo lenh lo
                if hit_sl:
                    return -1.0
                if hit_tp1:
                    tp1_hit = True
                    r1 = k_tp1 / k_sl
                    if move_to_be_after_tp1:
                        active_sl = entry  # doi SL ve breakeven cho phan con lai
                    continue
            else:
                if hit_sl and hit_tp2:
                    r_rem = (active_sl - entry) / risk  # bao thu: SL (breakeven hoac goc) cham truoc TP2
                    return tp1_frac * r1 + (1 - tp1_frac) * r_rem
                if hit_sl:
                    r_rem = (active_sl - entry) / risk
                    return tp1_frac * r1 + (1 - tp1_frac) * r_rem
                if hit_tp2:
                    r2 = k_tp2 / k_sl
                    return tp1_frac * r1 + (1 - tp1_frac) * r2

        # Het thoi gian nam giu ma chua cham het cac moc -> dong phan con lai theo gia dong cua cuoi ky
        exit_price = closes[end]
        r_timeout = (exit_price - entry) / risk
        if tp1_hit:
            return tp1_frac * r1 + (1 - tp1_frac) * r_timeout
        return r_timeout

    @staticmethod
    def _expectancy_grid_search(df, atr_series):
        """
        Grid search (k_sl, k_tp1, k_tp2) tren BACKTEST_LOOKBACK phien gan nhat, tinh expectancy (theo R, da
        blend chot 1 phan tai TP1 + phan con lai chay toi TP2) cho moi bo ba, chon bo co expectancy cao nhat
        (EV = WinRate*AvgWin - (1-WinRate)*AvgLoss, chuan hoa theo R de so sanh cong bang giua cac ma).
        Tie-break: uu tien n_trades nhieu hon (dang tin hon) roi toi RR2 thap hon (TP2 gan hon, it "ao").
        Tra ve dict ket qua tot nhat + bang top ket qua de tham khao.
        """
        highs = df['high'].values; lows = df['low'].values; closes = df['close'].values
        n = len(closes)
        lookback = min(RiskEng.BACKTEST_LOOKBACK, n - RiskEng.MAX_HOLD_SESSIONS - 1)
        if lookback < 20:
            return None
        start_idx = n - lookback - RiskEng.MAX_HOLD_SESSIONS
        start_idx = max(start_idx, CFG.ATR_PERIOD + 1)
        results = []
        for k_sl in RiskEng.SL_GRID:
            for k_tp1 in RiskEng.TP1_GRID:
                if k_tp1 <= k_sl * 0.4:
                    continue  # TP1 qua gan SL, R:R phan chot dau khong dang risk
                for k_tp2 in RiskEng.TP2_GRID:
                    if k_tp2 <= k_tp1 * 1.2:
                        continue  # TP2 phai ro rang xa hon TP1
                    outcomes = []
                    for i in range(start_idx, n - 1):
                        atr_i = atr_series[i]
                        if not atr_i or atr_i <= 0 or np.isnan(atr_i):
                            continue
                        r = RiskEng._simulate_trade_2tp(highs, lows, closes, i, k_sl, k_tp1, k_tp2,
                                                         atr_i, RiskEng.MAX_HOLD_SESSIONS)
                        if r is not None:
                            outcomes.append(r)
                    if len(outcomes) < 15:
                        continue
                    outcomes = np.array(outcomes)
                    wr = (outcomes > 0).mean()
                    avg_win = outcomes[outcomes > 0].mean() if (outcomes > 0).any() else 0
                    avg_loss = abs(outcomes[outcomes <= 0].mean()) if (outcomes <= 0).any() else 0
                    expectancy_r = wr * avg_win - (1 - wr) * avg_loss
                    rr2 = k_tp2 / k_sl
                    results.append({
                        'k_sl': k_sl, 'k_tp1': k_tp1, 'k_tp2': k_tp2,
                        'rr1': round(k_tp1 / k_sl, 2), 'rr2': round(rr2, 2),
                        'win_rate': round(wr * 100, 1), 'avg_win_r': round(avg_win, 2),
                        'avg_loss_r': round(avg_loss, 2), 'expectancy_r': round(expectancy_r, 3),
                        'n_trades': len(outcomes),
                    })
        if not results:
            return None
        # Chon theo expectancy cao nhat; neu hoa, uu tien mau lon hon roi RR2 thap hon (it "ao")
        results.sort(key=lambda x: (-x['expectancy_r'], -x['n_trades'], x['rr2']))
        return {'best': results[0], 'top5': results[:5]}

    @staticmethod
    def stops(df, rs=None, fc=None, exchange='HOSE', liquidity=None, costs=None):
        """
        rs: dict thong ke tu StatEngine.returns() (co VaR_95, CVaR_95) -- dung de sanity-check SL.
        fc: dict tu FcastEngine.ensemble() (co confidence, ensemble_ret_pct) -- dung de blend xac suat
            thanh cong va dieu chinh do "tham vong" cua TP theo muc do dong thuan cua mo hinh du bao.
        """
        rs = rs or {}; fc = fc or {}; liquidity = liquidity or {}; costs = costs or {}
        if len(df)<20: return {}
        e = df['close'].iloc[-1]; a = RiskEng.atr(df)
        if a <= 0: return {'entry': round(e,2), 'atr': 0, 'warning': 'ATR=0'}

        # ATR series theo tung phien (de backtest khong bi lookahead bias)
        atr_series = pd.Series(np.maximum(
            df['high'].values[1:] - df['low'].values[1:],
            np.maximum(abs(df['high'].values[1:] - df['close'].values[:-1]),
                       abs(df['low'].values[1:] - df['close'].values[:-1]))
        )).rolling(CFG.ATR_PERIOD).mean()
        atr_series = pd.concat([pd.Series([np.nan]), atr_series], ignore_index=True).values  # align voi df

        gs = RiskEng._expectancy_grid_search(df, atr_series) if CFG.ENABLE_INTERNAL_STOP_OPTIMIZATION else None

        if gs is None:
            # Khong du du lieu de backtest -> fallback ve mac dinh an toan (RR1 1:0.75, RR2 1:1.5)
            k_sl, k_tp1, k_tp2 = 2.0, 1.5, 3.0
            wr_backtest, n_trades = None, 0
            gs_note = 'Internal stop optimization OFF; dùng cấu hình ổn định SL=2.0xATR / TP1=1.5xATR / TP2=3.0xATR'
            gs_meta = {}
        else:
            b = gs['best']
            k_sl, k_tp1, k_tp2 = b['k_sl'], b['k_tp1'], b['k_tp2']
            wr_backtest, n_trades = b['win_rate'] / 100.0, b['n_trades']
            gs_note = (f"Grid search (lookback {min(RiskEng.BACKTEST_LOOKBACK, len(df))} phien, "
                       f"{n_trades} trades): SL={k_sl}xATR / TP1={k_tp1}xATR (R:R 1:{b['rr1']}) / "
                       f"TP2={k_tp2}xATR (R:R 1:{b['rr2']}), WR={b['win_rate']}%, "
                       f"expectancy={b['expectancy_r']}R/trade")
            gs_meta = gs

        # ---------------------------------------------------------------
        # 1) SANITY CHECK cho SL: chan cac gia tri vo ly (0/am/qua sat entry)
        #    - Bug goc: ATR nho (co phieu gia thap / it bien dong) + tick-round co the trieu tieu risk ve 0.
        #    - Fix: ap floor toi thieu theo % entry (MIN_SL_PCT), doi chieu voi VaR95 thuc te (SL phai >=
        #      VAR_SL_MULT x |VaR95|, neu khong se bi quet lien tuc boi bien dong ngay thuong), va cap tran
        #      (MAX_SL_PCT) de tranh ATR bung no lam SL phi thuc te.
        # ---------------------------------------------------------------
        sl_atr_pct = (k_sl * a) / e if e > 0 else 0
        var95_pct = abs(rs.get('VaR_95', 0)) / 100.0  # VaR_95 luu duoi dang % (vd -3.44 nghia la -3.44%)
        sl_floor_pct = max(RiskEng.MIN_SL_PCT, RiskEng.VAR_SL_MULT * var95_pct) if var95_pct > 0 else RiskEng.MIN_SL_PCT
        sl_target_pct = min(RiskEng.MAX_SL_PCT, max(sl_atr_pct, sl_floor_pct))
        sl_sanity_note = None
        if sl_atr_pct < sl_floor_pct:
            sl_sanity_note = (f"SL theo ATR ({sl_atr_pct*100:.2f}%) qua sat entry so voi VaR95 "
                               f"({var95_pct*100:.2f}%) -> da noi rong SL len {sl_target_pct*100:.2f}% de tranh "
                               f"bi quet boi bien dong ngay thuong.")
        elif sl_atr_pct > RiskEng.MAX_SL_PCT:
            sl_sanity_note = (f"SL theo ATR ({sl_atr_pct*100:.2f}%) qua rong (co phieu bien dong manh) -> "
                               f"da gioi han lai o muc tran {RiskEng.MAX_SL_PCT*100:.0f}%.")

        sl_swing = RiskEng.tick_round(e * (1 - sl_target_pct), 'down', exchange)
        risk = round(e - sl_swing, 4)
        if risk <= 0:
            # Phong ho cuoi cung: khong bao gio duoc phep tra ve SL >= entry
            sl_swing = RiskEng.tick_round(e * (1 - RiskEng.MIN_SL_PCT), 'down', exchange)
            risk = round(e - sl_swing, 4)
            sl_sanity_note = (sl_sanity_note or '') + ' [FORCE-FIX] SL tinh ra >= entry, da ep ve floor toi thieu.'

        # ---------------------------------------------------------------
        # 2) Quy doi RR toi uu (tu grid search) sang risk THUC TE (sau sanity-check + tick-round)
        # ---------------------------------------------------------------
        rr1 = k_tp1 / k_sl
        rr2 = k_tp2 / k_sl
        tp1_raw = e + rr1 * risk
        tp2_raw = e + rr2 * risk

        # ---------------------------------------------------------------
        # 3) Cap TP theo bien do thuc te: percentile 90 cua max-favorable-move trong MAX_HOLD_SESSIONS
        #    phien gan day, tranh truong hop TP vuot xa nhung gi co phieu tung dat duoc.
        # ---------------------------------------------------------------
        highs = df['high'].values; closes = df['close'].values
        mfe_pct = []
        for i in range(max(0, len(df) - RiskEng.BACKTEST_LOOKBACK), len(df) - 1):
            end = min(i + RiskEng.MAX_HOLD_SESSIONS, len(df) - 1)
            if end <= i: continue
            best_high = highs[i+1:end+1].max()
            mfe_pct.append((best_high - closes[i]) / closes[i] * 100)
        realistic_cap_pct = np.percentile(mfe_pct, 90) if len(mfe_pct) >= 10 else None
        tp_realistic_cap = e * (1 + realistic_cap_pct/100) if realistic_cap_pct else None

        # ---------------------------------------------------------------
        # 4) Dieu chinh TP theo do dong thuan cua mo hinh du bao (Confidence) + huong ky vong (EnsRet%):
        #    - Neu Confidence thap hoac EnsRet% <=0, mo hinh chua co edge ro rang -> keo TP2 ve gan TP1 hon
        #      (giam "tham vong", uu tien chot loi som) thay vi giu nguyen muc TP2 lac quan tu backtest.
        #    - Neu Confidence cao va EnsRet% ho tro huong TP2 goc thi giu nguyen.
        # ---------------------------------------------------------------
        conf_pct = fc.get('confidence', 50)
        ens_ret_pct = fc.get('ensemble_ret_pct', 0)
        conviction = max(0.0, min(1.0, (conf_pct - 33) / (100 - 33)))  # 0 khi conf<=33 (muc "tung dong xu")
        if ens_ret_pct <= 0:
            conviction *= 0.5  # mo hinh du bao khong ung ho chieu long -> giam tham vong TP hon nua
        tp2_adj = tp1_raw + conviction * (tp2_raw - tp1_raw)  # shrink TP2 ve TP1 khi conviction thap

        tp1 = RiskEng.tick_round(tp1_raw, 'up', exchange)
        tp2_before_cap = RiskEng.tick_round(tp2_adj, 'up', exchange)
        tp2 = RiskEng.tick_round(min(tp2_before_cap, tp_realistic_cap), 'up', exchange) if tp_realistic_cap else tp2_before_cap
        tp_was_capped = tp_realistic_cap is not None and tp2_before_cap > tp_realistic_cap
        if tp2 <= tp1:
            tp2 = RiskEng.tick_round(tp1_raw * 1.02, 'up', exchange)  # dam bao TP2 luon > TP1 sau moi buoc dieu chinh

        # ---------------------------------------------------------------
        # 5) Xac suat thanh cong "blended": ket hop WinRate tu backtest (empirical, phu thuoc mau) voi
        #    Confidence cua mo hinh du bao hien tai (phan anh trang thai tin hieu MOI NHAT, khong co trong
        #    backtest lich su). Trong so backtest tang theo can bac hai so luong trade (shrinkage chuan),
        #    tranh qua tin vao mau nho nhung cung khong bo qua tin hieu du bao hien tai.
        # ---------------------------------------------------------------
        # Agreement giữa mô hình KHÔNG phải xác suất thắng. Chỉ tính EV khi có xác suất đã calibration OOS.
        calibrated_p = fc.get('calibrated_prob_up')
        p_win_blended = float(calibrated_p) if calibrated_p is not None else None
        avg_r_win = RiskEng.TP1_EXIT_FRACTION * rr1 + (1 - RiskEng.TP1_EXIT_FRACTION) * (tp2 - e) / risk if risk > 0 else 0
        if p_win_blended is not None:
            ev_r = p_win_blended * avg_r_win - (1 - p_win_blended)
            ev_vnd_per_share = round(ev_r * risk, 2)
        else:
            ev_r = None; ev_vnd_per_share = None
        roundtrip_cost_vnd = e * float(costs.get('roundtrip_cost_pct', 0) or 0) / 100.0
        rr_net_cost = ((tp2 - e) - roundtrip_cost_vnd) / risk if risk > 0 else 0

        sl_wide  = RiskEng.tick_round(e - (k_sl+1)*a, 'down', exchange)
        sl_max   = RiskEng.tick_round(e - (k_sl+2)*a, 'down', exchange)
        swing_low_10 = df['low'].tail(10).min()
        swing_low_20 = df['low'].tail(20).min()
        trail_atr = round(2*a, 2)

        if len(df) >= 10:
            lock_hist_dd = []
            closes_arr = df['close'].values; lows = df['low'].values
            lock_sessions = CFG.MIN_HOLD_SESSIONS
            for i in range(lock_sessions, len(closes_arr)):
                entry_p = closes_arr[i-lock_sessions]
                worst_in_lock = lows[i-lock_sessions+1:i+1].min()
                lock_hist_dd.append((worst_in_lock - entry_p) / entry_p * 100)
            mae_lock = round(np.percentile(lock_hist_dd, 10), 2) if lock_hist_dd else 0
        else: mae_lock = 0

        # Historical low-based MAE uses the same post-entry session window as MC lock risk.
        sl_pct_display = -sl_target_pct * 100
        lock_risk_flag = mae_lock < sl_pct_display  # Lock MAE is deeper than the displayed stop buffer.

        return {
            'entry': round(e,2), 'atr': round(a,2),
            'lock_sessions': CFG.MIN_HOLD_SESSIONS,
            'construction': {'atr_stop_pct': sl_atr_pct * 100, 'stop_floor_pct': sl_floor_pct * 100,
                             'base_tp2': tp2_raw, 'conviction_tp2': tp2_adj,
                             'optimization_enabled': bool(gs)},
            'sl_swing': sl_swing, 'sl_wide': sl_wide, 'sl_max': sl_max,
            'sl_pct': round(sl_pct_display, 2), 'sl_sanity_note': sl_sanity_note,
            'lock_risk_flag': lock_risk_flag,
            'swing_low_10': round(swing_low_10,2), 'swing_low_20': round(swing_low_20,2),
            # TP1 (chot mot phan, vd 50%) + TP2 (phan con lai) toi uu hoa qua grid search + expectancy,
            # da dieu chinh theo conviction cua mo hinh du bao va cap theo bien do thuc te (MFE p90)
            'tp1': tp1, 'tp2': tp2, 'tp1_exit_fraction': RiskEng.TP1_EXIT_FRACTION,
            'rr1': round(rr1, 2), 'rr2': round((tp2-e)/risk, 2) if risk > 0 else 0,
            # Giu key 'tp_optimal'/'rr_optimal' de tuong thich nguoc voi report/Excel hien co (= TP2)
            'tp_optimal': tp2, 'rr_optimal': round((tp2-e)/risk, 2) if risk > 0 else 0,
            'k_sl': k_sl, 'k_tp1': k_tp1, 'k_tp2': k_tp2, 'tp_was_capped': tp_was_capped,
            'tp_realistic_cap': round(tp_realistic_cap, 2) if tp_realistic_cap else None,
            'mfe_p90_pct': round(realistic_cap_pct, 2) if realistic_cap_pct else None,
            'conviction': round(conviction, 2),
            'p_win_backtest': round(wr_backtest*100, 1) if wr_backtest is not None else None,
            'model_agreement_pct': round(conf_pct,1),
            'p_win_blended': round(p_win_blended*100, 1) if p_win_blended is not None else None,
            'n_trades_backtest': n_trades,
            'ev_r_per_trade': round(ev_r, 3) if ev_r is not None else None,
            'ev_vnd_per_share': ev_vnd_per_share, 'rr_net_cost': round(rr_net_cost,2),
            'grid_search_note': gs_note, 'grid_search_detail': gs_meta,
            # Giu lai cac moc 2R/3R/4R de tuong thich nguoc, tinh theo risk thuc te
            'tp_2r': RiskEng.tick_round(e + 2*risk, 'up', exchange), 'tp_3r': RiskEng.tick_round(e + 3*risk, 'up', exchange),
            'tp_4r': RiskEng.tick_round(e + 4*risk, 'up', exchange), 'risk_per_share': round(risk,2),
            'trail_atr': trail_atr, 'trail_note': f'Sau khi chứng khoán về T+2: trailing {trail_atr:,.0f} VND (2x ATR)',
            'mae_lock_10pct': mae_lock,
            'mae_lock_note': f'Lich su: 90% truong hop DD trong lock < {abs(mae_lock):.1f}%'
                              + (' -- CANH BAO: SL hien tai hep hon muc DD nay, co the "chet" trong settlement lock.' if lock_risk_flag else ''),
            'settlement': f'T+{CFG.SETTLEMENT_DAYS} — daily-bar approximation: sớm nhất bán chiều T+2', 'exchange': exchange,
        }
    @staticmethod
    def sizing(entry, stop, acct=None, hold_sessions=None, exchange='HOSE', liquidity=None):
        acct = float(acct or CFG.ACCOUNT_SIZE); liquidity = liquidity or {}
        rps = abs(float(entry) - float(stop))
        if rps <= 0 or entry <= 0: return {}
        risk_budget = acct * CFG.MAX_POSITION_RISK
        lot = VNMarketRules.profile(exchange)['lot_size']
        shares_risk = int((risk_budget / rps) // lot) * lot
        shares_alloc = int(((acct * CFG.MAX_POSITION_PCT) / entry) // lot) * lot
        capacity = max(0, int(liquidity.get('capacity_shares', shares_alloc) or 0))
        capacity = (capacity // lot) * lot
        shares = max(0, min(shares_risk, shares_alloc, capacity))
        limits = {'risk': shares_risk, 'allocation': shares_alloc, 'liquidity': capacity}
        binding = [name for name, limit in limits.items() if limit == min(limits.values())]
        value = shares * entry
        actual_max_loss = shares * rps
        pct_acct = value / acct * 100 if acct > 0 else 0
        hold = hold_sessions or CFG.SWING_DEFAULT
        capital_lock_days = round(hold * 7/5)
        return {
            'shares': shares, 'value': round(value,0), 'pct_acct': round(pct_acct,1),
            'risk_budget_vnd': round(risk_budget,0), 'max_loss': round(actual_max_loss,0),
            'risk_per_share_vnd': round(rps,0), 'shares_by_risk': shares_risk,
            'shares_by_allocation': shares_alloc, 'shares_by_liquidity': capacity,
            'binding_constraints': binding, 'account_size_vnd': acct,
            'risk_pct_nav': round(actual_max_loss / acct * 100, 4),
            'capital_lock_days': capital_lock_days,
            'capital_lock_note': f'Vốn {value:,.0f} VND dự kiến khóa ~{capital_lock_days} ngày',
        }

    @staticmethod
    def kelly(wr, aw, al):
        if al==0: return {}
        b = aw/al; p,q = wr, 1-wr; k = (p*b-q)/b
        return {'full_pct':round(max(0,k)*100,2), 'half_pct':round(max(0,k/2)*100,2),
                'edge_pct':round((p*b-q)*100,2), 'has_edge':bool(k>0)}

# ==============================================================
# ADAPTIVE SCORER — V4: CROSS-SECTIONAL Z-SCORE SCORING
# ==============================================================
#
# FIX #5: Magic numbers replaced with z-score normalization.
#   For each factor across N stocks in batch:
#     z_i = (f_i - median) / IQR   [robust z-score]
#     s_i = clip(z_i, -1, +1)      [bounded signal]
#   composite = sum(w_i * s_i), sum(w_i) = 1
#   final = 50 + composite * 40    [maps to ~[10,90]]
#   + VNI regime multiplicative adjustment
# ==============================================================

class AdaptiveScorer:
    WEIGHTS = {
        'sharpe': 0.08, 'forecast_ret': 0.12, 'forecast_conf': 0.05,
        'cmf': 0.08, 'trend_quality': 0.10, 'alpha_composite': 0.10,
        'hmm_regime': 0.07, 'lock_safety': 0.07, 'vol_regime': 0.05,
        'distribution_quality': 0.04, 'liquidity': 0.12, 'data_quality': 0.12,
    }

    @staticmethod
    def _robust_zscore(value, values_array):
        if len(values_array) < 3: return 0.0
        med = np.median(values_array)
        iqr = np.percentile(values_array, 75) - np.percentile(values_array, 25)
        if iqr < 1e-9: return 0.0
        return float(np.clip((value - med) / iqr, -1.0, 1.0))

    @staticmethod
    def extract_factors(report):
        rs = report.get('stats', {}); fc = report.get('fcast', {})
        fl = report.get('flow', {}); tr = report.get('trend', {})
        al = report.get('alpha', {}).get('composite', {}); hm = report.get('hmm', {})
        vl = report.get('vol', {}); di = report.get('dist', {}).get('verdict', {})
        lock = fc.get('lock_risk', {}); liq = report.get('liquidity', {}); dq = report.get('data_quality', {}); costs = report.get('costs', {})
        hmm_cur = hm.get('current', '')
        hmm_num = 1.0 if 'BULL' in hmm_cur else (-1.0 if 'BEAR' in hmm_cur else 0.0)
        hmm_score = hmm_num * (hm.get('prob_pct', 50) / 100.0)
        er = tr.get('er', 0); slope = tr.get('slope_pct', 0)
        trend_q = er * np.sign(slope) if slope != 0 else 0
        lock_safety = 100 - lock.get('prob_loss_gt_3pct', 50)
        vol_score = 1.0 - vl.get('vol_ratio', 1.0)
        dist_score = 1.0 if di.get('is_gaussian', True) else -0.5
        return {
            'sharpe': rs.get('sharpe', 0),
            'forecast_ret': costs.get('net_forecast_pct', fc.get('ensemble_ret_pct', 0)),
            'forecast_conf': (fc.get('confidence', 50) - 50) / 50,
            'cmf': fl.get('cmf', 0), 'trend_quality': trend_q,
            'alpha_composite': al.get('alpha', 0), 'hmm_regime': hmm_score,
            'lock_safety': (lock_safety - 50) / 50,
            'vol_regime': vol_score, 'distribution_quality': dist_score,
            'liquidity': {'A':1.0,'B':0.5,'C':0.0,'D':-1.0}.get(liq.get('tier','D'),-1.0),
            'data_quality': (dq.get('score',50)-50)/50,
        }

    @staticmethod
    def score_batch(reports, idx_df=None):
        if not reports: return {}
        factor_data = {}
        for rp in reports:
            if 'error' in rp: continue
            factor_data[rp['symbol']] = AdaptiveScorer.extract_factors(rp)
        if len(factor_data) < 2:
            return AdaptiveScorer._score_absolute(reports, idx_df)
        symbols = list(factor_data.keys())
        factor_names = list(AdaptiveScorer.WEIGHTS.keys())
        factor_arrays = {fn: np.array([factor_data[s].get(fn,0) for s in symbols]) for fn in factor_names}
        scores = {}
        vni_regime = AdaptiveScorer._detect_vni_regime(idx_df)
        regime_mult = {'BULL': 1.08, 'NEUTRAL': 1.00, 'WEAK': 0.92, 'BEAR': 0.85, 'CRISIS': 0.70}

        # V4.1 FIX #6: Compute absolute scores first for blending
        abs_scores = AdaptiveScorer._score_absolute(reports, idx_df)
        anchor = CFG.ABSOLUTE_SCORE_ANCHOR  # 0.5 = 50% absolute + 50% cross-sectional

        for i, sym in enumerate(symbols):
            weighted_sum = 0.0; details = {}
            for fn in factor_names:
                raw = factor_arrays[fn][i]
                z = AdaptiveScorer._robust_zscore(raw, factor_arrays[fn])
                w = AdaptiveScorer.WEIGHTS[fn]
                weighted_sum += w * z
                details[fn] = {'raw': round(float(raw),4), 'z': round(z,3), 'w': w,
                               'cs_contribution_points': (1-anchor) * 40 * w * z * regime_mult.get(vni_regime, 1.0)}
            cs_score = 50 + weighted_sum * 40
            cs_adj = cs_score * regime_mult.get(vni_regime, 1.0)

            # V4.1: Blend with absolute score to reduce universe-dependence
            abs_score = abs_scores.get(sym, {}).get('score', 50)
            blended = anchor * abs_score + (1 - anchor) * cs_adj

            rp = next((r for r in reports if r['symbol']==sym), None)
            sig = (rp.get('screener_signal','') or '') if rp else ''
            if 'MẠNH' in sig: blended += 4
            elif 'TÍCH LŨY' in sig and factor_data[sym].get('vol_regime',0)>0: blended += 3
            elif 'PHÂN PHỐI' in sig: blended -= 6
            final = int(np.clip(blended, 0, 100))
            rt = ('⭐⭐⭐⭐⭐ SWING BUY' if final>=80 else '⭐⭐⭐⭐ BUY & HOLD' if final>=65 else
                  '⭐⭐⭐ THEO DÕI' if final>=50 else '⭐⭐ YẾU' if final>=35 else '⭐ TRÁNH')
            fc = (rp.get('fcast',{}) if rp else {})
            scores[sym] = {'score': final, 'rating': rt, 'vni_regime': vni_regime,
                           'timing': fc.get('timing',''), 'hold_plan': fc.get('hold_plan_label',''),
                           'scoring_method': 'cross_sectional_zscore', 'factor_details': details}
        return scores

    @staticmethod
    def _score_absolute(reports, idx_df=None):
        scores = {}; vni_regime = AdaptiveScorer._detect_vni_regime(idx_df)
        regime_mult = {'BULL': 1.08, 'NEUTRAL': 1.0, 'WEAK': 0.92, 'BEAR': 0.85, 'CRISIS': 0.70}
        for rp in reports:
            if 'error' in rp: continue
            sym = rp['symbol']; f = AdaptiveScorer.extract_factors(rp)
            sc = 50
            sc += np.clip(f['sharpe'] * 10, -15, 15)
            sc += np.clip(f['forecast_ret'] * 2, -10, 10)
            sc += np.clip(f['cmf'] * 30, -7, 7)
            sc += np.clip(f['trend_quality'] * 12, -7, 7)
            sc += np.clip(f['alpha_composite'] * 10, -10, 10)
            sc += np.clip(f['hmm_regime'] * 5, -5, 5)
            sc += np.clip(f['lock_safety'] * 5, -5, 5)
            sc += np.clip(f['liquidity'] * 6, -6, 6)
            sc += np.clip(f['data_quality'] * 6, -6, 6)
            sc = sc * regime_mult.get(vni_regime, 1.0)
            sig = (rp.get('screener_signal','') or '')
            if 'MẠNH' in sig: sc += 4
            elif 'PHÂN PHỐI' in sig: sc -= 6
            final = int(np.clip(sc, 0, 100))
            rt = ('⭐⭐⭐⭐⭐ SWING BUY' if final>=80 else '⭐⭐⭐⭐ BUY & HOLD' if final>=65 else
                  '⭐⭐⭐ THEO DÕI' if final>=50 else '⭐⭐ YẾU' if final>=35 else '⭐ TRÁNH')
            fc = rp.get('fcast',{})
            scores[sym] = {'score': final, 'rating': rt, 'vni_regime': vni_regime,
                           'timing': fc.get('timing',''), 'hold_plan': fc.get('hold_plan_label',''),
                           'scoring_method': 'absolute_fallback'}
        return scores

    @staticmethod
    def _detect_vni_regime(idx_df):
        if idx_df is None or 'close' not in idx_df.columns or len(idx_df)<50: return 'NEUTRAL'
        try:
            c = idx_df['close']
            ma20 = c.rolling(20).mean().iloc[-1]; ma50 = c.rolling(50).mean().iloc[-1]
            last = c.iloc[-1]
            ret_20d = (last / c.iloc[-20] - 1) if len(c)>=20 else 0
            ret_5d = (last / c.iloc[-5] - 1) if len(c)>=5 else 0
            ret_1d = (last / c.iloc[-2] - 1) if len(c)>=2 else 0

            # V4.1 FIX #4: CRISIS = extreme short-term damage
            # Condition 1: 1-day crash AND 5-day decline
            if ret_1d < CFG.VNI_CRASH_1D_PCT and ret_5d < CFG.VNI_CRASH_5D_PCT:
                return 'CRISIS'
            # Condition 2: Severe 5-day crash alone
            if ret_5d < CFG.VNI_CRASH_5D_PCT * 1.5:  # -7.5% in 5 days
                return 'CRISIS'
            # Condition 3: Single-day capitulation (e.g. -3% flash crash)
            if ret_1d < CFG.VNI_CRASH_1D_PCT * 1.5:  # -3% in 1 day
                return 'CRISIS'
            # Existing logic
            if last > ma20 > ma50 and ret_20d > 0.02: return 'BULL'
            elif last < ma20 < ma50 and ret_20d < -0.02: return 'BEAR'
            elif last < ma20 and ret_5d < -0.01: return 'WEAK'
            else: return 'NEUTRAL'
        except: return 'NEUTRAL'

# ==============================================================
# ORCHESTRATOR — V4
# ==============================================================

class QuantPipeline:
    def __init__(self, cfg=None):
        self.cfg = cfg or CFG
        log.info("QuantPipeline V6 VN initialized (DQ + liquidity + costs + action gate)")

    def analyze(self, sym, df, sig=None, scores=None, idx_df=None, vni_regime='NEUTRAL', exchange='HOSE'):
        if df is None or len(df)<30: return {'symbol':sym, 'error':'Insufficient data'}
        p = df['close']
        r = {'symbol':sym, 'exchange':VNMarketRules.normalize_exchange(exchange), 'date':datetime.now().strftime('%Y-%m-%d %H:%M'),
             'n':len(df), 'range':f"{str(df.index.min())[:10]} -> {str(df.index.max())[:10]}",
             'screener_signal':sig, 'screener_scores':scores,
             '_price_dates': df.index.tolist(), '_price_close': p.tolist()}
        log.info(f"  [{sym}] M0: Data quality + liquidity...")
        r['data_quality'] = DataQualityEngine.audit(df, exchange)
        if r['data_quality']['status'] == 'FAIL':
            r['error'] = 'Data quality failed: ' + ', '.join(r['data_quality']['flags'])
            return r
        r['liquidity'] = LiquidityEngine.analyze(df, exchange)
        log.info(f"  [{sym}] M1: Distribution..."); r['dist'] = DistributionAnalyzer.full_test(p)
        log.info(f"  [{sym}] M2: Statistics..."); r['stats'] = StatEngine.returns(p); r['vol'] = StatEngine.vol_regime(p); r['ac'] = StatEngine.autocorr(p)
        log.info(f"  [{sym}] M3: ARIMA..."); r['arima'] = ARIMAEngine.fit(p)
        log.info(f"  [{sym}] M4: GARCH..."); r['garch'] = GARCHEngine.fit(p)
        # V4.1: Pass symbol for stable HMM seeding
        log.info(f"  [{sym}] M5: HMM (multivariate)..."); r['hmm'] = HMMEngine.fit(df, symbol=sym)
        log.info(f"  [{sym}] M6: Alpha..."); r['alpha'] = AlphaEngine.extract(df, idx_df)
        log.info(f"  [{sym}] M7: Structure (swing SR)..."); r['sr'] = StructureEngine.sr_levels(df); r['trend'] = StructureEngine.trend(df); r['flow'] = StructureEngine.flow(df)
        # V4.1: Pass GARCH result + symbol + VNI regime to MC
        log.info(f"  [{sym}] M8: Forecast (GARCH-MC)..."); r['fcast'] = FcastEngine.ensemble(p, garch_result=r.get('garch'), symbol=sym, vni_regime=vni_regime)
        r['costs'] = CostEngine.analyze(r.get('fcast',{}), r.get('liquidity',{}))
        log.info(f"  [{sym}] M9: Risk...")
        sl = RiskEng.stops(df, rs=r.get('stats', {}), fc=r.get('fcast', {}), exchange=exchange, liquidity=r.get('liquidity'), costs=r.get('costs')); r['sl'] = sl
        fc_hold = r.get('fcast',{}).get('hold_plan_sessions', CFG.SWING_DEFAULT)
        if sl and sl.get('sl_swing'):
            r['pos'] = RiskEng.sizing(sl['entry'], sl['sl_swing'], acct=self.cfg.ACCOUNT_SIZE, hold_sessions=fc_hold, exchange=exchange, liquidity=r.get('liquidity'))
        # Kelly chỉ hợp lệ khi đầu vào là thống kê trade đã calibration từ backtest riêng.
        r['kelly'] = {'disabled': True, 'note': 'Không tính Kelly từ tỷ lệ ngày tăng/giảm.'}
        # Scoring deferred to batch() for cross-sectional z-scores
        r['rec'] = {'score': 50, 'rating': '⭐⭐⭐ PENDING', 'vni_regime': 'NEUTRAL'}
        return r

    def batch(self, data, scr_df=None, idx_df=None, exchange_map=None):
        exchange_map = exchange_map or {}
        # V4.1: Detect VNI regime EARLY so each stock's forecast respects it
        vni_regime = AdaptiveScorer._detect_vni_regime(idx_df)
        log.info(f"\n  [VNI REGIME] {vni_regime}")
        if vni_regime == 'CRISIS':
            log.warning("  ⚠️ CRISIS DETECTED — circuit breaker active, all timing restricted")

        # V5: Nhóm ngành (ICB) + xu hướng ngành + tương quan chéo/VAR theo ngành
        log.info("\n  [SECTOR] Tra cứu ICB + tính xu hướng ngành + tương quan chéo/VAR...")
        sector_map = SectorEngine.map_symbols(list(data.keys()))
        sector_trends = SectorEngine.sector_trend(data, sector_map)
        sector_groups = {}
        for sym, sec in sector_map.items():
            sector_groups.setdefault(sec, []).append(sym)
        sector_corr = {}
        for sec, syms in sector_groups.items():
            if len(syms) >= 2:
                log.info(f"    Ngành '{sec}': {len(syms)} mã — {syms}")
                sector_corr[sec] = CrossCorrelationEngine.analyze_sector(data, syms)
            else:
                sector_corr[sec] = {'error': 'Chỉ có 1 mã trong ngành — không tính được tương quan',
                                     'avg_pairwise_corr': None}

        reports = []
        for i,(sym,df) in enumerate(data.items()):
            log.info(f"\n[{i+1}/{len(data)}] === {sym} ===")
            sig, scores = None, None
            if scr_df is not None and not scr_df.empty:
                mc = [c for c in scr_df.columns if 'ma' in c.lower() or 'symbol' in c.lower()]
                if mc:
                    m = scr_df[scr_df[mc[0]]==sym]
                    if not m.empty:
                        row = m.iloc[0]
                        sc2 = [c for c in scr_df.columns if 'tin hieu' in c.lower() or 'adj' in c.lower()]
                        sig = str(row[sc2[0]]) if sc2 else None
            rp = self.analyze(sym, df, sig, scores, idx_df, vni_regime=vni_regime, exchange=exchange_map.get(sym, CFG.DEFAULT_EXCHANGE))
            sec = sector_map.get(sym, 'Không xác định')
            rp['sector'] = {'name': sec, 'trend': sector_trends.get(sec, {})}
            rp['cross_corr'] = sector_corr.get(sec, {})
            reports.append(rp)
        # V4: Batch scoring
        log.info("\n  [BATCH] Adaptive cross-sectional scoring...")
        batch_scores = AdaptiveScorer.score_batch(reports, idx_df)
        for rp in reports:
            if 'error' in rp: continue
            if rp['symbol'] in batch_scores:
                rp['rec'] = batch_scores[rp['symbol']]
                rp['action'] = ActionEngine.decide(rp)
        reports.sort(key=lambda x:x.get('rec',{}).get('score',0), reverse=True)
        return reports

    # ==============================================================
    # COMMENTARY ENGINE — V4.1: Thesis-driven narrative
    # ==============================================================
    # Structure per stock:
    #   ① Sức khỏe tài chính — thesis + metrics as PROOF + explain WHY
    #   ② Trạng thái kỹ thuật — vol regime, trend, money flow as STORY
    #   ③ Mô hình thống kê — HMM/Forecast/Kelly as CONVICTION measure
    #   ④ Rủi ro — GARCH, distribution, lock risk as HONEST WARNING
    #   Kết luận — action + levels + sizing in VND
    # ==============================================================

    def generate_commentary(self, r, detailed=False):
        """Keep the legacy report narrative in Summary; use the new thesis only in BUY_ONLY."""
        if detailed:
            return commentary(r, detailed=True)
        if 'error' in r:
            return 'Không đủ dữ liệu để phân tích.'
        rs = r.get('stats', {}); vl = r.get('vol', {}); hm = r.get('hmm', {})
        fc = r.get('fcast', {}); ga = r.get('garch', {}).get('garch', {})
        eg = r.get('garch', {}).get('egarch', {}); sl = r.get('sl', {})
        di = r.get('dist', {}).get('verdict', {}); ft = r.get('dist', {}).get('fat_tail', {})
        ky = r.get('kelly', {}); rc = r.get('rec', {}); fl = r.get('flow', {})
        tr = r.get('trend', {}); al = r.get('alpha', {}); lock = fc.get('lock_risk', {})
        sym = r.get('symbol', '?')
        return '\n'.join([
            f'{sym} — Điểm {rc.get("score", 0)} ({rc.get("rating", "")})',
            self._section_health(rs, ky),
            self._section_technical(vl, tr, fl, al),
            self._section_models(hm, fc, ky, r.get('screener_signal', '') or ''),
            self._section_risk(ga, eg, di, ft, rs, lock, sl),
            self._section_conclusion(sym, rc, fc, sl, r.get('pos', {}), lock),
        ])

    # -----------------------------------------------------------------
    def _section_health(self, rs, ky):
        """① Hiệu suất giá và rủi ro lịch sử; không phải sức khỏe tài chính doanh nghiệp."""
        ar = rs.get('ann_return_pct', 0)
        sh = rs.get('sharpe', 0)
        so = rs.get('sortino', 0)
        cal = rs.get('calmar', 0)
        mdd = rs.get('max_dd_pct', 0)
        wr = rs.get('win_rate_pct', 0)
        pf = rs.get('profit_factor', 0)

        lines = []

        # Determine thesis
        if sh >= 1.5:
            thesis = 'Hiệu suất giá–rủi ro xuất sắc'
        elif sh >= 0.5:
            thesis = 'Hiệu suất giá–rủi ro tích cực'
        elif sh >= 0:
            thesis = 'Hiệu suất giá–rủi ro trung bình'
        else:
            thesis = 'Hiệu suất giá–rủi ro yếu'

        lines.append(f'① {thesis}')

        # Build narrative
        narr = f'AnnRet {ar:.1f}%'
        if sh >= 0.5:
            narr += f' với Sharpe {sh:.2f} — mỗi đơn vị rủi ro chấp nhận, cổ phiếu trả lại {sh:.1f} đơn vị lợi nhuận'
            if so > sh * 1.2:
                narr += f'. Sortino {so:.2f} cao hơn Sharpe đáng kể — phần lớn biến động là biến động tăng, không phải giảm'
        elif sh >= 0:
            narr += f', Sharpe {sh:.2f} — lợi nhuận có bù được rủi ro nhưng chưa ấn tượng'
        else:
            narr += f', Sharpe {sh:.2f} — lợi nhuận KHÔNG bù được rủi ro đã chấp nhận. '
            narr += f'Nói thẳng: gửi tiết kiệm có thể tốt hơn nếu chỉ nhìn lịch sử'

        # Calmar — relate to MaxDD
        if cal > 3:
            narr += f'. Calmar {cal:.1f} — lợi nhuận năm gấp {cal:.1f}x mức thua xấu nhất (MaxDD {mdd:.1f}%)'
        elif mdd < -15:
            narr += f'. MaxDD {mdd:.1f}% — từng mất hơn 15% từ đỉnh, đây là mức drawdown nặng'
        elif mdd < -10:
            narr += f'. MaxDD {mdd:.1f}% — cần chuẩn bị tâm lý cho drawdown 2 chữ số'

        # Win rate + profit factor context
        if wr > 0 and pf > 0:
            if pf > 1.5 and wr > 55:
                narr += f'. Tỷ lệ ngày tăng {wr:.0f}%, Daily PF {pf:.2f} — thắng nhiều hơn thua, và khi thắng thì lớn hơn khi thua'
            elif pf < 1:
                narr += f'. Daily PF {pf:.2f} (<1) — khi thua, mức lỗ trung bình lớn hơn mức lãi. Đây là dấu hiệu cần cải thiện'

        lines.append(narr + '.')
        return '\n'.join(lines)

    # -----------------------------------------------------------------
    def _section_technical(self, vl, tr, fl, al):
        """② Thesis: current technical state — vol, trend, flow, RS."""
        lines = []

        # Determine thesis from vol regime
        vol_rg = vl.get('regime', '')
        vol_ratio = vl.get('vol_ratio', 1)
        vol_pct = vl.get('vol_pctile', 50)

        if vol_rg == 'CONTRACTION':
            thesis = 'Đang "nén lò xo" — biến động thấp bất thường'
            vol_detail = (f'Vol Ratio {vol_ratio:.3f} tại Percentile {vol_pct:.1f}% — '
                          f'biến động 10 ngày chỉ bằng {vol_ratio:.0%} so với 60 ngày, '
                          f'thấp hơn {100-vol_pct:.0f}% lịch sử. '
                          f'Im lặng kéo dài thường đi trước bùng nổ — đây là giai đoạn tích lũy')
        elif vol_rg == 'EXPANSION':
            thesis = 'Biến động đang mở rộng — thị trường "nóng"'
            vol_detail = (f'Vol Ratio {vol_ratio:.3f} — biến động ngắn hạn cao hơn {vol_ratio:.0%} so với trung hạn. '
                          f'Cần cẩn trọng: biến động cao = cả cơ hội lẫn rủi ro lớn hơn')
        else:
            thesis = 'Biến động bình thường'
            vol_detail = f'Vol Ratio {vol_ratio:.3f} — không có tín hiệu bất thường về biến động'

        lines.append(f'② {thesis}')
        narr = vol_detail

        # Trend quality
        tr_label = tr.get('label', '')
        er = tr.get('er', 0)
        slope = tr.get('slope_pct', 0)
        if er > 0.5:
            narr += f'. Xu hướng rõ ràng (ER {er:.2f}, slope {slope:+.1f}%) — giá di chuyển có hướng, không lắc lư ngẫu nhiên'
        elif er > 0.3:
            narr += f'. Xu hướng nhẹ (ER {er:.2f}) — có hướng nhưng nhiễu khá nhiều'
        else:
            narr += f'. Chưa có xu hướng rõ (ER {er:.2f}) — giá lắc lư không phương hướng'

        # Money flow
        cmf = fl.get('cmf', 0)
        if cmf > 0.1:
            narr += f'. Dòng tiền tích cực (CMF {cmf:+.3f}) — tiền thông minh đang tích lũy'
        elif cmf < -0.1:
            narr += f'. Dòng tiền tiêu cực (CMF {cmf:+.3f}) — có dấu hiệu phân phối, tiền đang rút ra'
        elif abs(cmf) <= 0.1 and cmf != 0:
            narr += f'. Dòng tiền trung tính (CMF {cmf:+.3f})'

        # Relative Strength vs VNINDEX
        cs = al.get('cross_sectional', {})
        if cs:
            rs20 = cs.get('rs_20d_pct', 0)
            beta = cs.get('beta', 1)
            alpha_ann = cs.get('alpha_ann_pct', 0)
            if rs20 > 3:
                narr += f'. RS 20d +{rs20:.1f}% vs VNINDEX — vượt trội thị trường rõ rệt, dòng tiền ưu tiên mã này'
            elif rs20 > 0:
                narr += f'. RS 20d +{rs20:.1f}% vs VNINDEX — mạnh hơn thị trường nhẹ'
            elif rs20 < -3:
                narr += f'. RS 20d {rs20:.1f}% — yếu hơn VNINDEX đáng kể, dòng tiền đang né tránh'
            if abs(alpha_ann) > 5:
                narr += f' (alpha {alpha_ann:+.1f}%/năm, beta {beta:.2f})'

        # Momentum/RSI context
        mom = al.get('momentum', {})
        rsi = mom.get('rsi', 50)
        if rsi > 70:
            narr += f'. RSI {rsi:.0f} — vùng quá mua, cẩn trọng đuổi giá'
        elif rsi < 30:
            narr += f'. RSI {rsi:.0f} — vùng quá bán, có thể hồi kỹ thuật'

        lines.append(narr + '.')
        return '\n'.join(lines)

    # -----------------------------------------------------------------
    def _section_models(self, hm, fc, ky, sig=''):
        """③ Thesis: statistical models agree/disagree — HMM, forecast, Kelly."""
        lines = []

        hmm_cur = hm.get('current', '')
        hmm_prob = hm.get('prob_pct', 0)
        hmm_warning = hm.get('warning', '')
        ens_ret = fc.get('ensemble_ret_pct', 0)
        conf = fc.get('confidence', 0)
        mc_src = fc.get('mc', {}).get('vol_source', '')
        prob_up = fc.get('mc', {}).get('prob_up', 50)

        # Thesis from agreement level
        agree_count = 0
        if 'BULL' in hmm_cur and hmm_prob >= 60: agree_count += 1
        if ens_ret > 1: agree_count += 1
        if ky.get('has_edge'): agree_count += 1
        if conf >= 66: agree_count += 1

        if agree_count >= 3:
            thesis = 'Mô hình thống kê đồng thuận TĂNG'
        elif agree_count >= 2:
            thesis = 'Tín hiệu đang hình thành nhưng chưa chín muồi'
        elif agree_count == 1:
            thesis = 'Tín hiệu yếu — mô hình chưa đồng thuận'
        else:
            thesis = 'Không có tín hiệu rõ từ mô hình'

        lines.append(f'③ {thesis}')

        # HMM narrative
        narr = ''
        hmm_n = hm.get('n_features', 1)
        if 'BULL' in hmm_cur:
            narr = f'HMM {hmm_cur} xác suất {hmm_prob:.0f}%'
            if hmm_prob >= 80:
                narr += ' — xác suất áp đảo, regime ổn định'
            elif hmm_prob >= 60:
                narr += ' — nghiêng tăng nhưng chưa áp đảo'
            else:
                narr += ' — chưa chắc chắn, regime có thể đổi'
            if hmm_n > 1:
                narr += f' (phân tích {hmm_n} chiều: giá, vol, volume, momentum)'
        elif 'BEAR' in hmm_cur:
            narr = f'HMM {hmm_cur} xác suất {hmm_prob:.0f}% — cổ phiếu đang trong trạng thái giảm'
        else:
            narr = f'HMM {hmm_cur} — thị trường đi ngang, chưa rõ hướng'

        # Forecast
        narr += f'. Forecast {ens_ret:+.2f}% trong {fc.get("horizon",10)} phiên'
        if conf >= 66:
            narr += f', Agreement {conf:.0f}% — các mô hình khá đồng thuận'
        elif conf >= 50:
            narr += f', Agreement {conf:.0f}% — tín hiệu đang hình thành, chưa chín muồi'
        else:
            narr += f', Agreement {conf:.0f}% — ngang tung đồng xu, mô hình chưa rõ hướng'

        if mc_src:
            narr += f' (MC dùng {mc_src} vol)'

        # Kelly: chỉ hiển thị nếu được cấp thống kê trade calibration từ hệ thống ngoài.
        if ky.get('disabled'):
            pass
        elif ky.get('has_edge'):
            k_full = ky.get('full_pct', 0)
            k_half = ky.get('half_pct', 0)
            edge = ky.get('edge_pct', 0)
            narr += (f'. Kelly tham khảo từ thống kê ngày (KHÔNG phải edge trade): Half Kelly {k_half:.1f}%, '
                     f'edge {edge:.1f}% — kỳ vọng dương trên mỗi giao dịch')
        elif ky:
            narr += '. Kelly ngày âm — không dùng để sizing chiến lược, xác suất thắng chưa đủ bù thua'

        # Screener confluence
        if sig and sig != 'nan':
            if 'MẠNH' in sig:
                narr += f'. Screener cũng xác nhận tín hiệu MẠNH (đa tiêu chí kỹ thuật) — tăng niềm tin'
            elif 'TÍCH LŨY' in sig:
                narr += f'. Screener: đang TÍCH LŨY — phù hợp với giai đoạn chờ breakout'
            elif 'PHÂN PHỐI' in sig:
                narr += f'. ⚠️ Screener cảnh báo PHÂN PHỐI — mâu thuẫn với tín hiệu tăng, cần thận trọng'

        lines.append(narr + '.')
        return '\n'.join(lines)

    # -----------------------------------------------------------------
    def _section_risk(self, ga, eg, di, ft, rs, lock, sl):
        """④ Honest risk warnings with explanation of WHY each metric matters."""
        lines = []
        items = []

        # GARCH
        persist = ga.get('persistence', 0)
        alpha_g = ga.get('alpha', 0)
        shock = ga.get('shock_sensitivity', '')
        if persist > 0.9:
            p_label = 'IGARCH-like' if persist > 0.97 else 'cao'
            item = f'GARCH persistence {persist:.3f} ({p_label})'
            if alpha_g > 0.15:
                item += f', Alpha {alpha_g:.4f} — nếu có tin xấu bất ngờ, biến động cao sẽ kéo dài rất lâu'
            else:
                item += ' — cú sốc biến động tắt chậm'
            hl = ga.get('half_life')
            if hl and hl > 0:
                item += f' (half-life {hl:.0f} phiên)'
            items.append(item)

        # EGARCH leverage
        if eg and not eg.get('error'):
            gamma = eg.get('gamma', 0)
            if gamma < -0.05:
                items.append(f'EGARCH gamma {gamma:.3f} — tin xấu tăng biến động MẠNH hơn tin tốt (leverage effect)')

        # Distribution
        if not di.get('is_gaussian', True):
            reject_n = di.get('reject_count', 0)
            kurt = ft.get('excess_kurtosis', 0)
            sev = ft.get('severity', '')
            item = f'Phân phối NON-GAUSSIAN ({reject_n}/5 test bác bỏ)'
            if kurt > 5:
                item += f', kurtosis {kurt:.1f} ({sev}) — đuôi rất dày, "thiên nga đen" xảy ra thường hơn mô hình dự đoán'
            elif kurt > 1:
                item += f', kurtosis {kurt:.1f} — cần dùng VaR non-parametric thay vì Gaussian'
            items.append(item)

        # VaR / CVaR context
        var95 = rs.get('VaR_95', 0)
        cvar95 = rs.get('CVaR_95', 0)
        if var95 and cvar95:
            items.append(f'VaR 95%: {var95:.2f}% (95/100 ngày lỗ không quá mức này), CVaR: {cvar95:.2f}% (trung bình 5 ngày tệ nhất)')

        # Lock risk
        prob_l3 = lock.get('prob_loss_gt_3pct', 0)
        prob_lock = lock.get('prob_loss_in_lock_pct', 0)
        mae = sl.get('mae_lock_10pct', 0)
        if prob_l3 > 15 or prob_lock > 50:
            item = f'T+2 settlement lock: {prob_lock:.0f}% khả năng lỗ khi chưa bán được'
            if prob_l3 > 20:
                item += f', {prob_l3:.0f}% khả năng lỗ >3%'
            if mae:
                item += f'. Lịch sử: 90% trường hợp DD trong lock < {abs(mae):.1f}%'
            items.append(item)

        # Build section
        if items:
            lines.append('④ Rủi ro cần lưu ý')
            for item in items:
                lines.append(item + '.')
        else:
            lines.append('④ Rủi ro: Không có cảnh báo đặc biệt.')

        return '\n'.join(lines)

    # -----------------------------------------------------------------
    def _section_conclusion(self, sym, rc, fc, sl, pos, lock):
        """Conclusion: verdict + concrete levels + sizing."""
        score = rc.get('score', 0)
        rating = rc.get('rating', '')
        ens_ret = fc.get('ensemble_ret_pct', 0)
        hold_plan = fc.get('hold_plan_label', '')

        entry = sl.get('entry', 0)
        sl_swing = sl.get('sl_swing', 0)
        tp_opt = sl.get('tp_optimal', sl.get('tp_2r', 0))
        rr_opt = sl.get('rr_optimal', 0)
        shares = pos.get('shares', 0)
        value = pos.get('value', 0)

        lines = []

        # Verdict paragraph
        risk_per = entry - sl_swing if entry and sl_swing else 0
        risk_pct = risk_per / entry * 100 if entry > 0 else 0
        rr_actual = (tp_opt - entry) / risk_per if risk_per > 0 and tp_opt else 0

        if score >= 80:
            verdict = (f'Tổng kết {sym}: Đủ điều kiện SWING BUY — tín hiệu mạnh, rủi ro kiểm soát được')
        elif score >= 65:
            verdict = (f'Tổng kết {sym}: Có thể MUA với quản lý chặt — tín hiệu tích cực nhưng cần kỷ luật')
        elif score >= 50:
            verdict = (f'Tổng kết {sym}: THEO DÕI — tín hiệu đang hình thành nhưng chưa đủ mạnh để vào lệnh')
        elif score >= 35:
            verdict = (f'Tổng kết {sym}: TÍN HIỆU YẾU — không nên mở vị thế mới')
        else:
            verdict = (f'Tổng kết {sym}: TRÁNH — rủi ro cao hơn cơ hội')

        lines.append(verdict + '.')

        # Strategy line
        if entry and sl_swing and tp_opt and score >= 50:
            strat = f'Chiến lược: '
            if score >= 65:
                strat += f'Entry {entry:,.2f}'
            else:
                strat += f'Alert tại {entry:,.2f}, vào khi có xác nhận'
            strat += f'. SL {sl_swing:,.2f} (-{risk_pct:.0f}%) — TP {tp_opt:,.2f} (R:R 1:{rr_actual:.1f}'
            if sl.get('tp_was_capped'):
                strat += ', đã giới hạn theo biên độ thực tế'
            strat += ')'

            if shares > 0 and value > 0:
                strat += f'. Size: {shares:,.0f} cp ({value/1e6:,.1f}M VND)'

            if hold_plan:
                strat += f'. Nắm giữ: {hold_plan}'


            lines.append(strat + '.')

        # Priority ranking hint
        if score >= 75:
            lines.append(f'Ưu tiên CAO trong danh sách.')
        elif score >= 60:
            lines.append(f'Ưu tiên TRUNG BÌNH — cần thêm xác nhận.')

        return '\n'.join(lines)

    def summary_table(self, reports):
        rows = []
        for r in reports:
            rc = r.get('rec',{}); rs = r.get('stats',{}); fc = r.get('fcast',{})
            sl = r.get('sl',{}); ps = r.get('pos',{}); vl = r.get('vol',{})
            hm = r.get('hmm',{}); lock = fc.get('lock_risk',{}); unlock = fc.get('unlock',{})
            action = r.get('action',{}); dq = r.get('data_quality',{}); liq = r.get('liquidity',{}); costs = r.get('costs',{})
            sec = r.get('sector',{}); sec_trend = sec.get('trend',{}); cc = r.get('cross_corr',{})
            top_lead = cc.get('lead_lag',[{}])[0].get('relation','') if cc.get('lead_lag') else ''
            rows.append({
                'Symbol':r['symbol'], 'Exchange':r.get('exchange',''), 'Action':final_action(r),
                'GatePass':final_action(r) in BUY_ACTIONS, 'GateReasons':','.join(action.get('reason_codes',[])),
                'Score':rc.get('score',0), 'Rating':rc.get('rating',''),
                'Timing':rc.get('timing',''), 'HoldPlan':rc.get('hold_plan',''),
                'VNI':rc.get('vni_regime',''), 'Screener':r.get('screener_signal',''),
                'NhomNganh': sec.get('name',''),
                'XuHuongNganh': sec_trend.get('trend_label',''),
                'NganhRet20D%': sec_trend.get('ret_20D_pct'),
                'TuongQuanNganh': cc.get('avg_pairwise_corr'),
                'MaDanDatNganh': top_lead,
                'Sharpe':rs.get('sharpe',0), 'WinRate':rs.get('win_rate_pct',0),
                'MaxDD':rs.get('max_dd_pct',0), 'AnnRet':rs.get('ann_return_pct',0),
                'HMM':hm.get('current',''), 'VolReg':vl.get('regime',''),
                'Forecast':fc.get('consensus',''), 'EnsRet%':fc.get('ensemble_ret_pct',0),
                'Agreement':fc.get('agreement_pct',fc.get('confidence',0)),
                'GrossFc%':costs.get('gross_forecast_pct',fc.get('ensemble_ret_pct',0)),
                'CostRT%':costs.get('roundtrip_cost_pct',0), 'NetFc%':costs.get('net_forecast_pct',0),
                'MCVol':fc.get('mc',{}).get('vol_source',''),
                'LockDD%':lock.get('max_dd_lock_pct',0), 'PLoss3%':lock.get('prob_loss_gt_3pct',0),
                'DQStatus':dq.get('status',''), 'DQScore':dq.get('score',0), 'DQFlags':','.join(dq.get('flags',[])),
                'LiqTier':liq.get('tier',''), 'ADV20_Bn':round(liq.get('adv20_value_vnd',0)/1e9,2),
                'CapacityShares':liq.get('capacity_shares',0), 'GapP95%':liq.get('gap_abs_p95_pct',0),
                'Entry':sl.get('entry',0), 'SL':sl.get('sl_swing',0),
                'TP1':sl.get('tp1', 0), 'TP':sl.get('tp_optimal', sl.get('tp2', sl.get('tp_2r',0))),
                'ModelAgreement%':sl.get('model_agreement_pct',0), 'PWinCalibrated%':sl.get('p_win_blended'),
                'EV_R':sl.get('ev_r_per_trade'), 'RR_NetCost':sl.get('rr_net_cost',0),
                'Shares':ps.get('shares',0), 'PositionValue':ps.get('value',0), 'MaxLossVND':ps.get('max_loss',0),
                'RiskPctNAV':ps.get('risk_pct_nav'),
                'Method':rc.get('scoring_method',''),
                'Analysis': self.generate_commentary(r),
            })
        summary = pd.DataFrame(rows)
        summary.attrs['buy_analysis'] = {
            r['symbol']: self.generate_commentary(r, detailed=True)
            for r in reports if 'error' not in r and final_action(r) in BUY_ACTIONS
        }
        return summary

# ==============================================================
# FORECAST LOGGER
# ==============================================================

class ForecastLogger:
    COLUMNS = ['symbol','run_date','horizon','forecast_dir','forecast_ret_pct',
               'forecast_price','consensus','confidence','actual_price','actual_ret_pct','hit',
               'base_date','base_price','evaluation_method']
    def __init__(self, filepath='forecast_log.csv'):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            pd.DataFrame(columns=self.COLUMNS).to_csv(self.filepath, index=False)
        else:
            existing = pd.read_csv(self.filepath)
            if any(c not in existing.columns for c in self.COLUMNS):
                # Keep the original log before migrating its schema. Old results
                # remain visible but are not mixed with session-based evaluation.
                backup = self.filepath.with_name(self.filepath.name + '.pre_sessions.bak')
                if not backup.exists():
                    backup.write_bytes(self.filepath.read_bytes())
                existing.reindex(columns=self.COLUMNS).to_csv(self.filepath, index=False)
    def log_reports(self, reports, horizon=None):
        rows = []; now = datetime.now().strftime('%Y-%m-%d %H:%M')
        for rp in reports:
            if 'error' in rp: continue
            fc = rp.get('fcast',{}); 
            if not fc: continue
            rows.append({
                'symbol': rp['symbol'], 'run_date': now,
                'horizon': horizon if horizon is not None else fc.get('horizon', CFG.SWING_DEFAULT),
                'forecast_dir': 'UP' if 'TANG' in fc.get('consensus','') else ('DOWN' if 'GIAM' in fc.get('consensus','') else 'FLAT'),
                'forecast_ret_pct': fc.get('ensemble_ret_pct',0), 'forecast_price': fc.get('ensemble_price',0),
                'consensus': fc.get('consensus',''), 'confidence': fc.get('confidence',0),
                'actual_price': '', 'actual_ret_pct': '', 'hit': '',
                'base_date': str(rp['_price_dates'][-1]) if rp.get('_price_dates') else '',
                'base_price': rp['_price_close'][-1] if rp.get('_price_close') else np.nan,
                'evaluation_method': '',
            })
        if rows:
            pd.DataFrame(rows, columns=self.COLUMNS).to_csv(self.filepath, mode='a', header=False, index=False)
            log.info(f"Logged {len(rows)} forecasts")
    def evaluate(self, bridge, days_back=30):
        try: df = pd.read_csv(self.filepath)
        except: return {}
        if df.empty: return {}
        for column in ('hit', 'evaluation_method'):
            df[column] = df[column].astype(object)
        # Log cũ có cả HH:MM:SS, log mới dùng HH:MM. Pandas >= 2 cần
        # khai báo mixed thay vì suy luận một format duy nhất từ dòng đầu.
        df['run_date'] = pd.to_datetime(df['run_date'], format='mixed', errors='coerce')
        pending = df[(df['actual_price']=='')|(df['actual_price'].isna())]
        pending = pending[pending['base_date'].notna() & pending['base_price'].notna()]
        updated = 0
        for sym in pending['symbol'].unique():
            try:
                sym_pending = pending[pending['symbol'] == sym]
                earliest = pd.to_datetime(sym_pending['base_date'], format='mixed').min()
                fetch_days = max(60, days_back, (pd.Timestamp.now().normalize() - earliest.normalize()).days + 10)
                sd = bridge.fetch_ohlcv([sym], days=fetch_days, delay=0.1)
                if sym not in sd: continue
                prices = sd[sym]['close'].sort_index()
                if prices.index.has_duplicates: continue
                for idx in sym_pending.index:
                    base_dt = pd.Timestamp(df.loc[idx,'base_date']).normalize()
                    horizon = int(df.loc[idx,'horizon'])
                    if horizon < 1: continue
                    session_dates = prices.index.normalize()
                    if base_dt not in session_dates: continue
                    # Use completed daily bars only; today's bar may be intraday.
                    fp = prices[(session_dates > base_dt) &
                                (session_dates < pd.Timestamp.now().normalize())]
                    if len(fp) < horizon: continue
                    actual = float(fp.iloc[horizon - 1])
                    entry_price = float(df.loc[idx,'base_price'])
                    if not np.isfinite([actual, entry_price]).all() or entry_price <= 0: continue
                    actual_ret_pct = (actual/entry_price - 1)*100
                    fcast_dir = df.loc[idx,'forecast_dir']
                    hit = 'Y' if ((fcast_dir=='UP' and actual_ret_pct>0) or
                                  (fcast_dir=='DOWN' and actual_ret_pct<0) or
                                  (fcast_dir=='FLAT' and abs(actual_ret_pct)<1)) else 'N'
                    df.loc[idx,'actual_price'] = round(actual,0)
                    df.loc[idx,'actual_ret_pct'] = round(actual_ret_pct,2)
                    df.loc[idx,'hit'] = hit; updated += 1
                    df.loc[idx,'evaluation_method'] = 'observed_sessions_v1'
            except Exception as e: log.warning(f"Eval {sym}: {e}")
        df.to_csv(self.filepath, index=False)
        evaluated = df[df['hit'].isin(['Y','N']) & (df['evaluation_method'] == 'observed_sessions_v1')]
        if evaluated.empty: return {'total': 0}
        total = len(evaluated); hits = len(evaluated[evaluated['hit']=='Y'])
        return {'total': total, 'hits': hits, 'hit_rate_pct': round(hits/total*100,1)}

# ==============================================================
# FORECAST SHEET EXPORT — for report.py chart rendering
# ==============================================================

def _export_forecast_sheet(reports, output_path):
    """Append a 'Forecast' sheet to the existing quant_report.xlsx.
    
    Builds per-symbol rows with:
      - Historical close prices (last 30 trading days)
      - MC-predicted median/CI for the forecast horizon
      - Predicted = ARIMA fitted values (backfill) + MC median (forward)
    
    This sheet is consumed by report.py → Charts.price_forecast() and Charts.ml_forecast().
    """
    from openpyxl import load_workbook
    from openpyxl.styles import Font, Alignment

    rows = []
    for rp in reports:
        if 'error' in rp:
            continue
        sym = rp['symbol']
        fc = rp.get('fcast', {})
        mc_path = fc.get('mc_path')
        if not mc_path:
            continue

        dates_raw = rp.get('_price_dates', [])
        closes_raw = rp.get('_price_close', [])
        if not dates_raw or not closes_raw:
            continue

        # Take last N_HIST trading days for context
        N_HIST = 30
        dates_hist = dates_raw[-N_HIST:]
        closes_hist = closes_raw[-N_HIST:]

        # ARIMA fitted values as "predicted" for historical zone
        arima = rp.get('arima', {})
        fitted_raw = arima.get('fitted', [])
        if fitted_raw and len(fitted_raw) >= N_HIST:
            pred_hist = list(fitted_raw[-N_HIST:])
        else:
            # Fallback: use close as predicted (perfect fit) for historical
            pred_hist = list(closes_hist)

        # Forecast zone: MC median path
        median_path = mc_path['median']
        ci_upper = mc_path['upper']
        ci_lower = mc_path['lower']
        horizon = len(median_path)

        # Generate forecast dates (next business days)
        last_date = pd.Timestamp(dates_hist[-1])
        fc_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon)

        forecast_start_idx = len(dates_hist)

        # Historical rows
        for i, (d, c, p) in enumerate(zip(dates_hist, closes_hist, pred_hist)):
            rows.append({
                'Symbol': sym,
                'Date': pd.Timestamp(d),
                'Close': round(float(c), 2),
                'Predicted': round(float(p), 2),
                'Upper': None,
                'Lower': None,
                'Forecast_Start': forecast_start_idx if i == 0 else None,
            })

        # Forecast rows (close = NaN → tells report.py this is OOS)
        for i in range(horizon):
            rows.append({
                'Symbol': sym,
                'Date': fc_dates[i],
                'Close': None,  # NaN signals forecast zone
                'Predicted': round(float(median_path[i]), 2),
                'Upper': round(float(ci_upper[i]), 2),
                'Lower': round(float(ci_lower[i]), 2),
                'Forecast_Start': None,
            })

    if not rows:
        log.info("No forecast data to export")
        return

    df_fc = pd.DataFrame(rows)

    # Write to existing workbook
    try:
        wb = load_workbook(output_path)
        # Remove old Forecast sheet if exists
        if 'Forecast' in wb.sheetnames:
            del wb['Forecast']
        ws = wb.create_sheet('Forecast')
    except FileNotFoundError:
        from openpyxl import Workbook
        wb = Workbook()
        ws = wb.active
        ws.title = 'Forecast'

    # Header
    HEADER_FONT = Font(name='Arial', bold=True, size=10, color='FFFFFF')
    from openpyxl.styles import PatternFill
    HEADER_FILL = PatternFill('solid', fgColor='1F4E79')
    headers = ['Symbol', 'Date', 'Close', 'Predicted', 'Upper', 'Lower', 'Forecast_Start']
    for ci, h in enumerate(headers, 1):
        cell = ws.cell(row=1, column=ci, value=h)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal='center')

    # Data rows
    for ri, row in enumerate(rows, 2):
        ws.cell(row=ri, column=1, value=row['Symbol'])
        ws.cell(row=ri, column=2, value=row['Date'])
        ws.cell(row=ri, column=3, value=row['Close'])
        ws.cell(row=ri, column=4, value=row['Predicted'])
        ws.cell(row=ri, column=5, value=row['Upper'])
        ws.cell(row=ri, column=6, value=row['Lower'])
        ws.cell(row=ri, column=7, value=row['Forecast_Start'])

    # Format date column
    for ri in range(2, len(rows) + 2):
        ws.cell(row=ri, column=2).number_format = 'YYYY-MM-DD'

    # Column widths
    for ci, w in enumerate([10, 14, 12, 12, 12, 12, 14], 1):
        from openpyxl.utils import get_column_letter
        ws.column_dimensions[get_column_letter(ci)].width = w

    ws.freeze_panes = 'A2'
    wb.save(output_path)
    log.info(f"  Forecast sheet exported: {len(rows)} rows for {df_fc['Symbol'].nunique()} symbols → {output_path}")


# ==============================================================
# ENHANCED EXCEL FALLBACK (if backtest_engine not available)
# ==============================================================

# Excel reporting helpers: summary stays compact; BUY_ONLY carries the quant thesis.
BUY_ACTIONS = frozenset({'BUY_NOW', 'BUY_SETUP'})
RATINGS = {'BUY_NOW': 'MUA NGAY', 'BUY_SETUP': 'CHỜ ĐIỂM MUA',
           'WATCH': 'THEO DÕI', 'AVOID': 'TRÁNH'}


def final_action(r):
    if 'error' in r:
        return 'AVOID'
    action = r.get('action', {}).get('action', 'AVOID')
    return action if action in RATINGS else 'AVOID'


def action_rating(r):
    return RATINGS[final_action(r)]


def num(value, digits=2, suffix=''):
    try:
        value = float(value)
        return f'{value:,.{digits}f}{suffix}' if math.isfinite(value) else 'N/A'
    except (TypeError, ValueError):
        return 'N/A'


def above(value, threshold):
    try:
        return float(value) > threshold
    except (TypeError, ValueError):
        return False


def below(value, threshold):
    try:
        return float(value) < threshold
    except (TypeError, ValueError):
        return False


def commentary(r, detailed=False):
    action = final_action(r)
    if 'error' in r:
        return f'Không đủ dữ liệu: {r["error"]}. Action = {action}.'
    stats = r.get('stats', {}); alpha = r.get('alpha', {})
    rel = alpha.get('cross_sectional', {}); momentum = alpha.get('momentum', {})
    volume = alpha.get('volume', {}); trend = r.get('trend', {}); flow = r.get('flow', {})
    fc = r.get('fcast', {}); costs = r.get('costs', {}); hm = r.get('hmm', {})
    vol = r.get('vol', {}); ga = r.get('garch', {}).get('garch', {})
    eg = r.get('garch', {}).get('egarch', {}); sl = r.get('sl', {}); pos = r.get('pos', {})
    rc = r.get('rec', {}); dist = r.get('dist', {}); lock = fc.get('lock_risk', {})
    agreement = fc.get('agreement_pct', fc.get('confidence'))
    extended = above(momentum.get('rsi'), 70) or above(alpha.get('mean_reversion', {}).get('z_score'), 2)
    short = (f'RS20 {num(rel.get("rs_20d_pct"))}%, CMF {num(flow.get("cmf"))}; HMM {hm.get("current", "N/A")}. '
             f'Net forecast {num(costs.get("net_forecast_pct"))}%, Model Agreement {num(agreement, 0)}%; '
             f'entry {"extended" if extended else "cần đối chiếu hỗ trợ/kháng cự"}. Action = {action}.')
    if not detailed or action not in BUY_ACTIONS:
        return short

    # BUY_ONLY narrative: a readable thesis, while Summary keeps the short note above.
    symbol = r.get('symbol', '?')
    sr = r.get('sr', {})
    supports = ', '.join(num(x.get('price'), 0) for x in sr.get('supports', [])[:2]) or 'N/A'
    resistances = ', '.join(num(x.get('price'), 0) for x in sr.get('resistances', [])[:2]) or 'N/A'
    state = hm.get('current', 'N/A')
    prob_text = ', '.join(f'{k} {num(v, 1)}%' for k, v in hm.get('state_probs', {}).items()) or 'chưa có xác suất tin cậy'
    components = fc.get('component_returns', {}); weights = fc.get('weights', {})
    mixed = any(above(v, 0) for v in components.values()) and any(below(v, 0) for v in components.values())
    forecast_read = ('MC/Momentum đang nâng forecast trong khi Mean Reversion thận trọng hơn' if mixed
                     else 'các component hiện nghiêng cùng hướng, nên ensemble có nền tảng đồng nhất hơn')
    construction = sl.get('construction', {})
    constraints = {'risk': pos.get('shares_by_risk'), 'allocation': pos.get('shares_by_allocation'), 'liquidity': pos.get('shares_by_liquidity')}
    valid = {k: v for k, v in constraints.items() if isinstance(v, (int, float)) and v > 0}
    binding = ', '.join(k for k, v in valid.items() if v == min(valid.values())) if valid else 'chưa xác định'
    verdict = dist.get('verdict', {})
    tail_text = ('return distribution có fat-tail rõ, vì vậy VaR/CVaR lịch sử và GARCH-t đáng tin hơn giả định Gaussian' if verdict.get('is_gaussian') is False
                 else 'tail risk chưa cho thấy tín hiệu cực đoan, nhưng vẫn cần đọc cùng VaR/CVaR và gap risk')
    entry_text = 'đang extended, entry quality thấp' if extended else 'chưa extended rõ, entry quality ở mức trung bình'
    setup_quality = 'cao' if above(rc.get('score'), 74) else 'trung bình'
    risk_text = 'cao' if sl.get('lock_risk_flag') or above(lock.get('prob_loss_gt_3pct'), 20) else 'đáng theo dõi'
    blocks = [
        f'### {symbol} — Quality {setup_quality}, edge tồn tại nhưng timing cần được kiểm soát',
        f'{symbol} có historical return profile đáng chú ý: CAGR {num(stats.get("ann_return_pct"))}%, Sharpe {num(stats.get("sharpe"))}, Sortino {num(stats.get("sortino"))}, Calmar {num(stats.get("calmar"))} và MaxDD {num(stats.get("max_dd_pct"))}%. Chất lượng lựa chọn vì vậy không chỉ nằm ở mức tăng, mà ở hiệu quả so với volatility, downside và drawdown. RS20 {num(rel.get("rs_20d_pct"))}% so với VNINDEX, beta {num(rel.get("beta"))} và alpha proxy {num(rel.get("alpha_ann_pct"))}% cho thấy {"sức mạnh gần đây có phần độc lập với beta thị trường" if above(rel.get("rs_20d_pct"), 0) and below(rel.get("beta"), .8) else "relative edge chưa đủ tách khỏi biến động thị trường"}.',
        f'Bức tranh hiện tại được củng cố bởi volatility và money flow: Vol Ratio {num(vol.get("vol_ratio"))}, Vol Percentile {num(vol.get("vol_pctile"), 1)}%, CMF {num(flow.get("cmf"))} và OBV {volume.get("obv_trend", "N/A")}. {"Đây là confluence của volatility contraction, buying pressure và relative strength" if below(vol.get("vol_ratio"), .7) and above(flow.get("cmf"), 0) and above(rel.get("rs_20d_pct"), 0) else "Các tín hiệu chưa hoàn toàn đồng thuận, nên money flow cần được xác nhận thêm bằng price action"}.',
        f'Tuy nhiên trend quality mới là phần quyết định regime confirmation. ER {num(trend.get("er"))}, slope {num(trend.get("slope_pct"))}%/phiên và R² {num(trend.get("r2"))} cho thấy {"giá còn đi qua nhiều noise dù hướng chính tích cực" if below(trend.get("er"), .4) else "đường giá có cấu trúc tương đối liền mạch"}. HMM hiện ở {state}, với state probabilities {prob_text}; vì vậy stock selection quality có thể cao hơn regime confirmation. Đây là setup sớm hơn là breakout đã được xác nhận hoàn toàn.',
        f'Momentum đang tạo ra xung đột với execution: RSI {num(momentum.get("rsi"), 1)}, ROC {num(momentum.get("roc_10d"))}% và z-score {num(alpha.get("mean_reversion", {}).get("z_score"))}. {"RSI cao khiến giá bị extended và làm downside asymmetry xấu đi; selection quality tốt nhưng entry quality bắt đầu xấu" if extended else f"Momentum chưa cho thấy extension cực đoan, nhưng entry vẫn cần đặt quanh hỗ trợ {supports} thay vì đuổi theo giá"}. Kháng cự gần nhất là {resistances}.',
        f'Forecast {num(fc.get("ensemble_ret_pct"))}% trong {num(fc.get("horizon"), 0)} phiên đến từ MC {num(components.get("mc_ret_pct"))}% (w {num(weights.get("mc"), 2)}), Mean Reversion {num(components.get("mr_ret_pct"))}% (w {num(weights.get("mean_reversion"), 2)}) và Momentum {num(components.get("mom_ret_pct"))}% (w {num(weights.get("momentum"), 2)}). {forecast_read}. Gross {num(costs.get("gross_forecast_pct", fc.get("ensemble_ret_pct")))}% trừ cost {num(costs.get("roundtrip_cost_pct"))}% còn net {num(costs.get("net_forecast_pct"))}%. Agreement {num(agreement, 0)}% không phải xác suất lợi nhuận; mức đồng thuận {"còn thấp, nên forecast dương chưa đủ để chase" if below(agreement, 66) else "tương đối tốt"}.',
        f'Risk Engine phản ứng với mức conviction đó bằng trade construction có kiểm soát. Entry {num(sl.get("entry"), 0)}, ATR {num(sl.get("atr"))}, SL {num(sl.get("sl_swing"), 0)} sau khi so ATR stop với VaR-adjusted floor {num(construction.get("stop_floor_pct"))}%. TP1 {num(sl.get("tp1"), 0)}; base TP2 {num(construction.get("base_tp2"), 0)} được kéo về {num(construction.get("conviction_tp2"), 0)} rồi chốt final TP2 {num(sl.get("tp2"), 0)} vì conviction {num(sl.get("conviction"))}. Đây là risk-adjusted/conviction-adjusted target, không gọi là optimized khi internal optimization đang tắt. R:R TP1 1:{num(sl.get("rr1"))}, TP2 1:{num(sl.get("rr2"))}; expected holding {fc.get("hold_plan_label", "N/A")}.',
        f'Tail và settlement risk hiện ở mức {risk_text}. {tail_text}. VaR95 {num(stats.get("VaR_95"))}% và CVaR95 {num(stats.get("CVaR_95"))}%; GARCH persistence {num(ga.get("persistence"), 3)}, alpha {num(ga.get("alpha"), 3)} và EGARCH gamma {num(eg.get("gamma"), 3)} cho thấy {"negative shock có thể làm volatility tăng mạnh và kéo dài" if below(eg.get("gamma"), -.05) or above(ga.get("persistence"), .95) else "shock risk cần tiếp tục theo dõi"}. Trong settlement lock {num(lock.get("lock_sessions", sl.get("lock_sessions")), 0)} phiên, MC lock drawdown {num(lock.get("max_dd_lock_pct"))}% và historical MAE {num(sl.get("mae_lock_10pct"))}%; {"stop có thể không phải guaranteed loss boundary trước khi vị thế được xử lý" if sl.get("lock_risk_flag") else "lock risk chưa cho thấy cảnh báo vượt stop rõ ràng"}.',
        f'Position sizing cho phép {num(constraints.get("risk"), 0)} cp theo risk, {num(constraints.get("allocation"), 0)} cp theo allocation và {num(constraints.get("liquidity"), 0)} cp theo liquidity; vị thế cuối {num(pos.get("shares"), 0)} cp, giá trị {num(pos.get("value"), 0)} VND ({num(pos.get("pct_acct"), 1)}% NAV), theoretical loss tại SL {num(pos.get("max_loss"), 0)} VND ({num(pos.get("risk_pct_nav"))}% NAV). Binding constraint là {binding}. Score {num(rc.get("score"), 0)}/100 là composite ranking/quality score, không phải xác suất profit.',
        f'### Overall\n{symbol} — {num(rc.get("score"), 0)}/100 | selection edge {"mạnh" if above(rc.get("score"), 74) else "trung bình"} | entry quality {entry_text}. Action Gate = {action}. Luận điểm mua đến từ historical edge, money flow và forecast net dương; điểm cần kỷ luật là {"HMM chưa xác nhận BULL, Agreement thấp và RSI extension" if below(agreement, 66) else "giữ nguyên kỷ luật quanh entry và settlement risk"}. Đây là một swing setup cần chờ pullback/xác nhận khi timing chưa tương xứng, không phải kết luận buy-and-hold dài hạn.'
    ]
    return '\n\n'.join(blocks)


def export_buy_workbook(summary_df, output_path, reports=None):
    """Keep summary metrics intact; details live exclusively in BUY_ONLY.

    Optional reports are authoritative. String-only attrs support legacy callers
    passing summary_table() directly without changing its tabular schema.
    """
    import pandas as pd
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    frame = summary_df.copy()
    analysis = dict(summary_df.attrs.get('buy_analysis', {}))
    if reports is not None:
        by_symbol = {r['symbol']: r for r in reports}
        for index, row in frame.iterrows():
            r = by_symbol.get(row.get('Symbol'), {})
            action = final_action(r)
            if action in BUY_ACTIONS:
                analysis[row['Symbol']] = commentary(r, detailed=True)
    if 'Action' not in frame:
        frame['Action'] = 'AVOID'
    buy = frame[frame['Action'].isin(BUY_ACTIONS)]
    if 'Score' in buy:
        buy = buy.sort_values('Score', ascending=False)
    columns = [('Mã', 'Symbol'), ('Action', 'Action'), ('Score', 'Score'), ('Rating', 'Rating'),
               ('VNI Regime', 'VNI'), ('Sector', 'NhomNganh'), ('Net Forecast %', 'NetFc%'),
               ('Model Agreement %', 'Agreement'), ('HMM', 'HMM'), ('Liquidity Tier', 'LiqTier'),
               ('Entry', 'Entry'), ('SL', 'SL'), ('TP1', 'TP1'), ('TP2', 'TP'), ('Shares', 'Shares'),
               ('Position Value', 'PositionValue'), ('Risk % NAV', 'RiskPctNAV'),
               ('Holding Period', 'HoldPlan'), ('Analysis', 'Analysis')]
    wb = Workbook(); wb.remove(wb.active)

    def sheet(name, headers, rows, detailed=False):
        ws = wb.create_sheet(name); ws.append(headers)
        for row in rows:
            ws.append([None if v is None or (not isinstance(v, (str, list, dict)) and pd.isna(v)) else v for v in row])
        for cell in ws[1]:
            cell.font = Font(bold=True, color='FFFFFF')
            cell.fill = PatternFill('solid', fgColor='1F4E78')
        for j, header in enumerate(headers, 1):
            ws.column_dimensions[get_column_letter(j)].width = (190 if detailed else 65) if header == 'Analysis' else 18
            for row in ws.iter_rows(min_row=2, min_col=j, max_col=j):
                row[0].alignment = Alignment(wrap_text=True, vertical='top')
                if header == 'Analysis' and detailed:
                    row[0].font = Font(name='Arial', size=9)
                if header in ('Entry', 'SL', 'TP1', 'TP2', 'Shares', 'Position Value'):
                    row[0].number_format = '#,##0.##'
        for i in range(2, ws.max_row + 1):
            ws.row_dimensions[i].height = 409 if detailed else 55
        ws.freeze_panes = 'C2'; ws.auto_filter.ref = ws.dimensions
        return ws

    sheet('Summary', list(frame.columns), frame.itertuples(index=False, name=None))
    rows = []
    for _, row in buy.iterrows():
        values = row.to_dict()
        thesis = analysis.get(row['Symbol'], '')
        values['Analysis'] = thesis if f'FINAL: Action Gate {row["Action"]};' in thesis else (
            f'Action Gate {row["Action"]}. Detailed pipeline data unavailable; regenerate from reports for the full thesis.')
        rows.append([values.get(key) for _, key in columns])
    sheet('BUY_ONLY', [name for name, _ in columns], rows, detailed=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
    return output_path


def _export_enhanced_inline(summary_df, output_path):
    """Compatibility Excel entry point using the same two-level workbook."""
    return export_buy_workbook(summary_df, output_path)


# ==============================================================
# MAIN
# ==============================================================

# ==============================================================
# BUY-ONLY SUMMARY EXCEL — Tổng hợp chỉ mã khuyến nghị MUA
# ==============================================================

def _export_buy_summary(summary_df, output_path='buy_summary.xlsx', vni_regime='NEUTRAL', reports=None):
    """Export all-stock Summary plus final-action-gated BUY_ONLY; prices in VND."""
    return export_buy_workbook(summary_df, output_path, reports=reports)


# ==============================================================
# M-SECTOR-FLOW: DÒNG TIỀN NGÀNH (từ Watchlist.xlsx) -> Excel
# ==============================================================
#
# Đo dòng tiền theo ngành = tỷ trọng GTGD của các mã TĂNG / ĐỨNG / GIẢM
# giá trong ngành, weight theo giá trị giao dịch (close * volume) phiên
# gần nhất. Dùng lại SectorEngine.WATCHLIST_SECTORS (đã map sẵn từ file
# Watchlist.xlsx) — không phải đọc lại Excel mỗi lần chạy.
#
# Muốn nạp trực tiếp từ 1 file Watchlist.xlsx khác (không dùng bản
# hard-code trong SectorEngine) thì dùng load_watchlist_sectors_from_excel().
# ==============================================================

FLOW_FLAT_THRESHOLD = 0.0015  # |%thay đổi| < 0.15% => coi là "đứng giá"


def load_watchlist_sectors_from_excel(path='Watchlist.xlsx'):
    """Đọc trực tiếp Watchlist.xlsx (mỗi cột = 1 ngành, mã xếp dọc bên dưới).

    Dùng khi muốn cập nhật danh mục theo ngành mà không sửa code
    (SectorEngine.WATCHLIST_SECTORS), chỉ cần sửa file Excel.
    """
    raw = pd.read_excel(path, header=0)
    sectors = {}
    for col in raw.columns:
        name = str(col).strip()
        if not name or name.lower().startswith('unnamed'):
            continue
        syms = raw[col].dropna().astype(str).str.strip().str.upper()
        syms = [s for s in syms if s.isalpha() and 2 <= len(s) <= 4]
        if syms:
            sectors[name] = syms
    return sectors


def compute_sector_cashflow(ohlcv_data: dict, sectors: dict = None) -> pd.DataFrame:
    """Tính bảng dòng tiền theo ngành từ dữ liệu OHLCV đã fetch.

    Args:
        ohlcv_data: {symbol: DataFrame[open,high,low,close,volume]} — output
            của ScreenerBridge.fetch_ohlcv(). Cần >= 2 phiên/mã.
        sectors: {ten_nganh: [ma,...]}. None -> dùng SectorEngine.WATCHLIST_SECTORS.

    Returns:
        DataFrame cột: Nganh, SoMa, PctChange, GTGD_ty, TangGTGD_ty,
        DungGTGD_ty, GiamGTGD_ty, TangPct, DungPct, GiamPct, DongTienRong_ty.
        Sắp xếp giảm dần theo GTGD_ty.
    """
    if sectors is None:
        sectors = SectorEngine.WATCHLIST_SECTORS

    rows = []
    for sector_name, symbols in sectors.items():
        last_val = prev_val = 0.0
        gtgd = up_val = flat_val = down_val = 0.0
        n_ma = 0

        for sym in symbols:
            df = ohlcv_data.get(sym)
            if df is None or len(df) < 2:
                continue
            closes = df['close'].astype(float)
            vol_today = float(df['volume'].iloc[-1])
            last_close, prev_close = float(closes.iloc[-1]), float(closes.iloc[-2])
            if prev_close <= 0 or pd.isna(prev_close):
                continue

            pct = (last_close - prev_close) / prev_close
            # Pipeline V6 đã chuẩn hóa close về VND tại data boundary.
            value_today = last_close * vol_today
            value_prev = prev_close * vol_today

            gtgd += value_today
            last_val += value_today
            prev_val += value_prev
            n_ma += 1

            if pct > FLOW_FLAT_THRESHOLD:
                up_val += value_today
            elif pct < -FLOW_FLAT_THRESHOLD:
                down_val += value_today
            else:
                flat_val += value_today

        if n_ma == 0:
            continue

        pct_change = (last_val - prev_val) / prev_val if prev_val else 0.0
        total_flow = up_val + flat_val + down_val

        rows.append({
            'Nganh': sector_name,
            'SoMa': n_ma,
            'PctChange': pct_change,
            'GTGD_ty': gtgd / 1e9,
            'TangGTGD_ty': up_val / 1e9,
            'DungGTGD_ty': flat_val / 1e9,
            'GiamGTGD_ty': down_val / 1e9,
            'TangPct': (up_val / total_flow) if total_flow else 0.0,
            'DungPct': (flat_val / total_flow) if total_flow else 0.0,
            'GiamPct': (down_val / total_flow) if total_flow else 0.0,
            'DongTienRong_ty': (up_val - down_val) / 1e9,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values('GTGD_ty', ascending=False).reset_index(drop=True)
    return out


def _export_sector_flow_excel(flow_df: pd.DataFrame, output_path='sector_cashflow.xlsx'):
    """Xuất bảng dòng tiền ngành ra Excel — 1 sheet, có data-bar cho GTGD
    và cho tỷ trọng dòng tiền tăng/giảm, style đồng bộ với _export_buy_summary().
    """
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.formatting.rule import DataBarRule
    from openpyxl.utils import get_column_letter

    if flow_df.empty:
        log.info("Không tính được dòng tiền ngành nào — skip export.")
        return output_path

    now = datetime.now()
    FONT_TITLE = Font(name='Arial', bold=True, size=14, color='1F3864')
    FONT_SUB = Font(name='Arial', size=10, color='666666')
    FONT_HEADER = Font(name='Arial', bold=True, size=10, color='FFFFFF')
    FONT_BOLD = Font(name='Arial', bold=True, size=10, color='000000')
    FONT_NORMAL = Font(name='Arial', size=10, color='333333')
    FONT_RED = Font(name='Arial', bold=True, size=10, color='CC0000')
    FONT_GREEN = Font(name='Arial', bold=True, size=10, color='006100')
    FILL_HEADER = PatternFill('solid', fgColor='2E4057')
    FILL_ALT1 = PatternFill('solid', fgColor='F5F7FA')
    FILL_ALT2 = PatternFill('solid', fgColor='FFFFFF')
    FILL_LIGHT_GREEN = PatternFill('solid', fgColor='E2EFDA')
    FILL_LIGHT_RED = PatternFill('solid', fgColor='FFC7CE')
    BORDER = Border(bottom=Side(style='thin', color='CCCCCC'), right=Side(style='thin', color='CCCCCC'),
                    left=Side(style='thin', color='CCCCCC'), top=Side(style='thin', color='CCCCCC'))
    ALIGN_C = Alignment(horizontal='center', vertical='center')
    ALIGN_L = Alignment(horizontal='left', vertical='center')

    wb = Workbook()
    ws = wb.active; ws.title = 'Dòng tiền ngành'
    ws.merge_cells('B2:L2')
    ws['B2'].value = f'DÒNG TIỀN THEO NGÀNH — {now.strftime("%d/%m/%Y %H:%M")}'; ws['B2'].font = FONT_TITLE
    ws.merge_cells('B3:L3')
    ws['B3'].value = ('Dòng tiền = tỷ trọng GTGD của mã Tăng/Đứng/Giảm giá trong ngành, '
                       'weight theo giá trị giao dịch phiên gần nhất  |  Sắp xếp theo GTGD giảm dần')
    ws['B3'].font = FONT_SUB

    cols = [
        ('Ngành', 'Nganh', 26, ALIGN_L),
        ('Số mã', 'SoMa', 8, ALIGN_C),
        ('% Ngành', 'PctChange', 10, ALIGN_C),
        ('GTGD (tỷ)', 'GTGD_ty', 12, ALIGN_C),
        ('GTGD Tăng (tỷ)', 'TangGTGD_ty', 14, ALIGN_C),
        ('GTGD Đứng (tỷ)', 'DungGTGD_ty', 14, ALIGN_C),
        ('GTGD Giảm (tỷ)', 'GiamGTGD_ty', 14, ALIGN_C),
        ('% Tăng', 'TangPct', 9, ALIGN_C),
        ('% Đứng', 'DungPct', 9, ALIGN_C),
        ('% Giảm', 'GiamPct', 9, ALIGN_C),
        ('Dòng tiền ròng (tỷ)', 'DongTienRong_ty', 16, ALIGN_C),
    ]
    header_row = 5
    ws.column_dimensions['A'].width = 2
    for ci, (label, _, width, _) in enumerate(cols, 2):
        c = ws.cell(row=header_row, column=ci, value=label)
        c.font = FONT_HEADER; c.fill = FILL_HEADER; c.alignment = ALIGN_C; c.border = BORDER
        ws.column_dimensions[get_column_letter(ci)].width = width

    start_row = header_row + 1
    for ri, (_, row) in enumerate(flow_df.iterrows(), start_row):
        net = row['DongTienRong_ty']
        fill_row = FILL_LIGHT_GREEN if net > 0 else (FILL_LIGHT_RED if net < 0 else (FILL_ALT1 if ri % 2 == 0 else FILL_ALT2))
        for ci, (_, key, _, align) in enumerate(cols, 2):
            c = ws.cell(row=ri, column=ci); c.border = BORDER; c.fill = fill_row; c.alignment = align
            val = row[key]
            if key == 'Nganh':
                c.value = val; c.font = FONT_BOLD
            elif key == 'SoMa':
                c.value = int(val); c.font = FONT_NORMAL
            elif key == 'PctChange':
                c.value = val; c.number_format = '+0.00%;-0.00%'; c.font = FONT_GREEN if val >= 0 else FONT_RED
            elif key in ('TangPct', 'DungPct', 'GiamPct'):
                c.value = val; c.number_format = '0%'; c.font = FONT_NORMAL
            elif key == 'DongTienRong_ty':
                c.value = round(val, 1); c.number_format = '#,##0.0'; c.font = FONT_GREEN if val >= 0 else FONT_RED
            else:
                c.value = round(val, 1); c.number_format = '#,##0.0'; c.font = FONT_NORMAL

    end_row = start_row + len(flow_df) - 1

    # Data bar cho GTGD (cột E = index 5)
    gtgd_col = get_column_letter(5)
    ws.conditional_formatting.add(
        f'{gtgd_col}{start_row}:{gtgd_col}{end_row}',
        DataBarRule(start_type='min', end_type='max', color='2E75B6')
    )
    # Data bar xanh cho % Tăng, đỏ cho % Giảm (cột H, J = index 8, 10)
    tang_col = get_column_letter(8); giam_col = get_column_letter(10)
    ws.conditional_formatting.add(
        f'{tang_col}{start_row}:{tang_col}{end_row}',
        DataBarRule(start_type='min', end_type='max', color='26A69A')
    )
    ws.conditional_formatting.add(
        f'{giam_col}{start_row}:{giam_col}{end_row}',
        DataBarRule(start_type='min', end_type='max', color='EF5350')
    )

    ws.freeze_panes = f'C{start_row}'
    ws.auto_filter.ref = f"B{header_row}:{get_column_letter(len(cols)+1)}{end_row}"
    wb.save(output_path)
    log.info(f"\n📁 Sector cashflow exported: {output_path} ({len(flow_df)} ngành)")
    return output_path


def export_sector_cashflow(ohlcv_data: dict = None, symbols: list = None,
                            sectors: dict = None, watchlist_path: str = None,
                            output_path: str = 'sector_cashflow.xlsx',
                            source=None, days: int = 15,
                            push_to_gsheet: bool = True,
                            gsheet_name: str = 'LỌC CỔ PHIẾU',
                            gsheet_credentials: str = 'credentials.json',
                            gsheet_id: str = None,
                            gsheet_worksheet: str = 'Analysis'):
    """Entry point tiện dùng độc lập hoặc gọi từ run_pipeline().

    - Nếu truyền sẵn `ohlcv_data` (vd. tái sử dụng data đã fetch trong
      run_pipeline) -> dùng luôn, không fetch lại.
    - Nếu không, sẽ tự fetch OHLCV cho toàn bộ mã trong `sectors`
      (hoặc watchlist_path, hoặc SectorEngine.WATCHLIST_SECTORS mặc định)
      qua ScreenerBridge (KBS -> VCI).
    - `push_to_gsheet=True` (mặc định) -> đẩy thẳng 3 cột (Ngành, %Thay đổi,
      GTGD) + heatmap lên Google Sheet, tab `gsheet_worksheet` ('Analysis').
    """
    if sectors is None and watchlist_path:
        sectors = load_watchlist_sectors_from_excel(watchlist_path)
    if sectors is None:
        sectors = SectorEngine.WATCHLIST_SECTORS

    if ohlcv_data is None:
        all_syms = sorted({s for syms in sectors.values() for s in syms})
        missing = [s for s in all_syms if s not in (symbols or [])] if symbols else all_syms
        bridge = ScreenerBridge(source=source)
        log.info(f"[SECTOR-FLOW] Fetch OHLCV cho {len(missing)} mã...")
        ohlcv_data = bridge.fetch_ohlcv(missing, days=days) if missing else {}

    flow_df = compute_sector_cashflow(ohlcv_data, sectors)
    if not flow_df.empty:
        print(f"\n{'='*70}\n  DÒNG TIỀN NGÀNH (Top by GTGD)\n{'='*70}")
        print(flow_df[['Nganh', 'PctChange', 'GTGD_ty', 'TangPct', 'GiamPct', 'DongTienRong_ty']]
              .to_string(index=False))

    result = {'excel_path': None, 'gsheet_url': None}
    if output_path:
        result['excel_path'] = _export_sector_flow_excel(flow_df, output_path)
    if push_to_gsheet:
        result['gsheet_url'] = export_sector_cashflow_to_gsheet(
            flow_df, spreadsheet_name=gsheet_name, credentials_file=gsheet_credentials,
            spreadsheet_id=gsheet_id, worksheet_name=gsheet_worksheet,
        )
    return result


def _export_recommendations_parquet(summary_df, output_path='cache/recommendations_latest.parquet',
                                     vni_regime='NEUTRAL'):
    """
    Xuất recommendations ra parquet để valuation.py đọc.

    OUTPUT SCHEMA (chuẩn cho valuation.py):
        symbol     : Mã CK (uppercase)
        action     : BUY / HOLD / SELL
        score      : 0-100 (giữ nguyên scale từ pipeline)
        entry      : Giá vào lệnh (VND)
        stop       : Giá cắt lỗ (VND)
        target     : Giá chốt lời (VND, dùng TP tối ưu từ grid search/expectancy, cột TP)
        regime     : HMM regime + VNI regime (concatenated)
        rating     : Rating gốc từ pipeline (⭐⭐⭐⭐⭐ SWING BUY ...)
        forecast   : TANG / GIAM / SIDEWAYS
        confidence : 0-1 từ forecast engine

    LOGIC ACTION (consistent với _export_buy_summary):
        - BUY  : Action Gate = BUY_NOW hoặc BUY_SETUP
        - HOLD : WATCH
        - AVOID: hard gate fail hoặc tín hiệu yếu

    Args:
        summary_df: output của QuantPipeline.summary_table()
        output_path: đường dẫn parquet (default: cache/recommendations_latest.parquet)
        vni_regime: VNI regime hiện tại (đính kèm metadata)
    """
    if summary_df is None or summary_df.empty:
        log.warning("Summary df trống — skip recommendations parquet export.")
        return None

    # Map score → action
    def _score_to_action(row):
        action = str(row.get('Action', 'AVOID')).upper()
        if action in ('BUY_NOW','BUY_SETUP'): return 'BUY'
        if action == 'WATCH': return 'HOLD'
        return 'AVOID'

    # Build standardized dataframe
    out = pd.DataFrame()
    out['symbol'] = summary_df['Symbol'].astype(str).str.upper().str.strip()
    out['action'] = summary_df.apply(_score_to_action, axis=1)
    out['score'] = pd.to_numeric(summary_df['Score'], errors='coerce').fillna(0)
    out['entry'] = pd.to_numeric(summary_df['Entry'], errors='coerce')
    out['stop'] = pd.to_numeric(summary_df.get('SL', 0), errors='coerce')
    # Dùng TP (TP tối ưu hóa qua grid search + expectancy) làm target chính
    out['target'] = pd.to_numeric(summary_df.get('TP', 0), errors='coerce')

    # Regime: combine HMM + VNI để valuation/report có context đầy đủ
    hmm_regime = summary_df.get('HMM', pd.Series(['']*len(summary_df))).astype(str)
    vni_col = summary_df.get('VNI', pd.Series([vni_regime]*len(summary_df))).astype(str)
    out['regime'] = hmm_regime + ' | VNI:' + vni_col

    # Metadata cho audit & valuation context
    out['rating'] = summary_df.get('Rating', '').astype(str)
    out['forecast'] = summary_df.get('Forecast', '').astype(str)
    out['agreement'] = pd.to_numeric(summary_df.get('Agreement', 0), errors='coerce').fillna(0) / 100.0
    out['confidence'] = pd.to_numeric(summary_df.get('PWinCalibrated%', np.nan), errors='coerce') / 100.0
    out['sharpe'] = pd.to_numeric(summary_df.get('Sharpe', 0), errors='coerce').fillna(0)
    out['vol_regime'] = summary_df.get('VolReg', '').astype(str)
    out['screener_signal'] = summary_df.get('Screener', '').astype(str)

    # Timestamp khi pipeline chạy (để valuation.py biết freshness)
    out['generated_at'] = datetime.now().isoformat(timespec='seconds')
    out['vni_regime'] = vni_regime

    # Cleanup: drop rows không có symbol hoặc entry
    before = len(out)
    out = out.dropna(subset=['symbol', 'entry'])
    out = out[out['symbol'].str.len() > 0]
    after = len(out)
    if before != after:
        log.info(f"  Dropped {before-after} rows missing symbol/entry")

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save
    try:
        out.to_parquet(output_path, index=False, engine='pyarrow')
    except ImportError:
        # Fallback: dùng fastparquet nếu pyarrow không có
        try:
            out.to_parquet(output_path, index=False, engine='fastparquet')
        except ImportError:
            log.error("Cần install pyarrow hoặc fastparquet: pip install pyarrow")
            return None

    # Log summary
    action_counts = out['action'].value_counts().to_dict()
    log.info(f"✓ Recommendations parquet → {output_path}")
    log.info(f"  Total: {len(out)} symbols | "
             f"BUY: {action_counts.get('BUY', 0)} | "
             f"HOLD: {action_counts.get('HOLD', 0)} | "
             f"AVOID: {action_counts.get('AVOID', 0)}")

    return output_path


# ============================================================================
# GOOGLE SHEETS EXPORT (cho Quant Pipeline) — song song với Excel
# ============================================================================

def _symbol_to_sector_map(sectors: dict) -> dict:
    """Đảo {ngành: [mã,...]} -> {mã: ngành}. Mã trùng nhiều ngành -> lấy ngành đầu tiên gặp."""
    m = {}
    for sector_name, syms in sectors.items():
        for s in syms:
            m.setdefault(s, sector_name)
    return m


def merge_sector_flow_into_summary(summary_df: pd.DataFrame, flow_df: pd.DataFrame,
                                    sectors: dict = None) -> pd.DataFrame:
    """Gắn thêm 3 cột dòng tiền NGÀNH (của mã đó) vào summary_df theo từng dòng mã:
    'Nganh', 'Nganh_PctChange', 'Nganh_GTGD_ty'  — để tô heatmap ngay trong
    tab Summary hiện có, không cần tách sheet riêng.
    """
    if sectors is None:
        sectors = SectorEngine.WATCHLIST_SECTORS
    sym2sector = _symbol_to_sector_map(sectors)

    out = summary_df.copy()
    out['Nganh'] = out['Symbol'].astype(str).str.upper().str.strip().map(sym2sector).fillna('')

    if flow_df is None or flow_df.empty:
        out['Nganh_PctChange'] = np.nan
        out['Nganh_GTGD_ty'] = np.nan
        return out

    lookup = flow_df.set_index('Nganh')[['PctChange', 'GTGD_ty']]
    out['Nganh_PctChange'] = (out['Nganh'].map(lookup['PctChange']) * 100).round(2)
    out['Nganh_GTGD_ty'] = out['Nganh'].map(lookup['GTGD_ty']).round(1)
    return out


def _push_sector_flow_to_gsheet(spreadsheet, flow_df: pd.DataFrame, worksheet_name: str = 'Analysis'):
    """Ghi 3 cột (Ngành, % Thay đổi, Giá trị giao dịch) vào 1 worksheet cố định
    tên `worksheet_name` (không timestamp — ghi đè mỗi lần chạy), kèm heatmap
    (color scale) NATIVE của Google Sheets cho 2 cột số, qua conditional
    format rule (gradientRule), không phải màu tô cứng — sẽ tự cập nhật
    nếu sau này bạn sửa số liệu tay trên sheet.
    """
    import gspread

    cols = ['Nganh', 'PctChange', 'GTGD_ty']
    df = flow_df[cols].copy()
    df.columns = ['Ngành', '% Thay đổi', 'Giá trị giao dịch (tỷ)']
    df['% Thay đổi'] = (df['% Thay đổi'] * 100).round(2)
    df['Giá trị giao dịch (tỷ)'] = df['Giá trị giao dịch (tỷ)'].round(3)

    try:
        ws = spreadsheet.worksheet(worksheet_name)
        ws.clear()
    except gspread.exceptions.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title=worksheet_name, rows=len(df) + 5, cols=5)

    values = [df.columns.tolist()] + df.values.tolist()
    ws.update(range_name='A1', values=values)

    n_rows = len(df)
    header_fmt = {
        'textFormat': {'bold': True, 'fontSize': 11, 'foregroundColor': {'red': 1, 'green': 1, 'blue': 1}},
        'backgroundColor': {'red': 0.12, 'green': 0.22, 'blue': 0.39},
        'horizontalAlignment': 'CENTER',
    }
    ws.format('1:1', header_fmt)
    ws.freeze(rows=1)
    ws.columns_auto_resize(0, 3)

    sheet_id = ws.id
    last_row = n_rows + 1  # data rows are 2..last_row (1-indexed), 0-indexed exclusive end = last_row

    def _grad_rule(col_index_0based, colors):
        """colors: (min_hex, mid_hex, max_hex) dict rgb 0-1."""
        return {
            'addConditionalFormatRule': {
                'rule': {
                    'ranges': [{
                        'sheetId': sheet_id,
                        'startRowIndex': 1, 'endRowIndex': last_row,
                        'startColumnIndex': col_index_0based, 'endColumnIndex': col_index_0based + 1,
                    }],
                    'gradientRule': {
                        'minpoint': {'type': 'MIN', 'color': colors[0]},
                        'midpoint': {'type': 'PERCENTILE', 'value': '50', 'color': colors[1]},
                        'maxpoint': {'type': 'MAX', 'color': colors[2]},
                    }
                },
                'index': 0,
            }
        }

    RED = {'red': 0.96, 'green': 0.42, 'blue': 0.42}
    WHITE = {'red': 1.0, 'green': 1.0, 'blue': 1.0}
    GREEN = {'red': 0.30, 'green': 0.69, 'blue': 0.62}
    BLUE_LIGHT = {'red': 0.90, 'green': 0.95, 'blue': 1.0}
    BLUE_DARK = {'red': 0.11, 'green': 0.44, 'blue': 0.85}

    requests = [
        _grad_rule(1, (RED, WHITE, GREEN)),        # % Thay đổi: đỏ -> trắng -> xanh
        _grad_rule(2, (BLUE_LIGHT, BLUE_LIGHT, BLUE_DARK)),  # GTGD: nhạt -> đậm
    ]
    spreadsheet.batch_update({'requests': requests})
    log.info(f"Pushed sector cashflow ({n_rows} ngành) + heatmap → worksheet '{worksheet_name}'")


def export_sector_cashflow_to_gsheet(flow_df: pd.DataFrame,
                                      spreadsheet_name: str = 'LỌC CỔ PHIẾU',
                                      credentials_file: str = 'credentials.json',
                                      spreadsheet_id: str = None,
                                      worksheet_name: str = 'Analysis'):
    """Đẩy bảng dòng tiền ngành (Ngành / % Thay đổi / GTGD + heatmap) thẳng
    lên Google Sheet — sheet đích phải tồn tại & đã share Editor cho service
    account (giống _export_to_gsheet() ở trên).
    """
    if flow_df is None or flow_df.empty:
        log.info("Sector cashflow rỗng — skip Google Sheets export.")
        return None
    try:
        client = _gsheet_client(credentials_file)
        spreadsheet = client.open_by_key(spreadsheet_id) if spreadsheet_id else client.open(spreadsheet_name)
    except Exception as e:
        log.warning(f"Sector cashflow → Google Sheets thất bại (auth/open): {e}")
        return None
    try:
        _push_sector_flow_to_gsheet(spreadsheet, flow_df, worksheet_name)
        log.info(f"📊 Sector cashflow Google Sheet: {spreadsheet.url}")
        return spreadsheet.url
    except Exception as e:
        log.warning(f"Push sector cashflow to gsheet failed: {e}")
        return None


def _gsheet_client(credentials_file: str = 'credentials.json'):
    """Xác thực và trả về gspread client. Raise nếu thiếu file/thư viện."""
    import gspread
    return gspread.service_account(filename=credentials_file)


def _apply_heatmap(spreadsheet, ws, df_columns: list, heatmap_cols: list, n_rows: int):
    """Áp gradientRule (heatmap 3 màu đỏ-trắng-xanh) native Google Sheets
    cho các cột trong `heatmap_cols` (tên cột, phải có trong df_columns).
    """
    if not heatmap_cols or n_rows == 0:
        return
    sheet_id = ws.id
    RED = {'red': 0.96, 'green': 0.42, 'blue': 0.42}
    WHITE = {'red': 1.0, 'green': 1.0, 'blue': 1.0}
    GREEN = {'red': 0.30, 'green': 0.69, 'blue': 0.62}
    requests = []
    for col_name in heatmap_cols:
        if col_name not in df_columns:
            continue
        col_idx = df_columns.index(col_name)  # 0-based
        requests.append({
            'addConditionalFormatRule': {
                'rule': {
                    'ranges': [{
                        'sheetId': sheet_id,
                        'startRowIndex': 1, 'endRowIndex': n_rows + 1,
                        'startColumnIndex': col_idx, 'endColumnIndex': col_idx + 1,
                    }],
                    'gradientRule': {
                        'minpoint': {'type': 'MIN', 'color': RED},
                        'midpoint': {'type': 'PERCENTILE', 'value': '50', 'color': WHITE},
                        'maxpoint': {'type': 'MAX', 'color': GREEN},
                    }
                },
                'index': 0,
            }
        })
    if requests:
        spreadsheet.batch_update({'requests': requests})


def _clean_df_for_gsheet(df: pd.DataFrame) -> pd.DataFrame:
    """Ép DataFrame về dạng an toàn cho gspread (str/số, không NaN/NaT)."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype.kind == 'f':
            out[col] = out[col].round(4)
    out = out.astype(object).where(pd.notnull(out), '')
    for col in out.columns:
        out[col] = out[col].apply(lambda v: v.strftime('%Y-%m-%d %H:%M') if hasattr(v, 'strftime') else v)
    return out


USER_HEADER_NOTES = {
    'Đạt bộ lọc': 'Cho biết mã có vượt qua toàn bộ điều kiện mua và kiểm soát rủi ro hay không.',
    'Giải thích điều kiện': 'Các nguyên nhân khiến mã chưa đạt hoặc cần thận trọng.',
    'Điểm': 'Điểm tổng hợp từ 0 đến 100. Điểm càng cao thì tín hiệu tổng thể càng tốt.',
    'Đánh giá': 'Mức đánh giá tổng quát được suy ra từ điểm tổng hợp.',
    'Tín hiệu bộ lọc': 'Tín hiệu ban đầu từ file do crawl_data.py tạo.',
    'Xu hướng ngành': 'Xu hướng giá chung của nhóm ngành mà cổ phiếu thuộc về.',
    'Lợi nhuận ngành 20 phiên (%)': 'Mức tăng hoặc giảm của nhóm ngành trong 20 phiên gần nhất.',
    'Tương quan ngành': 'Mức độ cổ phiếu biến động cùng chiều với các mã trong ngành.',
    'Mã dẫn dắt ngành': 'Cổ phiếu có ảnh hưởng hoặc chuyển động nổi bật trong cùng ngành.',
    'Hiệu quả/rủi ro (Sharpe)': 'Lợi nhuận vượt trội trên mỗi đơn vị rủi ro. Cao hơn thường tốt hơn.',
    'Tỷ lệ phiên tăng (%)': 'Tỷ lệ số phiên có lợi nhuận dương trong dữ liệu lịch sử.',
    'Sụt giảm lớn nhất (%)': 'Mức giảm sâu nhất từ đỉnh xuống đáy trong giai đoạn phân tích.',
    'Lợi nhuận năm hóa (%)': 'Lợi nhuận lịch sử quy đổi theo một năm.',
    'Trạng thái giá (HMM)': 'Trạng thái tăng, giảm hoặc đi ngang do mô hình HMM nhận diện.',
    'Trạng thái biến động': 'Cho biết biến động đang bình thường, mở rộng hay thu hẹp.',
    'Chất lượng dữ liệu': 'Dữ liệu giá và khối lượng có đầy đủ, mới và hợp lệ hay không.',
    'Điểm dữ liệu': 'Mức tin cậy của dữ liệu, từ 0 đến 100.',
    'Cảnh báo dữ liệu': 'Các vấn đề cụ thể được phát hiện trong dữ liệu đầu vào.',
    'Thanh khoản': 'Khả năng mua bán: A tốt nhất, D thấp nhất.',
    'GTGD TB 20 phiên (tỷ)': 'Giá trị giao dịch trung bình 20 phiên, đơn vị tỷ đồng.',
    'Sức chứa tối đa (cổ phiếu)': 'Số cổ phiếu tối đa ước tính có thể mua mà không ảnh hưởng lớn đến thị trường.',
    'Khoảng trống giá 95% (%)': 'Mức nhảy giá bất lợi mà 95% quan sát lịch sử không vượt quá.',
    'Dự báo': 'Hướng giá mà các mô hình dự kiến trong kỳ phân tích.',
    'Lợi nhuận dự báo (%)': 'Mức lợi nhuận do tổ hợp mô hình dự báo.',
    'Lợi nhuận gộp (%)': 'Lợi nhuận dự kiến trước khi trừ chi phí.',
    'Chi phí (%)': 'Phí, thuế và trượt giá ước tính cho lượt mua và bán.',
    'Lợi nhuận ròng (%)': 'Lợi nhuận dự kiến sau khi trừ mọi chi phí.',
    'Đồng thuận (%)': 'Tỷ lệ mô hình cho cùng một hướng dự báo.',
    'Biến động mô phỏng': 'Phương pháp hoặc mức biến động dùng trong mô phỏng Monte Carlo.',
    'Sụt giảm khi chờ T+2 (%)': 'Mức giảm giá lịch sử trong thời gian cổ phiếu chưa thể bán.',
    'Xác suất lỗ trên 3% (%)': 'Khả năng lỗ quá 3% trong thời gian chờ thanh toán T+2.',
    'Đồng thuận mô hình (%)': 'Tỷ lệ các mô hình độc lập cùng cho một hướng.',
    'Xác suất thắng hiệu chỉnh (%)': 'Xác suất giao dịch có lãi sau khi hiệu chỉnh bằng dữ liệu lịch sử.',
    'Giá trị kỳ vọng/rủi ro': 'Lợi nhuận kỳ vọng của giao dịch so với số tiền chấp nhận rủi ro.',
    'Lời/rủi ro': 'Số đồng có thể lời trên mỗi đồng chấp nhận lỗ.',
    'Phương pháp chấm điểm': 'Cách hệ thống tổng hợp các chỉ số thành điểm cuối cùng.',
    'Nhận xét chi tiết': 'Ghi chú screening ngắn, tối đa 3 câu; phân tích quant đầy đủ nằm trong BUY_ONLY của Excel.',
}


def _simplify_user_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa bảng người dùng, giữ tên chỉ số quant bằng tiếng Anh."""
    if summary_df is None or summary_df.empty:
        return pd.DataFrame()

    hidden_columns = [
        'Exchange', 'Timing', 'Screener', 'PWinCalibrated%', 'EV_R',
        'RR_NetCost', 'Shares', 'PositionValue', 'MaxLossVND', 'Method',
    ]
    out = summary_df.drop(columns=hidden_columns, errors='ignore').copy()

    action_map = {'BUY_NOW': 'MUA NGAY', 'BUY_SETUP': 'CHỜ ĐIỂM MUA',
                  'HOLD': 'NẮM GIỮ', 'WATCH': 'THEO DÕI', 'AVOID': 'TRÁNH'}
    dq_map = {'PASS': 'TỐT', 'WARN': 'CẦN LƯU Ý', 'FAIL': 'KHÔNG ĐẠT'}
    gate_map = {
        'DATA_QUALITY_FAIL': 'Dữ liệu chưa đủ tin cậy',
        'LIQUIDITY_FAIL': 'Thanh khoản thấp, khó mua bán',
        'NET_FORECAST_TOO_LOW': 'Lợi nhuận dự kiến không đủ bù chi phí',
        'FORECAST_NOT_UP': 'Giá chưa được dự báo tăng',
        'TIMING_BLOCKED': 'Chưa phải thời điểm phù hợp để mua',
        'SETTLEMENT_LOCK_RISK': 'Có thể giảm giá trong thời gian chờ T+2',
        'VNI_CRISIS': 'Thị trường chung đang có rủi ro cao',
        'POSITION_TOO_SMALL': 'Quy mô mua không phù hợp',
        'RISK_REWARD_TOO_LOW': 'Mức lời chưa tương xứng rủi ro',
    }

    if 'Action' in out:
        out['Action'] = out['Action'].map(action_map).fillna(out['Action'])
    if 'DQStatus' in out:
        out['DQStatus'] = out['DQStatus'].map(dq_map).fillna(out['DQStatus'])
    if 'GatePass' in out:
        out['GatePass'] = out['GatePass'].map({True: 'ĐẠT', False: 'KHÔNG ĐẠT'}).fillna(out['GatePass'])
    if 'Forecast' in out:
        out['Forecast'] = (out['Forecast'].astype(str)
                           .str.replace('📈 ', '', regex=False)
                           .str.replace('📉 ', '', regex=False)
                           .str.replace('TANG', 'TĂNG', regex=False)
                           .str.replace('GIAM', 'GIẢM', regex=False)
                           .str.replace('TRUNG LAP', 'TRUNG LẬP', regex=False))
    if 'GateReasons' in out:
        def explain_gate(value):
            reasons = [x.strip() for x in str(value or '').split(',') if x.strip()]
            return '; '.join(gate_map.get(x, x.replace('_', ' ').capitalize()) for x in reasons) or 'Đạt điều kiện'
        out['GateReasons'] = out['GateReasons'].apply(explain_gate)
    if 'HoldPlan' in out:
        out['HoldPlan'] = (out['HoldPlan'].astype(str)
                           .str.replace('phien', 'phiên', regex=False)
                           .str.replace('ngay', 'ngày', regex=False))
    translations = {
        'Rating': [('SWING BUY', 'MUA LƯỚT SÓNG'), ('BUY & HOLD', 'MUA VÀ NẮM GIỮ')],
        'VNI': [('NEUTRAL', 'TRUNG TÍNH'), ('CRISIS', 'RỦI RO CAO'), ('BEAR', 'GIẢM'), ('BULL', 'TĂNG')],
        'VolReg': [('EXPANSION', 'MỞ RỘNG'), ('CONTRACTION', 'THU HẸP'), ('NORMAL', 'BÌNH THƯỜNG')],
        'Analysis': [('SWING BUY', 'MUA LƯỚT SÓNG'), ('BUY & HOLD', 'MUA VÀ NẮM GIỮ'),
                     ('Forecast', 'Dự báo'), ('Agreement', 'Đồng thuận'),
                     ('Entry', 'Giá mua'), ('Size', 'Số lượng'),
                     ('Score', 'Điểm'), ('MaxDD', 'Sụt giảm lớn nhất'),
                     ('AnnRet', 'Lợi nhuận năm hóa')],
    }
    for col, replacements in translations.items():
        if col not in out:
            continue
        out[col] = out[col].astype(str)
        for old, new in replacements:
            out[col] = out[col].str.replace(old, new, regex=False)

    for col in ('Entry', 'SL', 'TP1', 'TP'):
        if col in out:
            out[col] = (pd.to_numeric(out[col], errors='coerce') / 1000).round(2)
    for col in ('PositionValue', 'MaxLossVND'):
        if col in out:
            out[col] = (pd.to_numeric(out[col], errors='coerce') / 1_000_000).round(2)

    return out.rename(columns={
        'Symbol': 'Mã', 'Action': 'Khuyến nghị', 'GatePass': 'Đạt bộ lọc',
        'Score': 'Điểm', 'Rating': 'Đánh giá',
        'GateReasons': 'Giải thích điều kiện', 'DQStatus': 'Chất lượng dữ liệu',
        'HoldPlan': 'Thời gian nắm giữ', 'VNI': 'Xu hướng VN-Index',
        'Screener': 'Tín hiệu bộ lọc', 'NhomNganh': 'Nhóm ngành',
        'XuHuongNganh': 'Xu hướng ngành', 'NganhRet20D%': 'Lợi nhuận ngành 20 phiên (%)',
        'TuongQuanNganh': 'Tương quan ngành', 'MaDanDatNganh': 'Mã dẫn dắt ngành',
        'Sharpe': 'Sharpe', 'WinRate': 'Win Rate (%)',
        'MaxDD': 'Max DD (%)', 'AnnRet': 'Annual Return (%)',
        'HMM': 'HMM', 'VolReg': 'Volatility Regime',
        'DQScore': 'Điểm dữ liệu', 'LiqTier': 'Thanh khoản',
        'ADV20_Bn': 'GTGD TB 20 phiên (tỷ)', 'Forecast': 'Dự báo',
        'EnsRet%': 'Lợi nhuận dự báo (%)',
        'GrossFc%': 'Lợi nhuận gộp (%)', 'CostRT%': 'Chi phí (%)',
        'NetFc%': 'Lợi nhuận ròng (%)', 'Agreement': 'Đồng thuận (%)',
        'MCVol': 'Biến động mô phỏng', 'LockDD%': 'Sụt giảm khi chờ T+2 (%)',
        'PLoss3%': 'Xác suất lỗ trên 3% (%)', 'DQFlags': 'Cảnh báo dữ liệu',
        'CapacityShares': 'Sức chứa tối đa (cổ phiếu)', 'GapP95%': 'Khoảng trống giá 95% (%)',
        'Entry': 'Entry', 'SL': 'SL', 'TP1': 'TP1',
        'TP': 'TP2', 'ModelAgreement%': 'Model Agreement (%)',
        'PWinCalibrated%': 'Xác suất thắng hiệu chỉnh (%)',
        'EV_R': 'Giá trị kỳ vọng/rủi ro', 'RR_NetCost': 'Lời/rủi ro',
        'Shares': 'Số cổ phiếu',
        'PositionValue': 'Giá trị mua (triệu)', 'MaxLossVND': 'Lỗ tối đa (triệu)',
        'Method': 'Phương pháp chấm điểm', 'Analysis': 'Nhận xét chi tiết',
        'Nganh_PctChange': 'Thay đổi ngành (%)',
        'Nganh_GTGD_ty': 'GTGD ngành (tỷ)',
    })


def _push_two_block_to_worksheet(spreadsheet, df: pd.DataFrame, df_thematic: pd.DataFrame,
                                  worksheet_name: str, heatmap_cols: list = None,
                                  thematic_start_col: str = 'K'):
    """Ghi 2 khối vào CÙNG 1 worksheet, cạnh nhau:
    - Khối 1 (df): bắt đầu cột A — Ngành ICB thuần túy, dùng tính Tỷ trọng %.
    - Khối 2 (df_thematic, optional): bắt đầu cột `thematic_start_col`
      (mặc định K) — Nhóm chuyên đề cắt ngang, trùng mã, chỉ để theo dõi.
    """
    n_rows = max(len(df), len(df_thematic) if df_thematic is not None else 0) + 5
    n_cols = 13 if df_thematic is not None else len(df.columns) + 2
    try:
        ws = spreadsheet.add_worksheet(title=worksheet_name, rows=n_rows, cols=n_cols)
    except Exception:
        ws = spreadsheet.worksheet(worksheet_name)
        ws.clear()

    # ── Khối 1: cột A ──
    df_export = _clean_df_for_gsheet(df)
    all_values = [df_export.columns.tolist()] + df_export.values.tolist()
    ws.update(range_name='A1', values=all_values)
    ws.format('1:1', {
        'textFormat': {'bold': True, 'fontSize': 11, 'foregroundColor': {'red': 1, 'green': 1, 'blue': 1}},
        'backgroundColor': {'red': 0.12, 'green': 0.22, 'blue': 0.39},
        'horizontalAlignment': 'CENTER'
    })

    # ── Khối 2: cột K (nếu có) ──
    if df_thematic is not None and not df_thematic.empty:
        t_export = _clean_df_for_gsheet(df_thematic)
        title_row = [['NHÓM CHUYÊN ĐỀ (trùng mã — chỉ theo dõi, KHÔNG cộng vào tổng GTGD thị trường)']]
        ws.update(range_name=f'{thematic_start_col}1', values=title_row)
        header_range = f'{thematic_start_col}2'
        t_values = [t_export.columns.tolist()] + t_export.values.tolist()
        ws.update(range_name=header_range, values=t_values)
        ws.format(f'{thematic_start_col}1', {
            'textFormat': {'bold': True, 'italic': True, 'fontSize': 10, 'foregroundColor': {'red': 0.4, 'green': 0.4, 'blue': 0.4}}
        })
        ws.format(f'{thematic_start_col}2:{chr(ord(thematic_start_col) + len(t_export.columns) - 1)}2', {
            'textFormat': {'bold': True, 'fontSize': 11, 'foregroundColor': {'red': 1, 'green': 1, 'blue': 1}},
            'backgroundColor': {'red': 0.35, 'green': 0.35, 'blue': 0.35},
            'horizontalAlignment': 'CENTER'
        })

    ws.freeze(rows=1)
    _apply_heatmap(spreadsheet, ws, df_export.columns.tolist(), heatmap_cols, len(df_export))


def _push_df_to_worksheet(spreadsheet, df: pd.DataFrame, worksheet_name: str, heatmap_cols: list = None):
    """Ghi 1 DataFrame vào 1 worksheet (tạo mới hoặc clear nếu đã tồn tại).
    `heatmap_cols`: tên cột sẽ được tô heatmap (color scale) ngay trong sheet này.
    """
    try:
        ws = spreadsheet.add_worksheet(
            title=worksheet_name, rows=len(df) + 5, cols=len(df.columns) + 2
        )
    except Exception:
        ws = spreadsheet.worksheet(worksheet_name)
        ws.clear()

    df_export = _clean_df_for_gsheet(df)

    all_values = [df_export.columns.tolist()] + df_export.values.tolist()
    ws.update(range_name='A1', values=all_values)
    ws.format('1:1', {
        'textFormat': {'bold': True, 'fontSize': 11, 'foregroundColor': {'red': 1, 'green': 1, 'blue': 1}},
        'backgroundColor': {'red': 0.12, 'green': 0.22, 'blue': 0.39},
        'horizontalAlignment': 'CENTER'
    })
    notes = {}
    for col_idx, col_name in enumerate(df_export.columns, 1):
        if col_name in USER_HEADER_NOTES:
            from openpyxl.utils import get_column_letter
            notes[f'{get_column_letter(col_idx)}1'] = USER_HEADER_NOTES[col_name]
    if notes:
        ws.update_notes(notes)
    ws.freeze(rows=1)
    _apply_heatmap(spreadsheet, ws, df_export.columns.tolist(), heatmap_cols, len(df_export))


def _export_to_gsheet(
    summary_df: pd.DataFrame,
    spreadsheet_name: str = 'LỌC CỔ PHIẾU',
    credentials_file: str = 'credentials.json',
    tag_with_timestamp: bool = True,
    spreadsheet_id: str = None,
    flow_df: pd.DataFrame = None,
    sectors: dict = None,
):
    """
    Đẩy summary (toàn bộ) + buy-only (Score>=50 & Forecast TANG) lên Google Sheets.
    Không raise ra ngoài pipeline — lỗi chỉ log.warning.

    Nếu truyền `flow_df` (bảng dòng tiền ngành từ compute_sector_cashflow()),
    hàm sẽ tự gắn thêm 3 cột 'Nganh' / 'Nganh_PctChange' / 'Nganh_GTGD_ty'
    vào NGAY tab Summary (không tách sheet riêng) và tô heatmap màu
    đỏ-trắng-xanh cho 2 cột số đó, để nhìn trực quan dòng tiền đang chảy
    vào ngành nào ngay trên từng dòng mã cổ phiếu.

    Lưu ý: Service Account KHÔNG có quota Drive để tự tạo file mới, nên
    sheet đích PHẢI đã tồn tại và đã được share (Editor) cho email service account.
    Ưu tiên dùng spreadsheet_id (từ URL sheet) để tránh lỗi lệch tên/dấu.
    """
    if summary_df is None or summary_df.empty:
        log.info("Summary rỗng — skip Google Sheets export.")
        return None

    try:
        client = _gsheet_client(credentials_file)
    except FileNotFoundError:
        log.error(
            f"Không tìm thấy file credentials: {credentials_file}\n"
            "Vào Google Cloud Console → tạo Service Account → tải JSON key → "
            f"share Google Sheet '{spreadsheet_name}' cho email service account."
        )
        return None
    except Exception as e:
        log.warning(f"Google Sheets auth failed: {e}")
        return None

    try:
        if spreadsheet_id:
            spreadsheet = client.open_by_key(spreadsheet_id)
        else:
            spreadsheet = client.open(spreadsheet_name)
    except Exception as e:
        log.error(
            f"Không mở được Google Sheet '{spreadsheet_name}'"
            f"{f' (id={spreadsheet_id})' if spreadsheet_id else ''}: {e}\n"
            "Kiểm tra lại:\n"
            "  1) Sheet đã tồn tại và tên/id chính xác\n"
            "  2) Đã Share sheet (quyền Editor) cho email service account "
            "trong credentials.json (field 'client_email')\n"
            "  (Service Account KHÔNG thể tự tạo sheet mới do quota Drive = 0)"
        )
        return None

    suffix = datetime.now().strftime('_%Y%m%d_%H%M') if tag_with_timestamp else ''

    # Gắn dòng tiền ngành vào summary trước khi push (nếu có flow_df)
    push_df = summary_df
    heatmap_cols = None
    if flow_df is not None and not flow_df.empty:
        push_df = merge_sector_flow_into_summary(summary_df, flow_df, sectors)
        heatmap_cols = ['Thay đổi ngành (%)', 'GTGD ngành (tỷ)']

    push_df = _simplify_user_summary(push_df)

    try:
        _push_df_to_worksheet(spreadsheet, push_df, f'Summary{suffix}', heatmap_cols=heatmap_cols)
        log.info(f"Pushed Summary ({len(push_df)} rows) → Google Sheets"
                 + (" (+ heatmap dòng tiền ngành)" if heatmap_cols else ""))
    except Exception as e:
        log.warning(f"Push Summary to gsheet failed: {e}")

    try:
        buy_mask = summary_df.get('Action', pd.Series(['AVOID']*len(summary_df))).isin(['BUY_NOW','BUY_SETUP'])
        buy_df = push_df[buy_mask.to_numpy()].copy().sort_values('Điểm', ascending=False).reset_index(drop=True)
        # Summary remains compact; the Buy_Only worksheet receives the full
        # thesis captured by summary_table() for the same final Action Gate.
        buy_analysis = summary_df.attrs.get('buy_analysis', {})
        if buy_analysis and 'Mã' in buy_df and 'Nhận xét chi tiết' in buy_df:
            buy_df['Nhận xét chi tiết'] = buy_df['Mã'].map(buy_analysis).fillna(buy_df['Nhận xét chi tiết'])
        if not buy_df.empty:
            _push_df_to_worksheet(spreadsheet, buy_df, f'Buy_Only{suffix}', heatmap_cols=heatmap_cols)
            log.info(f"Pushed Buy_Only ({len(buy_df)} rows) → Google Sheets")
        else:
            log.info("Không có mã MUA — skip Buy_Only worksheet.")
    except Exception as e:
        log.warning(f"Push Buy_Only to gsheet failed: {e}")

    log.info(f"📊 Google Sheet: {spreadsheet.url}")
    return spreadsheet.url



def _export_dashboard_architecture(summary_df, reports, output_path='quant_dashboard_architecture.xlsx'):
    """Workbook giám sát kiến trúc; không chứa backtest."""
    if not output_path:
        return None
    from openpyxl import Workbook
    from openpyxl.comments import Comment
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter
    wb = Workbook(); wb.remove(wb.active)
    def add_df(name, df):
        ws = wb.create_sheet(name[:31])
        if df is None or df.empty:
            ws['A1'] = 'No data'; return
        for j, col in enumerate(df.columns, 1):
            c = ws.cell(1,j,col); c.font = Font(bold=True,color='FFFFFF'); c.fill = PatternFill('solid',fgColor='1F4E78')
            c.alignment = Alignment(horizontal='center')
            if col in USER_HEADER_NOTES:
                c.comment = Comment(USER_HEADER_NOTES[col], 'Bot Quant')
        for i, row in enumerate(df.itertuples(index=False, name=None), 2):
            for j, val in enumerate(row, 1):
                if isinstance(val, (list,dict,tuple)): val = json.dumps(val, ensure_ascii=False)
                ws.cell(i,j,val)
        ws.freeze_panes='A2'; ws.auto_filter.ref=ws.dimensions
        for j,col in enumerate(df.columns,1):
            width = min(45, max(10, len(str(col))+2))
            ws.column_dimensions[get_column_letter(j)].width = width

    add_df('00_Architecture', DashboardArchitecture.as_dataframe())
    if summary_df is not None and not summary_df.empty:
        user_summary = _simplify_user_summary(summary_df).sort_values('Điểm', ascending=False)
        add_df('01_Khuyen_Nghi', user_summary)
        add_df('02_Data_Quality', summary_df[[c for c in ['Symbol','Exchange','DQStatus','DQScore','DQFlags'] if c in summary_df.columns]])
        add_df('03_Liquidity', summary_df[[c for c in ['Symbol','LiqTier','ADV20_Bn','CapacityShares','GapP95%'] if c in summary_df.columns]])
        add_df('04_Risk_Position', summary_df[[c for c in ['Symbol','Entry','SL','TP1','TP','RR_NetCost','Shares','PositionValue','MaxLossVND'] if c in summary_df.columns]])
        add_df('05_Model_Monitor', summary_df[[c for c in ['Symbol','HMM','Forecast','GrossFc%','Agreement','CostRT%','NetFc%','VolReg','Method'] if c in summary_df.columns]])
        add_df('06_Sector', summary_df[[c for c in ['Symbol','NhomNganh','XuHuongNganh','NganhRet20D%','TuongQuanNganh','MaDanDatNganh'] if c in summary_df.columns]])
    cfg_df = pd.DataFrame([{'Parameter':k,'Value':str(v)} for k,v in vars(CFG).items()])
    add_df('07_Config', cfg_df)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path); log.info(f'📊 Architecture dashboard exported: {output_path}')
    return output_path

def run_pipeline(excel_path=None, symbols=None, account_size=500_000_000,
                 lookback_days=252, dashboard_architecture_excel='quant_dashboard_architecture.xlsx',
                 buy_summary_excel=None, source=None,
                 use_google_sheets=True, gsheet_name='LỌC CỔ PHIẾU',
                 gsheet_credentials='credentials.json',
                 gsheet_id='1KMf4vwX7uqkIAvr2KGMXnYRc-_f8jmdkTYIG3TyUpG4',
                 export_sector_flow=True, sector_flow_excel=None,
                 sector_flow_watchlist=None, generate_visuals=None, visual_top_n=None,
                 visual_output_dir=None):
    global CFG
    CFG = QuantConfig(); CFG.ACCOUNT_SIZE = account_size
    # source=None → dùng fallback chain KBS → VCI (khuyến nghị)
    # source='KBS' hoặc 'VCI' → ép dùng 1 source duy nhất
    date_str = datetime.now().strftime('%d-%m-%y')
    if buy_summary_excel is None:
        buy_summary_excel = f'khuyến nghị {date_str}.xlsx'
    bridge = ScreenerBridge(source=source); scr_df = pd.DataFrame()
    if excel_path:
        syms, scr_df = bridge.read_screener_excel(excel_path)
        if not symbols: symbols = syms
    elif not symbols:
        syms, scr_df = bridge.read_screener_excel(); symbols = syms
    if not symbols: log.error("Khong co symbols!"); return [], pd.DataFrame()
    log.info(f"\n{'='*60}\n  QUANT PIPELINE V6 VN — {len(symbols)} stocks\n  Account: {account_size:,.0f} VND\n{'='*60}")
    data = bridge.fetch_ohlcv(symbols, days=lookback_days)
    idx_df = bridge.fetch_index('VNINDEX', days=lookback_days)
    exchange_map = bridge.fetch_exchange_map(list(data.keys()))
    cfg = CFG
    pipe = QuantPipeline(cfg)
    reports = pipe.batch(data, scr_df, idx_df, exchange_map=exchange_map)
    summary = pipe.summary_table(reports)
    do_visuals = CFG.GENERATE_VISUALS if generate_visuals is None else bool(generate_visuals)
    top_n_visuals = CFG.VISUAL_TOP_N if visual_top_n is None else max(0, int(visual_top_n))
    visuals_dir = visual_output_dir or CFG.VISUAL_OUTPUT_DIR
    if do_visuals and reports:
        try:
            from quant_visuals import generate_batch_visuals
            generate_batch_visuals(reports[:top_n_visuals], data, summary,
                                   output_dir=visuals_dir)
        except Exception as e:
            log.warning(f"Quant visuals batch error: {e}")
    if dashboard_architecture_excel:
        try: _export_dashboard_architecture(summary, reports, dashboard_architecture_excel)
        except Exception as e: log.warning(f'Dashboard architecture export error: {e}')
    if not summary.empty:
        console_cols = ['Mã', 'Khuyến nghị', 'Điểm', 'Dự báo', 'Lợi nhuận ròng (%)',
                        'Giá mua', 'Cắt lỗ', 'Chốt lời 2', 'Giải thích điều kiện']
        console_summary = _simplify_user_summary(summary)
        console_cols = [c for c in console_cols if c in console_summary.columns]
        print(f"\n{'='*80}\n  KẾT QUẢ PHÂN TÍCH — {len(summary)} MÃ\n{'='*80}")
        print(console_summary[console_cols].to_string(index=False))
    # ── Export Buy-Only Summary ──
    if buy_summary_excel and not summary.empty:
        try:
            vni_reg = AdaptiveScorer._detect_vni_regime(idx_df)
            _export_buy_summary(summary, buy_summary_excel, vni_regime=vni_reg, reports=reports)
        except Exception as e: log.warning(f"Buy summary export error: {e}")
    # ── Export Recommendations Parquet (cho valuation.py) ──
    if not summary.empty:
        try:
            vni_reg = AdaptiveScorer._detect_vni_regime(idx_df)
            _export_recommendations_parquet(summary,
                                            output_path='cache/recommendations_latest.parquet',
                                            vni_regime=vni_reg)
        except Exception as e: log.warning(f"Recommendations parquet export error: {e}")
    # ── Tính Dòng tiền ngành (từ Watchlist.xlsx / SectorEngine map) ──
    # PURE_SECTORS: không trùng mã -> Tỷ trọng (%) phản ánh đúng tỷ trọng
    # GTGD thị trường thật. THEMATIC_GROUPS: trùng mã cố ý (Vingroup, FPT
    # nhóm, Viettel, Tự doanh, List CMSC...) -> chỉ để theo dõi riêng, KHÔNG
    # cộng vào tổng GTGD thị trường.
    flow_df = pd.DataFrame()
    sector_map = None
    if export_sector_flow:
        try:
            sector_map = (load_watchlist_sectors_from_excel(sector_flow_watchlist)
                          if sector_flow_watchlist else SectorEngine.PURE_SECTORS)
            all_sector_syms = sorted({s for syms in sector_map.values() for s in syms})
            missing = [s for s in all_sector_syms if s not in symbols]
            extra_ohlcv = bridge.fetch_ohlcv(missing, days=lookback_days) if missing else {}
            combined_ohlcv = {**data, **extra_ohlcv}
            flow_df = compute_sector_cashflow(combined_ohlcv, sector_map)
            if not flow_df.empty:
                print(f"\n{'='*70}\n  DÒNG TIỀN NGÀNH (Top by GTGD)\n{'='*70}")
                print(flow_df[['Nganh', 'PctChange', 'GTGD_ty', 'TangPct', 'GiamPct', 'DongTienRong_ty']]
                      .to_string(index=False))
            if sector_flow_excel:
                _export_sector_flow_excel(flow_df, sector_flow_excel)
        except Exception as e:
            log.warning(f"Sector cashflow compute/export error: {e}")
    # ── Export Google Sheets (song song Excel, không thay thế) ──
    # flow_df -> gộp thẳng 3 cột Nganh/%/GTGD + heatmap vào NGAY tab Summary,
    # không tách sheet riêng.
    if use_google_sheets and not summary.empty:
        try:
            _export_to_gsheet(summary, spreadsheet_name=gsheet_name,
                              credentials_file=gsheet_credentials,
                              spreadsheet_id=gsheet_id)
        except Exception as e: log.warning(f"Google Sheets export error: {e}")
    try:
        flogger = ForecastLogger('forecast_log.csv')
        flogger.log_reports(reports); flogger.evaluate(bridge)
    except Exception as e: log.warning(f"Forecast logger: {e}")
    return reports, summary

if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════╗
    ║    QUANT PIPELINE V6 VN.1 — HOSE Swing Trading  ║
    ║    FIX: HMM stable seed, GARCH-MC vol cap,   ║
    ║         VNI crisis breaker, blended scoring,  ║
    ║         forecast cap, regime-aware timing     ║
    ╚══════════════════════════════════════════════╝
    """)
    if '--symbols' in sys.argv:
        idx = sys.argv.index('--symbols')
        s = sys.argv[idx+1].split(',') if idx+1<len(sys.argv) else []
        run_pipeline(symbols=[x.strip().upper() for x in s])
    else:
        run_pipeline()
