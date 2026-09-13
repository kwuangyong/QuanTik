# -*- coding: utf-8 -*-
"""
BACKTEST ĐỐI CHIẾU KHUYẾN NGHỊ BOT vs GIÁ THỊ TRƯỜNG THẬT
===========================================================
BƯỚC 1: Thu thập output bot   (xlsx → Entry, Score, Forecast_dir, EnsRet%, SL, TP, HMM)
BƯỚC 2: Thu thập giá thị trường thực tế từ vnstock API
BƯỚC 3: Đối chiếu 3 tầng
  ├─ Tầng 1: Regime — Bot BEAR/CRISIS vs VNI thực tế
  ├─ Tầng 2: Giá Entry bot vs giá thị trường
  └─ Tầng 3: Forecast_dir (TANG/GIAM) vs diễn biến giá các phiên nắm giữ
BƯỚC 4: Tính các chỉ số thống kê
  ├─ Hit rate (% dự báo đúng chiều)
  ├─ Avg actual return vs EnsRet% forecast
  ├─ Alpha so với VNIndex benchmark
  └─ R:R thực tế (SL chưa bị hit, TP đã đạt chưa)

Cách dùng:
    python backtest_simple.py                        # quét thư mục ./reports
    python backtest_simple.py D:/path/to/reports     # chỉ định thư mục khác

Yêu cầu: pip install vnstock pandas openpyxl matplotlib
"""

import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "DejaVu Sans"

# =============================================================================
# CẤU HÌNH
# =============================================================================
REPORTS_DIR        = Path("./reports")
OUTPUT_DIR         = Path("./backtest_out")
HORIZON_DAYS       = 20       # số phiên tối đa nắm giữ trước khi time-out
ENTRY_WINDOW_DAYS  = 5        # số phiên chờ khớp giá Entry
ENTRY_TOL_PCT      = 0.005    # cho phép giá Low ≤ entry*(1+0.5%) coi như khớp
COST_BPS           = 40       # chi phí round-trip (bps)
VNI_SYMBOL         = "VNINDEX"
OOS_FRACTION       = 0.30     # 30% ngay bao cao cuoi dung lam temporal holdout
TOP_N              = 5        # chi xet 5 ma dau tien trong moi bao cao
MIN_SCORE_EXCLUSIVE = 75      # Score > 75 (khong bao gom Score = 75)
ALLOWED_TIMINGS    = ("vao_ngay", "mua_ngay", "cho_pullback")
BACKTEST_START_DATE = pd.Timestamp("2026-07-25")


# =============================================================================
# BƯỚC 1: ĐỌC FILE KHUYẾN NGHỊ XLSX
# =============================================================================
def _normalize(s: str) -> str:
    """Bỏ dấu + lowercase + chuẩn hóa khoảng trắng/gạch dưới."""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.lower().replace(" ", "_")


def _parse_report_date(path: Path) -> pd.Timestamp:
    """Chấp nhận DD-MM-YY hoặc DD MM YY (phân cách bằng - hoặc khoảng trắng)."""
    m = re.search(r"(\d{2})[-\s](\d{2})[-\s](\d{2})", path.stem)
    if not m:
        raise ValueError(f"Không tìm thấy ngày trong tên file: {path.name}")
    dd, mm, yy = m.groups()
    return pd.Timestamp(year=2000 + int(yy), month=int(mm), day=int(dd))


def _parse_float(s) -> float:
    try:
        return float(re.sub(r"[^\d.\-+]", "", str(s)))
    except (ValueError, TypeError):
        return np.nan


def _find_col(df: pd.DataFrame, candidates: list) -> str | None:
    """Tìm tên cột đầu tiên khớp trong danh sách ứng viên."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def read_one_xlsx(path: Path) -> pd.DataFrame:
    """
    Đọc 1 file khuyến_nghị*.xlsx.
    Thu thập: ticker, entry, sl, tp, score, conf, ens_ret_pct,
              forecast_dir (TANG/GIAM), hmm_state, vni_signal.
    """
    report_date = _parse_report_date(path)

    # --- Tìm sheet "Danh Sách Mua" ---
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True)
    sheet_name = None
    for sn in wb.sheetnames:
        if "danh" in _normalize(sn) and "mua" in _normalize(sn):
            sheet_name = sn
            break
    wb.close()
    if sheet_name is None:
        raise ValueError(f"{path.name}: không tìm thấy sheet 'Danh Sách Mua'")

    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)

    # --- Tìm dòng header ---
    header_row = None
    for i, row in raw.iterrows():
        vals = [str(v).strip() for v in row.values]
        vals_norm = {_normalize(v) for v in vals}
        if "ma" in vals_norm and any(x in vals_norm for x in
                                      ("score", "entry", "sl", "diem", "gia_mua")):
            header_row = i
            break
    if header_row is None:
        raise ValueError(f"{path.name}: không tìm thấy dòng header 'Mã'")

    df = pd.read_excel(path, sheet_name=sheet_name, header=header_row)
    df = df.dropna(how="all").reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]
    normalized_cols = {_normalize(c): c for c in df.columns}
    col_ticker = normalized_cols.get("ma")
    if col_ticker is None:
        raise ValueError(f"{path.name}: khong tim thay cot ma co phieu")
    df = df[df[col_ticker].astype(str).str.match(r"^[A-Z]{2,5}$")].reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()

    col_tp   = _find_col(df, ["TP1(2R)", "TP2R", "TP", "TP1", "Chốt lời 1"])
    col_dir  = _find_col(df, ["Forecast_dir", "Forecast", "Dir", "FDir", "Dự báo"])
    col_hmm  = _find_col(df, ["HMM", "HMMState", "HMM_State"])
    col_vni  = _find_col(df, ["VNI_Signal", "VNI", "Regime"])
    col_ens  = _find_col(df, ["EnsRet%", "EnsRet", "Ens_Ret", "Lợi nhuận ròng (%)"])
    col_conf = _find_col(df, ["Conf%", "Conf", "Confidence", "Đồng thuận (%)"])
    col_timing = _find_col(df, ["Timing", "Thời điểm", "Thoi diem", "Khuyến nghị"])
    col_score = _find_col(df, ["Score", "Điểm"])
    col_entry = _find_col(df, ["Entry", "Giá mua"])
    col_sl = _find_col(df, ["SL", "Cắt lỗ"])

    out = pd.DataFrame({
        "report_date":   report_date,
        "ticker":        df[col_ticker].astype(str).str.strip(),
        "score":         df[col_score].apply(_parse_float) if col_score else np.nan,
        "conf":          df[col_conf].apply(_parse_float) if col_conf else np.nan,
        "entry":         df[col_entry].apply(_parse_float) if col_entry else np.nan,
        "sl":            df[col_sl].apply(_parse_float) if col_sl else np.nan,
        "tp":            df[col_tp].apply(_parse_float) if col_tp else np.nan,
        "ens_ret_pct":   df[col_ens].apply(_parse_float) if col_ens else np.nan,
        "forecast_dir":  df[col_dir].astype(str).str.strip().str.upper() if col_dir else "N/A",
        "hmm_state":     df[col_hmm].astype(str).str.strip().str.upper() if col_hmm else "N/A",
        "vni_signal":    df[col_vni].astype(str).str.strip().str.upper() if col_vni else "N/A",
        "timing":        df[col_timing].astype(str).str.strip() if col_timing else "N/A",
        # Preserve the displayed order before any later sorting/filtering.
        "recommendation_rank": np.arange(1, len(df) + 1),
    })
    return out


def load_recommendations(folder: Path) -> pd.DataFrame:
    """Quét thư mục, đọc tất cả file khuyến_nghị*.xlsx (accent-insensitive)."""
    all_xlsx = sorted(folder.glob("*.xlsx"))
    matched = [p for p in all_xlsx if "khuyen_nghi" in _normalize(p.stem)]
    if not matched:
        print(f"[WARN] Không tìm thấy file 'khuyến_nghị*.xlsx' trong {folder}")
        print(f"       Các file xlsx hiện có: {[p.name for p in all_xlsx]}")
        return pd.DataFrame()

    dfs = []
    for p in matched:
        try:
            d = read_one_xlsx(p)
            if not d.empty:
                d["source_file"] = p.name
                d["source_mtime"] = p.stat().st_mtime
                dfs.append(d)
                print(f"  [OK] {p.name}: {len(d)} khuyến nghị")
        except Exception as e:
            print(f"  [WARN] {p.name} lỗi đọc: {e}")

    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)

    # Fixed evaluation window requested by the user. Reports before this date
    # are excluded before strategy filtering and all summary calculations.
    combined = combined[
        pd.to_datetime(combined["report_date"]) >= BACKTEST_START_DATE
    ].copy()

    # Trading rule used in practice: only the first five displayed ideas,
    # strictly above 75, with an actionable entry timing.
    timing_norm = combined["timing"].map(_normalize)
    strategy_mask = (
        (combined["recommendation_rank"] <= TOP_N)
        & (combined["score"] > MIN_SCORE_EXCLUSIVE)
        & timing_norm.map(lambda s: any(t in s for t in ALLOWED_TIMINGS))
    )
    n_unfiltered = len(combined)
    combined = combined[strategy_mask].copy()
    print(
        f"  [INFO] Bo loc chien luoc: Top {TOP_N}, Score > {MIN_SCORE_EXCLUSIVE}, "
        f"Timing vao/mua ngay hoac cho pullback -> {len(combined)}/{n_unfiltered} tin hieu."
    )

    combined = combined.sort_values(
        ["report_date", "ticker", "source_mtime", "source_file"]
    )
    n_before = len(combined)
    # Morning and afternoon reports are separate decision events. Only remove
    # an accidental duplicate inside the same source report.
    combined = combined.drop_duplicates(["source_file", "ticker"], keep="last")
    n_dropped = n_before - len(combined)
    if n_dropped:
        print(f"  [INFO] Loai {n_dropped} dong trung ma trong cung mot file bao cao.")
    return combined.drop(columns=["source_mtime"]).reset_index(drop=True)


# =============================================================================
# BƯỚC 2: CÀO GIÁ THẬT — vnstock
# =============================================================================
_PRICE_CACHE: dict = {}

def _fetch_vnstock(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Gọi vnstock API, trả về DataFrame index=Date, cột OHLC (đơn vị nghìn VNĐ).
    Thứ tự ưu tiên: KBS → VCI (fallback).
    """
    from vnstock import Vnstock
    last_err = None
    for source in ("KBS", "VCI"):
        try:
            stock = Vnstock().stock(symbol=symbol, source=source)
            df = stock.quote.history(start=start, end=end, interval="1D")
            break
        except Exception as e:
            last_err = e
    else:
        raise RuntimeError(f"Cả KBS lẫn VCI đều thất bại: {last_err}")

    rename = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("time", "date", "tradingdate"):
            rename[c] = "Date"
        elif cl == "open":  rename[c] = "Open"
        elif cl == "high":  rename[c] = "High"
        elif cl == "low":   rename[c] = "Low"
        elif cl == "close": rename[c] = "Close"
    df = df.rename(columns=rename)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df[["Open", "High", "Low", "Close"]].astype(float)
    # Quy đổi VNĐ → nghìn VNĐ nếu cần
    if df["Close"].mean() > 1000:
        df = df / 1000
    return df


def fetch_price(symbol: str) -> pd.DataFrame:
    """Cào giá với cache. Trả về DataFrame rỗng nếu thất bại."""
    if symbol in _PRICE_CACHE:
        return _PRICE_CACHE[symbol]
    try:
        df = _fetch_vnstock(
            symbol,
            start="2024-01-01",
            end=pd.Timestamp.today().strftime("%Y-%m-%d"),
        )
        _PRICE_CACHE[symbol] = df
        return df
    except Exception as e:
        print(f"  [WARN] Cào giá thất bại — {symbol}: {e}")
        _PRICE_CACHE[symbol] = pd.DataFrame()
        return pd.DataFrame()


# =============================================================================
# BƯỚC 3: ĐỐI CHIẾU 3 TẦNG
# =============================================================================

# ---------- Tầng 1: Regime --------------------------------------------------
def _vni_regime_at(vni_df: pd.DataFrame, date: pd.Timestamp,
                   lookback: int = 20) -> dict:
    """
    Tính regime VNIndex tại thời điểm date dựa trên:
    - ret_20d: return 20 phiên trước đó
    - dd_from_peak: drawdown từ đỉnh trong lookback
    Trả về: regime_actual ("BULL"/"BEAR"/"CRISIS"), ret_20d, dd_from_peak
    """
    result = {"regime_actual": "N/A", "vni_ret_20d": np.nan, "vni_dd_peak": np.nan}
    if vni_df.empty:
        return result
    past = vni_df[vni_df.index <= date].tail(lookback + 1)
    if len(past) < 2:
        return result

    ret = (past["Close"].iloc[-1] / past["Close"].iloc[0] - 1) * 100
    peak = past["Close"].max()
    dd = (past["Close"].iloc[-1] / peak - 1) * 100

    result["vni_ret_20d"] = round(ret, 2)
    result["vni_dd_peak"] = round(dd, 2)
    if dd <= -15:
        result["regime_actual"] = "CRISIS"
    elif ret < -5 or dd <= -8:
        result["regime_actual"] = "BEAR"
    else:
        result["regime_actual"] = "BULL"
    return result


def _regime_match(bot_signal: str, actual_regime: str) -> bool | None:
    """
    So sánh regime bot báo vs thực tế.
    Bot báo BEAR/CRISIS → đúng nếu actual là BEAR/CRISIS.
    Bot báo BULL        → đúng nếu actual là BULL.
    """
    if bot_signal in ("N/A", "nan", ""):
        return None
    bear_signals = {"BEAR", "CRISIS", "GIAM", "WARNING"}
    bull_signals = {"BULL", "TANG", "OK"}
    if bot_signal.upper() in bear_signals:
        return actual_regime in ("BEAR", "CRISIS")
    if bot_signal.upper() in bull_signals:
        return actual_regime == "BULL"
    return None


# ---------- Tầng 2: Entry fill ----------------------------------------------
def _check_entry(rec: pd.Series, price_df: pd.DataFrame) -> dict:
    """Kiểm tra xem giá có chạm vùng Entry trong ENTRY_WINDOW_DAYS phiên không."""
    result = {"entered": False, "entry_date": None, "entry_fill": np.nan}
    if price_df.empty or pd.isna(rec.get("entry")):
        return result
    # Strictly use the next trading session. A report dated intraday or after
    # close must not benefit from that session's already-known Open/Low.
    report_day = pd.Timestamp(rec["report_date"]).normalize()
    post = price_df[price_df.index.normalize() > report_day]
    window = post.head(ENTRY_WINDOW_DAYS)
    tol = rec["entry"] * (1 + ENTRY_TOL_PCT)
    touched = window[window["Low"] <= tol]
    if touched.empty:
        return result
    bar = touched.iloc[0]
    result["entered"] = True
    result["entry_date"] = bar.name
    result["entry_fill"] = float(min(bar["Open"], rec["entry"]))
    return result


# ---------- Tầng 3: Forecast_dir + Triple-barrier exit ----------------------
def _evaluate_trade(rec: pd.Series, fill: float, entry_bar_name,
                    price_df: pd.DataFrame) -> dict:
    """
    Từ điểm vào lệnh:
    - Tính ret tại các horizon 3/5/10/20d
    - Xác định kết quả triple-barrier (TP / SL / time_out)
    - So sánh Forecast_dir với actual direction
    """
    result = {
        "result": "pending",
        "exit_date": None, "exit_price": np.nan, "days_to_exit": None,
        "pnl_pct": np.nan, "pnl_net_pct": np.nan,
        "ret_3d": np.nan, "ret_5d": np.nan, "ret_10d": np.nan, "ret_20d": np.nan,
        "forecast_dir_correct": None,
    }

    forward = price_df[price_df.index >= entry_bar_name].head(HORIZON_DAYS + 1)
    if len(forward) < 2:
        return result

    # Calculate observable horizons even when the trade is still pending.
    closes = forward["Close"]
    for h, key in ((3, "ret_3d"), (5, "ret_5d"), (10, "ret_10d"), (20, "ret_20d")):
        if len(closes) > h:
            result[key] = round((closes.iloc[h] / fill - 1) * 100, 3)

    # --- Triple barrier ---
    exit_found = False
    for i, (date, bar) in enumerate(forward.iterrows()):
        if i == 0:
            continue
        sl_val = rec.get("sl", np.nan)
        tp_val = rec.get("tp", np.nan)

        # Long-position gap handling: a barrier crossed at the open fills at
        # the opening price, not at the more favorable stated SL/TP price.
        if (not pd.isna(sl_val)) and bar["Open"] <= sl_val:
            result.update(result="SL", exit_date=date,
                          exit_price=float(bar["Open"]), days_to_exit=i)
            exit_found = True; break
        if (not pd.isna(tp_val)) and bar["Open"] >= tp_val:
            result.update(result="TP", exit_date=date,
                          exit_price=float(bar["Open"]), days_to_exit=i)
            exit_found = True; break

        hit_sl = (not pd.isna(sl_val)) and (bar["Low"]  <= sl_val)
        hit_tp = (not pd.isna(tp_val)) and (bar["High"] >= tp_val)

        if hit_sl and hit_tp:
            # cùng phiên → an toàn: tính SL trước
            result.update(result="SL", exit_date=date,
                          exit_price=float(sl_val), days_to_exit=i)
            exit_found = True; break
        elif hit_sl:
            result.update(result="SL", exit_date=date,
                          exit_price=float(sl_val), days_to_exit=i)
            exit_found = True; break
        elif hit_tp:
            result.update(result="TP", exit_date=date,
                          exit_price=float(tp_val), days_to_exit=i)
            exit_found = True; break

    if not exit_found and len(forward) < HORIZON_DAYS + 1:
        result.update(result="pending", exit_date=None, exit_price=np.nan,
                      days_to_exit=len(forward) - 1)
    elif not exit_found:
        last = forward.iloc[-1]
        result.update(result="time_out", exit_date=forward.index[-1],
                      exit_price=float(last["Close"]),
                      days_to_exit=len(forward) - 1)

    # --- PnL ---
    ep = result["exit_price"]
    if pd.notna(ep) and fill > 0:
        raw_pnl = (ep / fill - 1) * 100
        result["pnl_pct"]     = round(raw_pnl, 3)
        result["pnl_net_pct"] = round(raw_pnl - COST_BPS / 100, 3)

    # --- Fixed horizon returns ---
    # Fixed-horizon returns were calculated above so pending observations are
    # retained without being mislabeled as completed trades.

    # --- Forecast_dir accuracy (Tầng 3) ---
    fdir = _normalize(str(rec.get("forecast_dir", "N/A"))).upper()
    ret20 = result["ret_20d"]
    if fdir not in ("N/A", "NAN", "") and pd.notna(ret20):
        tang_signals = {"TANG", "UP", "BUY", "LONG", "TĂNG"}
        giam_signals = {"GIAM", "DOWN", "SELL", "SHORT", "GIẢM"}
        if any(s in fdir for s in ("TANG", "UP", "BUY", "LONG")):
            result["forecast_dir_correct"] = bool(ret20 > 0)
        elif any(s in fdir for s in ("GIAM", "DOWN", "SELL", "SHORT")):
            result["forecast_dir_correct"] = bool(ret20 < 0)

    return result


# =============================================================================
# Orchestrator đối chiếu toàn bộ
# =============================================================================
def run_backtest(recs: pd.DataFrame) -> pd.DataFrame:
    # Cào VNIndex trước
    print(f"\n[BƯỚC 2] Cào giá {len(recs.ticker.unique())} mã + VNIndex...")
    vni_df = fetch_price(VNI_SYMBOL)
    if vni_df.empty:
        print(f"  [WARN] Không cào được VNIndex — Tầng 1 (Regime) sẽ bị bỏ qua.")

    tickers = sorted(recs.ticker.unique())
    ok, fail = 0, 0
    for t in tickers:
        px = fetch_price(t)
        if px.empty: fail += 1
        else: ok += 1
    print(f"  -> Cào giá thành công: {ok}/{len(tickers)} mã"
          + (f"  |  Thất bại: {fail}" if fail else ""))

    rows = []
    for _, rec in recs.iterrows():
        row = {
            "ticker":        rec["ticker"],
            "report_date":   rec["report_date"],
            "entry":         rec.get("entry", np.nan),
            "sl":            rec.get("sl", np.nan),
            "tp":            rec.get("tp", np.nan),
            "score":         rec.get("score", np.nan),
            "conf":          rec.get("conf", np.nan),
            "ens_ret_pct":   rec.get("ens_ret_pct", np.nan),
            "forecast_dir":  rec.get("forecast_dir", "N/A"),
            "hmm_state":     rec.get("hmm_state", "N/A"),
            "vni_signal":    rec.get("vni_signal", "N/A"),
            "source_file":   rec.get("source_file", "N/A"),
            "timing":        rec.get("timing", "N/A"),
            "recommendation_rank": rec.get("recommendation_rank", np.nan),
            "vni_trade_ret_pct": np.nan,
            "alpha_vs_vni_pct": np.nan,
        }

        # --- Tầng 1: Regime ---
        regime_info = _vni_regime_at(vni_df, rec["report_date"])
        row.update(regime_info)
        row["regime_match"] = _regime_match(rec.get("vni_signal", "N/A"),
                                            regime_info["regime_actual"])

        # --- Tầng 2: Entry fill ---
        px = fetch_price(rec["ticker"])
        entry_info = _check_entry(rec, px)
        row.update(entry_info)

        if not entry_info["entered"]:
            row.update(result="not_entered", exit_date=None, exit_price=np.nan,
                       days_to_exit=None, pnl_pct=np.nan, pnl_net_pct=np.nan,
                       ret_3d=np.nan, ret_5d=np.nan, ret_10d=np.nan, ret_20d=np.nan,
                       forecast_dir_correct=None)
        elif px.empty:
            row.update(result="no_data", exit_date=None, exit_price=np.nan,
                       days_to_exit=None, pnl_pct=np.nan, pnl_net_pct=np.nan,
                       ret_3d=np.nan, ret_5d=np.nan, ret_10d=np.nan, ret_20d=np.nan,
                       forecast_dir_correct=None)
        else:
            # --- Tầng 3: Trade evaluation ---
            trade = _evaluate_trade(rec, entry_info["entry_fill"],
                                    entry_info["entry_date"], px)
            row.update(trade)

            # Compare each completed trade with VNIndex over its own holding
            # dates. This is a trade-level benchmark, not a portfolio return.
            if (trade["result"] in ("TP", "SL", "time_out")
                    and not vni_df.empty and trade["exit_date"] is not None):
                bench = vni_df[(vni_df.index >= entry_info["entry_date"])
                               & (vni_df.index <= trade["exit_date"])]
                if len(bench) >= 2:
                    vni_ret = (bench["Close"].iloc[-1] / bench["Close"].iloc[0] - 1) * 100
                    row["vni_trade_ret_pct"] = round(vni_ret, 3)
                    row["alpha_vs_vni_pct"] = round(trade["pnl_net_pct"] - vni_ret, 3)

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# BƯỚC 4: THỐNG KÊ + BIỂU ĐỒ
# =============================================================================
def _alpha(portfolio_ret: float, vni_ret: float) -> float:
    """Alpha đơn giản = return portfolio - return benchmark."""
    if pd.isna(vni_ret):
        return np.nan
    return round(portfolio_ret - vni_ret, 3)


def _trade_stats(sample: pd.DataFrame) -> dict:
    """Risk/return statistics for independent completed trade observations."""
    pnl = sample["pnl_net_pct"].dropna()
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    downside = pnl[pnl < 0]
    return {
        "n": len(pnl),
        "mean": pnl.mean() if len(pnl) else np.nan,
        "median": pnl.median() if len(pnl) else np.nan,
        "win_rate": (pnl > 0).mean() * 100 if len(pnl) else np.nan,
        "profit_factor": gains / losses if losses > 0 else np.nan,
        "worst": pnl.min() if len(pnl) else np.nan,
        "downside_dev": downside.std(ddof=1) if len(downside) > 1 else np.nan,
    }


def print_summary(df: pd.DataFrame, vni_df: pd.DataFrame):
    sep = "=" * 72
    print(f"\n{sep}")
    print("KẾT QUẢ ĐỐI CHIẾU TỪNG MÃ")
    print(sep)
    show_cols = ["ticker", "report_date", "entry", "sl", "tp",
                 "entered", "entry_fill", "result",
                 "pnl_net_pct", "days_to_exit",
                 "forecast_dir", "forecast_dir_correct",
                 "vni_signal", "regime_actual", "regime_match"]
    disp = df[[c for c in show_cols if c in df.columns]].copy()
    for col in ("pnl_net_pct",):
        if col in disp:
            disp[col] = disp[col].round(2)
    print(disp.to_string(index=False))

    closed  = df[df["result"].isin(["TP", "SL", "time_out"])]
    entered = df[df["entered"] == True]
    no_data = df[df["result"] == "no_data"]
    not_ent = df[df["result"] == "not_entered"]

    print(f"\n{sep}")
    print("TỔNG HỢP HIỆU SUẤT")
    print(sep)
    print(f"Tổng khuyến nghị          : {len(df)}")
    if not no_data.empty:
        print(f"⚠ Không có dữ liệu giá    : {len(no_data)} "
              f"({len(no_data)/len(df)*100:.1f}%)")
    print(f"Không chạm Entry           : {len(not_ent)} ({len(not_ent)/max(len(df),1)*100:.1f}%)")
    print(f"Số lệnh khớp Entry         : {len(entered)} ({len(entered)/max(len(df),1)*100:.1f}%)")

    if not closed.empty:
        n_tp = (closed["result"] == "TP").sum()
        n_sl = (closed["result"] == "SL").sum()
        n_to = (closed["result"] == "time_out").sum()
        avg_pnl = closed["pnl_net_pct"].mean()
        med_pnl = closed["pnl_net_pct"].median()

        print(f"\n── Triple Barrier ──────────────────────────────")
        print(f"  Chạm TP (thắng)         : {n_tp}  ({n_tp/max(len(closed),1)*100:.1f}%)")
        print(f"  Chạm SL (thua)          : {n_sl}  ({n_sl/max(len(closed),1)*100:.1f}%)")
        print(f"  Hết hạn (time-out)      : {n_to}  ({n_to/max(len(closed),1)*100:.1f}%)")
        print(f"  Win rate TP/(TP+SL)     : {n_tp/max(n_tp+n_sl,1)*100:.1f}%")
        print(f"  PnL trung bình (net)    : {avg_pnl:.2f}%")
        print(f"  PnL trung vị   (net)    : {med_pnl:.2f}%")

        stats = _trade_stats(closed)
        print(f"  Tỷ lệ lệnh dương       : {stats['win_rate']:.1f}%")
        print(f"  Profit factor          : {stats['profit_factor']:.2f}")
        print(f"  Lệnh lỗ lớn nhất       : {stats['worst']:.2f}%")
        print(f"  Downside deviation     : {stats['downside_dev']:.2f}%")

        unique_dates = np.array(sorted(pd.to_datetime(closed["report_date"]).unique()))
        if len(unique_dates) >= 4:
            split_at = max(1, int(np.floor(len(unique_dates) * (1 - OOS_FRACTION))))
            split_at = min(split_at, len(unique_dates) - 1)
            cutoff = unique_dates[split_at]
            ins = closed[pd.to_datetime(closed["report_date"]) < cutoff]
            oos = closed[pd.to_datetime(closed["report_date"]) >= cutoff]
            ins_s, oos_s = _trade_stats(ins), _trade_stats(oos)
            print(f"\n── Temporal holdout (30% ngày cuối) ─────────")
            print(f"  Mốc OOS                 : {pd.Timestamp(cutoff).date()}")
            print(f"  In-sample  n/mean/PF     : {ins_s['n']} / {ins_s['mean']:.2f}% / {ins_s['profit_factor']:.2f}")
            print(f"  Out-of-sample n/mean/PF  : {oos_s['n']} / {oos_s['mean']:.2f}% / {oos_s['profit_factor']:.2f}")

        # Forecast_dir hit rate (Tầng 3)
        has_dir = closed["forecast_dir_correct"].notna()
        if has_dir.any():
            hit_rate = closed.loc[has_dir, "forecast_dir_correct"].mean() * 100
            print(f"\n── Tầng 3: Forecast_dir ────────────────────────")
            print(f"  Hit rate dự báo chiều   : {hit_rate:.1f}%  "
                  f"({has_dir.sum()} lệnh có đủ dữ liệu)")

            # EnsRet% forecast vs actual
            ens_valid = closed[closed["ens_ret_pct"].notna() & closed["ret_20d"].notna()]
            if not ens_valid.empty:
                bias = (ens_valid["ens_ret_pct"] - ens_valid["ret_20d"]).mean()
                mae  = (ens_valid["ens_ret_pct"] - ens_valid["ret_20d"]).abs().mean()
                print(f"  EnsRet% avg forecast    : {ens_valid['ens_ret_pct'].mean():.2f}%")
                print(f"  Actual ret_20d avg      : {ens_valid['ret_20d'].mean():.2f}%")
                print(f"  BIAS (forecast−actual)  : {bias:+.2f}%")
                print(f"  MAE                     : {mae:.2f}%")

        # Alpha (Tầng 1 + so sánh benchmark)
        if False and not vni_df.empty and "report_date" in closed.columns:
            dates = pd.to_datetime(closed["report_date"])
            start, end = dates.min(), dates.max() + pd.Timedelta(days=HORIZON_DAYS)
            end = min(end, vni_df.index.max())
            vni_sub = vni_df[(vni_df.index >= start) & (vni_df.index <= end)]
            if len(vni_sub) >= 2:
                vni_ret = (vni_sub["Close"].iloc[-1] / vni_sub["Close"].iloc[0] - 1) * 100
                port_ret = avg_pnl
                alpha_val = _alpha(port_ret, vni_ret)
                print(f"\n── Alpha vs Benchmark ──────────────────────────")
                print(f"  VNIndex return (kỳ BT) : {vni_ret:.2f}%")
                print(f"  Portfolio avg net PnL  : {port_ret:.2f}%")
                print(f"  Alpha                  : {alpha_val:+.2f}%")

        alpha_valid = closed["alpha_vs_vni_pct"].dropna()
        if not alpha_valid.empty:
            print(f"\n── So sánh VNIndex cùng thời gian giữ lệnh ────")
            print(f"  Số lệnh có benchmark : {len(alpha_valid)}")
            print(f"  Alpha trung bình/lệnh   : {alpha_valid.mean():+.2f}%")
            print(f"  Alpha trung vị/lệnh    : {alpha_valid.median():+.2f}%")

        # Regime match (Tầng 1)
        regime_valid = df[df["regime_match"].notna()]
        if not regime_valid.empty:
            r_acc = regime_valid["regime_match"].mean() * 100
            print(f"\n── Tầng 1: Regime ──────────────────────────────")
            print(f"  Bot regime accuracy     : {r_acc:.1f}%  "
                  f"({len(regime_valid)} phiên có dữ liệu VNI)")
            for regime in ("BULL", "BEAR", "CRISIS"):
                sub = regime_valid[regime_valid["regime_actual"] == regime]
                if not sub.empty:
                    print(f"    {regime:<8}: {sub['regime_match'].mean()*100:.0f}% đúng  "
                          f"(n={len(sub)})")


def make_charts(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(exist_ok=True, parents=True)
    closed = df[df["result"].isin(["TP", "SL", "time_out"])].copy()
    if closed.empty:
        print("[WARN] Chưa có lệnh nào đóng — không vẽ biểu đồ.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("BACKTEST — Đối chiếu khuyến nghị bot vs thị trường", fontsize=14, y=1.01)

    # 1. Pie: tỉ lệ TP/SL/time_out
    counts = closed["result"].value_counts()
    labels_vi = {"TP": "Chạm TP (thắng)", "SL": "Chạm SL (thua)", "time_out": "Hết hạn"}
    colors_map = {"TP": "#2ecc71", "SL": "#e74c3c", "time_out": "#95a5a6"}
    axes[0, 0].pie(
        counts.values,
        labels=[labels_vi.get(k, k) for k in counts.index],
        autopct="%1.0f%%",
        colors=[colors_map.get(k, "#bdc3c7") for k in counts.index],
        startangle=90,
    )
    axes[0, 0].set_title("Tỉ lệ kết quả các lệnh")

    # 2. Bar: PnL% từng mã
    sorted_c = closed.sort_values("pnl_net_pct")
    bar_cols = ["#2ecc71" if v >= 0 else "#e74c3c" for v in sorted_c["pnl_net_pct"]]
    axes[0, 1].barh(sorted_c["ticker"], sorted_c["pnl_net_pct"], color=bar_cols)
    axes[0, 1].axvline(0, color="black", linewidth=0.8)
    axes[0, 1].set_xlabel("PnL net (%)")
    axes[0, 1].set_title("Lợi nhuận/lỗ từng mã (đã trừ phí)")

    # 3. Scatter: Conf% vs PnL (kiểm tra calibration)
    if "conf" in closed.columns and closed["conf"].notna().any():
        axes[1, 0].scatter(closed["conf"], closed["pnl_net_pct"],
                           c=bar_cols, s=80, edgecolors="black", linewidths=0.5)
        axes[1, 0].axhline(0, color="black", linewidth=0.8)
        axes[1, 0].set_xlabel("Conf% (bot báo)")
        axes[1, 0].set_ylabel("PnL net (%)")
        axes[1, 0].set_title("Conf% vs Kết quả thực tế")
    else:
        axes[1, 0].axis("off")

    # 4. Scatter: EnsRet% forecast vs actual ret_20d
    ens_valid = closed[closed["ens_ret_pct"].notna() & closed["ret_20d"].notna()]
    if not ens_valid.empty:
        axes[1, 1].scatter(ens_valid["ens_ret_pct"], ens_valid["ret_20d"],
                           c="#3498db", s=80, edgecolors="black", linewidths=0.5, alpha=0.8)
        mn = min(ens_valid["ens_ret_pct"].min(), ens_valid["ret_20d"].min())
        mx = max(ens_valid["ens_ret_pct"].max(), ens_valid["ret_20d"].max())
        axes[1, 1].plot([mn, mx], [mn, mx], "k--", linewidth=0.8, label="perfect forecast")
        axes[1, 1].axhline(0, color="gray", linewidth=0.5)
        axes[1, 1].axvline(0, color="gray", linewidth=0.5)
        axes[1, 1].set_xlabel("EnsRet% (dự báo bot)")
        axes[1, 1].set_ylabel("Actual ret_20d (%)")
        axes[1, 1].set_title("Forecast vs Thực tế (20 phiên)")
        axes[1, 1].legend(fontsize=8)
    else:
        axes[1, 1].axis("off")
        axes[1, 1].text(0.5, 0.5, "Không có dữ liệu EnsRet%",
                        ha="center", va="center", transform=axes[1, 1].transAxes)

    plt.tight_layout()
    out = output_dir / "backtest_charts.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[OK] Đã lưu biểu đồ: {out}")


# =============================================================================
# MAIN
# =============================================================================
def run(reports_dir: Path = REPORTS_DIR, output_dir: Path = OUTPUT_DIR):
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"{'='*72}")
    print(f"BACKTEST — {reports_dir}")
    print(f"{'='*72}")

    # BƯỚC 1
    print(f"\n[BƯỚC 1] Đọc file khuyến nghị...")
    recs = load_recommendations(reports_dir)
    if recs.empty:
        print("[ERROR] Không có khuyến nghị nào. Dừng lại.")
        return
    print(f"  -> Tổng cộng {len(recs)} khuyến nghị "
          f"| {recs['ticker'].nunique()} mã duy nhất "
          f"| {recs['report_date'].nunique()} phiên báo cáo")

    # BƯỚC 2 + 3
    print(f"\n[BƯỚC 3] Đối chiếu 3 tầng...")
    result = run_backtest(recs)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    result_path = output_dir / f"backtest_result_{ts}.csv"
    result.to_csv(result_path, index=False)
    print(f"  -> Đã lưu chi tiết: {result_path}")

    # BƯỚC 4
    vni_df = _PRICE_CACHE.get(VNI_SYMBOL, pd.DataFrame())
    print_summary(result, vni_df)
    make_charts(result, output_dir)


if __name__ == "__main__":
    reports = Path(sys.argv[1]) if len(sys.argv) > 1 else REPORTS_DIR
    if not reports.exists():
        print(f"[ERROR] Không tìm thấy thư mục: {reports}")
        print("Sửa REPORTS_DIR ở đầu file, hoặc chạy:")
        print("  python backtest_simple.py D:/path/to/reports")
        sys.exit(1)
    run(reports)
