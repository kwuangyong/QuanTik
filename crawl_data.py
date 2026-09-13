import sys
import argparse

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        try:
            _stream.reconfigure(
                encoding='utf-8', errors='replace',
                line_buffering=True, write_through=True,
            )
        except (OSError, ValueError):
            pass

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import warnings
import time
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


# PHẦN 1: DATA LAYER - Thu thập & chuẩn bị dữ liệ

class DataProvider:
 
    # Thứ tự thử source: KBS primary, VCI fallback (TCBS đã đóng từ 15/12/2024)
    FALLBACK_SOURCES = ['VCI', 'KBS']

    # Retry config cho mỗi source
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0  # giây, tăng dần theo exponential backoff

    def __init__(self, source: str = 'KBS'):
        from vnstock import Vnstock
        self.vs = Vnstock()
        self.source = source
        logger.info(f"DataProvider initialized | source={source}")

    def _get_sources(self) -> List[str]:
        """Trả về danh sách source theo thứ tự ưu tiên (source chính trước, fallback sau)."""
        sources = [self.source]
        for s in self.FALLBACK_SOURCES:
            if s != self.source:
                sources.append(s)
        return sources

    def _retry_call(self, func, *args, max_retries: int = None, **kwargs):
        """
        Gọi 1 function với retry + exponential backoff.
        Trả về kết quả nếu thành công, raise exception cuối cùng nếu hết retry.
        """
        max_retries = max_retries or self.MAX_RETRIES
        last_exc = None
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exc = e
                if attempt < max_retries - 1:
                    wait = self.RETRY_DELAY * (2 ** attempt)
                    time.sleep(wait)
        raise last_exc

    # Nhóm gộp toàn bộ 3 sàn — dùng khi group='ALL' (hoặc các alias bên dưới)
    ALL_EXCHANGES = ['HOSE', 'HNX', 'UPCOM']
    ALL_GROUP_ALIASES = {'ALL', 'TOANBO', 'TOÀN BỘ', 'HOSE_HNX_UPCOM', 'VNALL_EXCHANGE'}

    def get_stock_list(self, group: str = 'VN30') -> List[str]:
        """
        Lấy danh sách mã cổ phiếu theo nhóm.
        Tự động fallback source nếu source chính lỗi.

        Parameters:
            group: 'VN30', 'HOSE', 'HNX', 'UPCOM', 'VN100', 'VNALL',
                   hoặc 'ALL' (= toàn bộ mã trên cả 3 sàn HOSE + HNX + UPCOM)
        Returns:
            List[str]: Danh sách mã cổ phiếu
        """
        # Gộp toàn bộ 3 sàn: HOSE + HNX + UPCOM
        if group.upper() in self.ALL_GROUP_ALIASES:
            all_symbols = []
            seen = set()
            for exch in self.ALL_EXCHANGES:
                syms = self.get_stock_list(group=exch)
                if not syms:
                    logger.warning(f"get_stock_list: Không lấy được mã nào từ sàn {exch}")
                    continue
                for s in syms:
                    if s not in seen:
                        seen.add(s)
                        all_symbols.append(s)
                logger.info(f"get_stock_list: {exch} → {len(syms)} mã (lũy kế {len(all_symbols)} mã)")
            return all_symbols

        for src in self._get_sources():
            try:
                listing = self.vs.stock(symbol='VNM', source=src).listing
                if group == 'VNALL':
                    df = listing.all_symbols()
                    return df['symbol'].tolist() if 'symbol' in df.columns else df.iloc[:, 0].tolist()
                else:
                    symbols = listing.symbols_by_group(group=group)
                    return symbols.tolist() if hasattr(symbols, 'tolist') else list(symbols)
            except Exception as e:
                logger.warning(f"get_stock_list: {src} failed ({e}), trying next source...")

        logger.error("get_stock_list: All sources failed!")
        return []

    def _normalize_ohlcv(self, df: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
        """Chuẩn hóa tên cột và validate DataFrame OHLCV."""
        if df is None or df.empty:
            return None

        col_map = {}
        for col in df.columns:
            cl = col.lower()
            if 'time' in cl or 'date' in cl:
                col_map[col] = 'time'
            elif cl == 'open':
                col_map[col] = 'open'
            elif cl == 'high':
                col_map[col] = 'high'
            elif cl == 'low':
                col_map[col] = 'low'
            elif cl == 'close':
                col_map[col] = 'close'
            elif 'volume' in cl:
                col_map[col] = 'volume'

        df = df.rename(columns=col_map)

        required = ['time', 'open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                logger.warning(f"{symbol}: Missing column '{col}'")
                return None

        df = df.copy()
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        invalid = (
            df['time'].isna().any()
            or not np.isfinite(df[numeric_cols].to_numpy(dtype=float)).all()
            or (df[['open', 'high', 'low', 'close']] <= 0).any().any()
            or (df['volume'] < 0).any()
            or (df['high'] < df[['open', 'close']].max(axis=1)).any()
            or (df['low'] > df[['open', 'close']].min(axis=1)).any()
            or (df['high'] < df['low']).any()
            or df['time'].duplicated().any()
        )
        if invalid:
            logger.warning(f"{symbol}: Invalid OHLCV; reject source instead of scoring corrupt data")
            return None
        df = df.sort_values('time').reset_index(drop=True)
        df['symbol'] = symbol
        return df

    def get_ohlcv(self, symbol: str, days: int = 60) -> Optional[pd.DataFrame]:
        """
        Lấy dữ liệu OHLCV cho 1 mã cổ phiếu.
        Retry với exponential backoff trên mỗi source, fallback sang KBS khi VCI hỏng.

        Parameters:
            symbol: Mã cổ phiếu (VD: 'VNM', 'FPT')
            days: Số ngày lịch sử cần lấy
        Returns:
            DataFrame với columns: time, open, high, low, close, volume
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        def _fetch(src: str):
            stock = self.vs.stock(symbol=symbol, source=src)
            return stock.quote.history(start=start_date, end=end_date, interval='1D')

        for src in self._get_sources():
            try:
                df = self._retry_call(_fetch, src)
                result = self._normalize_ohlcv(df, symbol)
                if result is not None:
                    return result
            except Exception as e:
                logger.warning(f"{symbol}: {src} OHLCV failed sau {self.MAX_RETRIES} retries ({e}), trying next source...")

        logger.warning(f"{symbol}: All sources failed for OHLCV")
        return None

    def get_index_data(self, index_symbol: str = 'VNINDEX', days: int = 60) -> Optional[pd.DataFrame]:
        """Lấy dữ liệu chỉ số thị trường (VNINDEX, VN30, HNX...). Retry + fallback KBS."""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        def _fetch(src: str):
            stock = self.vs.stock(symbol=index_symbol, source=src)
            return stock.quote.history(start=start_date, end=end_date, interval='1D')

        for src in self._get_sources():
            try:
                df = self._retry_call(_fetch, src)
                if df is not None and not df.empty:
                    col_map = {}
                    for col in df.columns:
                        cl = col.lower()
                        if 'time' in cl or 'date' in cl:
                            col_map[col] = 'time'
                        elif cl == 'close':
                            col_map[col] = 'close'
                    df = df.rename(columns=col_map)
                    df = df.sort_values('time').reset_index(drop=True)
                    return df
            except Exception as e:
                logger.warning(f"Index {index_symbol}: {src} failed sau {self.MAX_RETRIES} retries ({e}), trying next source...")

        logger.warning(f"Error fetching index {index_symbol}: All sources failed")
        return None

    def get_price_board(self, symbols: List[str]) -> Optional[pd.DataFrame]:
        """Lấy bảng giá real-time cho danh sách mã. Tự động fallback."""
        for src in self._get_sources():
            try:
                stock = self.vs.stock(symbol=symbols[0], source=src)
                df = stock.trading.price_board(symbols_list=symbols)
                return df
            except Exception as e:
                logger.warning(f"Price board: {src} failed ({e}), trying next source...")

        logger.warning("Error fetching price board: All sources failed")
        return None


# ============================================================================
# PHẦN 2: FEATURE ENGINEERING - Tính toán chỉ báo kỹ thuật
# ============================================================================

class FeatureEngine:
    """
    Tính toán các chỉ báo kỹ thuật cần thiết cho bộ lọc.
    Tất cả tính toán đều vectorized trên pandas/numpy.
    """

    @staticmethod
    def compute_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Tính toàn bộ features cho 1 mã cổ phiếu.
        Input: DataFrame OHLCV đã sắp xếp theo thời gian (cũ → mới)
        Output: DataFrame bổ sung các cột chỉ báo
        """
        df = df.copy()

        # --- Moving Averages ---
        df['ma5']  = df['close'].rolling(5).mean()
        df['ma10'] = df['close'].rolling(10).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma50'] = df['close'].rolling(50).mean()

        # --- Volume metrics ---
        df['vol_avg5']  = df['volume'].rolling(5).mean()
        df['vol_avg10'] = df['volume'].rolling(10).mean()
        df['vol_avg20'] = df['volume'].rolling(20).mean()

        # Volume ratio (hôm nay vs trung bình 20 phiên)
        df['vol_ratio_20'] = df['volume'] / df['vol_avg20']

        # --- Price metrics ---
        df['high20'] = df['high'].rolling(20).max()
        df['low20']  = df['low'].rolling(20).min()
        df['high52'] = df['high'].rolling(52 * 5).max()  # ~52 tuần giao dịch

        # % Close so với High20
        df['pct_of_high20'] = df['close'] / df['high20']

        # % thay đổi giá
        df['pct_change_1d']  = df['close'].pct_change(1)
        df['pct_change_5d']  = df['close'].pct_change(5)
        df['pct_change_10d'] = df['close'].pct_change(10)
        df['pct_change_20d'] = df['close'].pct_change(20)

        # --- Biên độ giá (Price Range) ---
        df['range_20d'] = (df['high20'] - df['low20']) / df['low20']

        # --- Râu nến (Upper Shadow) ---
        df['body'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_shadow'] = df[['close', 'open']].min(axis=1) - df['low']
        df['candle_range'] = df['high'] - df['low']
        df['upper_shadow_ratio'] = np.where(
            df['candle_range'] > 0,
            df['upper_shadow'] / df['candle_range'],
            0
        )

        # --- Relative Strength vs Index (sẽ được tính ở Screener) ---
        # Placeholder columns
        df['rs_vs_index_1d']  = np.nan
        df['rs_vs_index_10d'] = np.nan

        # --- Local peaks and troughs detection ---
        df['is_local_high'] = (
            (df['high'] > df['high'].shift(1)) &
            (df['high'] > df['high'].shift(-1))
        )
        df['is_local_low'] = (
            (df['low'] < df['low'].shift(1)) &
            (df['low'] < df['low'].shift(-1))
        )

        return df

    @staticmethod
    def compute_relative_strength(stock_df: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
        """
        Tính sức mạnh tương đối (Relative Strength) so với VNINDEX.

        RS = %Change_stock - %Change_index
        RS > 0: Cổ phiếu mạnh hơn thị trường
        """
        stock_df = stock_df.copy()

        if index_df is None or index_df.empty:
            return stock_df

        # Merge theo ngày gần nhất
        idx_close = index_df[['time', 'close']].rename(columns={'close': 'index_close'})
        stock_df = stock_df.merge(idx_close, on='time', how='left')
        stock_df['index_close'] = stock_df['index_close'].ffill()

        # RS 1 ngày
        stock_df['index_pct_1d'] = stock_df['index_close'].pct_change(1)
        stock_df['rs_vs_index_1d'] = stock_df['pct_change_1d'] - stock_df['index_pct_1d']

        # RS 10 ngày
        stock_df['index_pct_10d'] = stock_df['index_close'].pct_change(10)
        stock_df['rs_vs_index_10d'] = stock_df['pct_change_10d'] - stock_df['index_pct_10d']

        return stock_df

    @staticmethod
    def detect_higher_highs_higher_lows(df: pd.DataFrame, lookback: int = 20) -> bool:
        """
        Kiểm tra cấu trúc giá tăng: đỉnh sau > đỉnh trước, đáy sau > đáy trước.
        (Tiêu chí 6 - Nhóm Mạnh)
        """
        recent = df.tail(lookback)

        highs = recent[recent['is_local_high']]['high'].values
        lows  = recent[recent['is_local_low']]['low'].values

        if len(highs) >= 2 and len(lows) >= 2:
            hh = highs[-1] > highs[-2]  # Higher High
            hl = lows[-1] > lows[-2]    # Higher Low
            return hh and hl
        return False

    @staticmethod
    def detect_spring(df: pd.DataFrame, lookback: int = 10) -> bool:
        """
        Phát hiện Spring (rũ bỏ): Giá thủng hỗ trợ trong phiên nhưng đóng cửa cao.
        (Tiêu chí 7 - Nhóm Tích lũy)

        Logic: Low < Low20 trước đó NHƯNG Close > Open (nến tăng)
        """
        if len(df) < lookback + 1:
            return False

        recent = df.tail(lookback)
        low20_prev = df['low'].rolling(20).min().shift(1)

        for idx in recent.index:
            if idx not in low20_prev.index:
                continue
            support = low20_prev.get(idx, np.nan)
            if pd.isna(support):
                continue
            row = df.loc[idx]
            if row['low'] < support and row['close'] > row['open']:
                return True
        return False

    @staticmethod
    def detect_failed_breakout(df: pd.DataFrame, lookback: int = 10) -> bool:
        """
        Phát hiện Break thất bại: Giá vượt đỉnh nhưng nhanh chóng quay xuống.
        (Tiêu chí 3 - Nhóm Phân phối)
        """
        if len(df) < 21:
            return False

        recent = df.tail(lookback)

        for i in range(1, len(recent)):
            idx = recent.index[i]
            prev_idx = recent.index[i - 1]

            prev_high20 = df.loc[:prev_idx, 'high'].tail(20).max()
            row = df.loc[idx]

            # Giá từng vượt đỉnh (high > high20) nhưng close < prev_high20
            if row['high'] > prev_high20 and row['close'] < prev_high20:
                return True
        return False

    @staticmethod
    def count_distribution_sessions(df: pd.DataFrame, lookback: int = 10) -> int:
        """
        Đếm số phiên phân phối: Volume lớn nhưng giá đi ngang.
        (Tiêu chí 7 - Nhóm Phân phối)
        """
        if len(df) < 20:
            return 0

        recent = df.tail(lookback)
        count = 0

        for idx in recent.index:
            row = df.loc[idx]
            vol_avg20 = df.loc[:idx, 'volume'].tail(20).mean()

            if pd.isna(vol_avg20) or vol_avg20 == 0:
                continue

            vol_ratio = row['volume'] / vol_avg20
            abs_change = abs(row['pct_change_1d']) if not pd.isna(row['pct_change_1d']) else 0

            # Volume >= 1.5x avg AND giá không đổi đáng kể (< 1.5%)
            if vol_ratio >= 1.5 and abs_change <= 0.015:
                count += 1

        return count

    @staticmethod
    def detect_absorption(df: pd.DataFrame, lookback: int = 20) -> bool:
        """
        Hấp thụ cung: Volume >= 2x AvgVol20 nhưng giá không giảm mạnh.
        (Tiêu chí 3 - Nhóm Tích lũy)
        """
        if len(df) < 20:
            return False

        recent = df.tail(lookback)

        for idx in recent.index:
            row = df.loc[idx]
            vol_avg20 = df.loc[:idx, 'volume'].tail(20).mean()

            if pd.isna(vol_avg20) or vol_avg20 == 0:
                continue

            vol_ratio = row['volume'] / vol_avg20
            pct_chg = row['pct_change_1d'] if not pd.isna(row['pct_change_1d']) else 0

            # Volume >= 2x VÀ giá không giảm quá -2%
            if vol_ratio >= 2.0 and pct_chg > -0.02:
                return True

        return False


# ============================================================================
# PHẦN 3: SCREENING ENGINE - Áp dụng 21 tiêu chí lọc
# ============================================================================

class ScreeningEngine:
    """
    Áp dụng 3 bộ tiêu chí lọc, tính điểm và xếp hạng cho mỗi cổ phiếu.

    Mỗi tiêu chí trả về True/False.
    Điểm tổng = Số tiêu chí đạt / Tổng tiêu chí (0-100%).
    """

    # --------------------------------------------------------
    # NHÓM 1: CỔ PHIẾU ĐANG MẠNH (7 tiêu chí)
    # --------------------------------------------------------

    @staticmethod
    def screen_strong(df: pd.DataFrame) -> Dict[str, any]:
        """
        Lọc cổ phiếu đang MẠNH theo 7 tiêu chí.

        Góc nhìn NĐT: "Cổ phiếu nào đang dẫn dắt thị trường?"
        Góc nhìn Môi giới: "Mã nào đang có momentum mạnh để tư vấn?"
        """
        if df is None or len(df) < 20:
            return {'score': 0, 'criteria': {}, 'detail': {}}

        last = df.iloc[-1]
        criteria = {}
        detail = {}

        # 1. Sức mạnh so với thị trường
        rs_1d = last.get('rs_vs_index_1d', np.nan)
        criteria['S1_RS_market'] = bool(not pd.isna(rs_1d) and rs_1d > 0)
        detail['S1'] = f"RS 1D = {rs_1d:.4f}" if not pd.isna(rs_1d) else "N/A"

        # 2. Volume đột biến (>= 1.5x AvgVol20)
        vol_ratio = last.get('vol_ratio_20', 0)
        criteria['S2_vol_spike'] = bool(not pd.isna(vol_ratio) and vol_ratio >= 1.5)
        detail['S2'] = f"Vol Ratio = {vol_ratio:.2f}x"

        # 3. Giá gần đỉnh 20 phiên (Close >= 95% High20)
        pct_high20 = last.get('pct_of_high20', 0)
        criteria['S3_near_high20'] = bool(not pd.isna(pct_high20) and pct_high20 >= 0.95)
        detail['S3'] = f"% of High20 = {pct_high20:.2%}"

        # 4. Volume giảm khi điều chỉnh (Vol5 < Vol20)
        vol5  = last.get('vol_avg5', np.nan)
        vol20 = last.get('vol_avg20', np.nan)
        criteria['S4_vol_dry_pullback'] = bool(
            not pd.isna(vol5) and not pd.isna(vol20) and vol5 < vol20
        )
        detail['S4'] = f"AvgVol5={vol5:,.0f} vs AvgVol20={vol20:,.0f}" if not pd.isna(vol5) else "N/A"

        # 5. Break đỉnh (Close >= High20)
        close = last['close']
        high20 = last.get('high20', np.nan)
        criteria['S5_breakout_high20'] = bool(
            not pd.isna(high20) and close >= high20
        )
        detail['S5'] = f"Close={close:,.0f} vs High20={high20:,.0f}" if not pd.isna(high20) else "N/A"

        # 6. Cấu trúc giá tăng (HH-HL)
        hh_hl = FeatureEngine.detect_higher_highs_higher_lows(df, lookback=20)
        criteria['S6_hh_hl_structure'] = hh_hl
        detail['S6'] = "Higher-High & Higher-Low" if hh_hl else "Không rõ cấu trúc"

        # 7. Relative Strength 10 ngày > VNINDEX
        rs_10d = last.get('rs_vs_index_10d', np.nan)
        criteria['S7_RS_10d'] = bool(not pd.isna(rs_10d) and rs_10d > 0)
        detail['S7'] = f"RS 10D = {rs_10d:.4f}" if not pd.isna(rs_10d) else "N/A"

        score = sum(criteria.values()) / len(criteria) * 100

        return {
            'score': round(score, 1),
            'criteria': criteria,
            'detail': detail,
            'met_count': sum(criteria.values()),
            'total_criteria': len(criteria)
        }

    # --------------------------------------------------------
    # NHÓM 2: CỔ PHIẾU TÍCH LŨY (7 tiêu chí)
    # --------------------------------------------------------

    @staticmethod
    def screen_accumulation(df: pd.DataFrame) -> Dict[str, any]:
        """
        Lọc cổ phiếu đang TÍCH LŨY chuẩn bị tăng.

        Góc nhìn NĐT: "Cổ phiếu nào đang nằm chờ, chuẩn bị bùng nổ?"
        Góc nhìn Môi giới: "Mã nào đáng để theo dõi cho cú breakout sắp tới?"
        """
        if df is None or len(df) < 20:
            return {'score': 0, 'criteria': {}, 'detail': {}}

        last = df.iloc[-1]
        criteria = {}
        detail = {}

        # 1. Sideway chặt (biên độ 20 ngày <= 15%)
        range_20d = last.get('range_20d', np.nan)
        criteria['A1_tight_range'] = bool(not pd.isna(range_20d) and range_20d <= 0.15)
        detail['A1'] = f"Range 20D = {range_20d:.2%}" if not pd.isna(range_20d) else "N/A"

        # 2. Volume giảm dần (Vol5 < Vol20)
        vol5  = last.get('vol_avg5', np.nan)
        vol20 = last.get('vol_avg20', np.nan)
        criteria['A2_vol_declining'] = bool(
            not pd.isna(vol5) and not pd.isna(vol20) and vol5 < vol20
        )
        detail['A2'] = f"AvgVol5={vol5:,.0f} < AvgVol20={vol20:,.0f}" if not pd.isna(vol5) else "N/A"

        # 3. Hấp thụ cung
        absorption = FeatureEngine.detect_absorption(df, lookback=20)
        criteria['A3_absorption'] = absorption
        detail['A3'] = "Có hấp thụ cung" if absorption else "Chưa phát hiện"

        # 4. Giữ hỗ trợ MA20 (Close > MA20)
        ma20 = last.get('ma20', np.nan)
        criteria['A4_above_ma20'] = bool(
            not pd.isna(ma20) and last['close'] > ma20
        )
        detail['A4'] = f"Close={last['close']:,.0f} vs MA20={ma20:,.0f}" if not pd.isna(ma20) else "N/A"

        # 5. Gần kháng cự (Close >= 90% High20)
        pct_high20 = last.get('pct_of_high20', 0)
        criteria['A5_near_resistance'] = bool(
            not pd.isna(pct_high20) and pct_high20 >= 0.90
        )
        detail['A5'] = f"% of High20 = {pct_high20:.2%}"

        # 6. Tích lũy dài (sideway >= 1 tháng = range nhỏ qua nhiều phiên)
        # Kiểm tra: range 20 ngày đều nhỏ trong 20 phiên gần nhất
        if len(df) >= 40:
            ranges_last20 = df['range_20d'].tail(20)
            long_accumulation = bool((ranges_last20 <= 0.15).sum() >= 15)
        else:
            long_accumulation = False
        criteria['A6_long_accumulation'] = long_accumulation
        detail['A6'] = "Tích lũy >= 1 tháng" if long_accumulation else "Chưa đủ thời gian"

        # 7. Spring (rũ bỏ)
        spring = FeatureEngine.detect_spring(df, lookback=10)
        criteria['A7_spring'] = spring
        detail['A7'] = "Phát hiện Spring" if spring else "Không có Spring"

        score = sum(criteria.values()) / len(criteria) * 100

        return {
            'score': round(score, 1),
            'criteria': criteria,
            'detail': detail,
            'met_count': sum(criteria.values()),
            'total_criteria': len(criteria)
        }

    # --------------------------------------------------------
    # NHÓM 3: CỔ PHIẾU PHÂN PHỐI (7 tiêu chí)
    # --------------------------------------------------------

    @staticmethod
    def screen_distribution(df: pd.DataFrame) -> Dict[str, any]:
        """
        Lọc cổ phiếu đang PHÂN PHỐI (cảnh báo).

        Góc nhìn NĐT: "Cổ phiếu nào đang bị bán ra, nên tránh?"
        Góc nhìn Môi giới: "Mã nào cần cảnh báo khách hàng?"
        """
        if df is None or len(df) < 20:
            return {'score': 0, 'criteria': {}, 'detail': {}}

        last = df.iloc[-1]
        criteria = {}
        detail = {}

        # 1. Volume lớn nhưng giá không tăng
        vol_ratio = last.get('vol_ratio_20', 0)
        pct_chg = last.get('pct_change_1d', 0) or 0
        criteria['D1_vol_no_price_up'] = bool(
            not pd.isna(vol_ratio) and vol_ratio >= 2.0 and abs(pct_chg) <= 0.01
        )
        detail['D1'] = f"VolRatio={vol_ratio:.2f}x, Change={pct_chg:.2%}"

        # 2. Râu nến trên dài (upper shadow > 50% candle range)
        us_ratio = last.get('upper_shadow_ratio', 0)
        criteria['D2_long_upper_shadow'] = bool(
            not pd.isna(us_ratio) and us_ratio >= 0.5
        )
        detail['D2'] = f"Upper Shadow Ratio = {us_ratio:.2%}"

        # 3. Break thất bại
        failed_breakout = FeatureEngine.detect_failed_breakout(df, lookback=10)
        criteria['D3_failed_breakout'] = failed_breakout
        detail['D3'] = "Có break thất bại" if failed_breakout else "Không"

        # 4. Volume tăng khi giá giảm
        # So sánh avg volume phiên giảm vs phiên tăng trong 10 phiên gần nhất
        recent = df.tail(10)
        down_vol = recent[recent['pct_change_1d'] < 0]['volume'].mean()
        up_vol   = recent[recent['pct_change_1d'] > 0]['volume'].mean()
        criteria['D4_vol_on_down'] = bool(
            not pd.isna(down_vol) and not pd.isna(up_vol) and up_vol > 0 and down_vol > up_vol
        )
        detail['D4'] = f"DownVol={down_vol:,.0f} vs UpVol={up_vol:,.0f}" if not pd.isna(down_vol) else "N/A"

        # 5. Thủng MA20 (Close < MA20)
        ma20 = last.get('ma20', np.nan)
        criteria['D5_below_ma20'] = bool(
            not pd.isna(ma20) and last['close'] < ma20
        )
        detail['D5'] = f"Close={last['close']:,.0f} < MA20={ma20:,.0f}" if not pd.isna(ma20) else "N/A"

        # 6. Giảm mạnh từ đỉnh (>= 5% từ high20)
        high20 = last.get('high20', np.nan)
        if not pd.isna(high20) and high20 > 0:
            drop_from_peak = (high20 - last['close']) / high20
        else:
            drop_from_peak = 0
        criteria['D6_drop_from_peak'] = bool(drop_from_peak >= 0.05)
        detail['D6'] = f"Giảm {drop_from_peak:.2%} từ đỉnh"

        # 7. Phân phối nhiều phiên (3-5 phiên vol lớn giá ngang)
        dist_sessions = FeatureEngine.count_distribution_sessions(df, lookback=10)
        criteria['D7_multi_dist_sessions'] = bool(dist_sessions >= 3)
        detail['D7'] = f"{dist_sessions} phiên phân phối"

        score = sum(criteria.values()) / len(criteria) * 100

        return {
            'score': round(score, 1),
            'criteria': criteria,
            'detail': detail,
            'met_count': sum(criteria.values()),
            'total_criteria': len(criteria)
        }


# ============================================================================
# PHẦN 4: ANALYSIS ENGINE - Luận điểm đánh giá tổng hợp
# ============================================================================

class AnalysisEngine:
    """
    Tổng hợp kết quả lọc thành các luận điểm đánh giá cho NĐT & Môi giới.

    Cung cấp góc nhìn:
    - Dòng tiền (Money Flow): Volume, hấp thụ, phân phối
    - Xu hướng (Trend): MA, cấu trúc giá, relative strength
    - Sức mạnh (Strength): RS vs market, breakout, range
    """

    @staticmethod
    def generate_assessment(
        symbol: str,
        df: pd.DataFrame,
        strong_result: Dict,
        accum_result: Dict,
        dist_result: Dict
    ) -> Dict:
        """
        Tạo luận điểm đánh giá tổng hợp cho 1 cổ phiếu.
        """
        if df is None or df.empty:
            return {
                'symbol': symbol,
                'overall_signal': 'N/A',
                'money_flow': 'N/A',
                'trend': 'N/A',
                'strength': 'N/A',
                'investor_note': 'Không đủ dữ liệu',
                'broker_note': 'Không đủ dữ liệu'
            }

        last = df.iloc[-1]
        assessment = {'symbol': symbol}

        # --- Xác định tín hiệu tổng hợp ---
        s_score = strong_result['score']
        a_score = accum_result['score']
        d_score = dist_result['score']

        if s_score >= 57 and d_score < 43:
            assessment['overall_signal'] = '🟢 MẠNH'
            assessment['signal_code'] = 'STRONG'
        elif a_score >= 57 and d_score < 43:
            assessment['overall_signal'] = '🟡 TÍCH LŨY'
            assessment['signal_code'] = 'ACCUMULATION'
        elif d_score >= 57:
            assessment['overall_signal'] = '🔴 PHÂN PHỐI'
            assessment['signal_code'] = 'DISTRIBUTION'
        elif s_score >= 43:
            assessment['overall_signal'] = '🟢 THIÊN MẠNH'
            assessment['signal_code'] = 'LEAN_STRONG'
        elif a_score >= 43:
            assessment['overall_signal'] = '🟡 THIÊN TÍCH LŨY'
            assessment['signal_code'] = 'LEAN_ACCUM'
        else:
            assessment['overall_signal'] = '⚪ TRUNG LẬP'
            assessment['signal_code'] = 'NEUTRAL'

        # --- Góc nhìn DÒNG TIỀN ---
        vol_ratio = last.get('vol_ratio_20', 0) or 0
        vol5 = last.get('vol_avg5', 0) or 0
        vol20 = last.get('vol_avg20', 1) or 1

        if vol_ratio >= 2.0 and last.get('pct_change_1d', 0) > 0.01:
            money_flow = "Dòng tiền MẠNH - Volume đột biến kèm giá tăng. Tiền lớn đang MUA."
        elif vol_ratio >= 2.0 and abs(last.get('pct_change_1d', 0)) <= 0.01:
            money_flow = "Dòng tiền BẤT THƯỜNG - Volume lớn nhưng giá đi ngang. Có thể phân phối."
        elif vol_ratio >= 2.0 and last.get('pct_change_1d', 0) < -0.01:
            money_flow = "Dòng tiền XẤU - Volume lớn kèm giá giảm. Tiền lớn đang BÁN."
        elif vol5 < vol20 * 0.7:
            money_flow = "Dòng tiền CẠN - Khối lượng suy giảm mạnh. Thị trường thiếu quan tâm."
        elif vol5 < vol20:
            money_flow = "Dòng tiền GIẢM DẦN - Giai đoạn chờ. Cung cầu đang cân bằng."
        else:
            money_flow = "Dòng tiền ỔN ĐỊNH - Thanh khoản bình thường."

        assessment['money_flow'] = money_flow

        # --- Góc nhìn XU HƯỚNG ---
        ma20 = last.get('ma20', np.nan)
        ma50 = last.get('ma50', np.nan)
        close = last['close']

        trend_signals = []
        if not pd.isna(ma20):
            if close > ma20:
                trend_signals.append("Trên MA20 ✓")
            else:
                trend_signals.append("Dưới MA20 ✗")
        if not pd.isna(ma50):
            if close > ma50:
                trend_signals.append("Trên MA50 ✓")
            else:
                trend_signals.append("Dưới MA50 ✗")
        if not pd.isna(ma20) and not pd.isna(ma50):
            if ma20 > ma50:
                trend_signals.append("MA20 > MA50 (Golden)")
            else:
                trend_signals.append("MA20 < MA50 (Death)")

        hh_hl = strong_result['criteria'].get('S6_hh_hl_structure', False)
        if hh_hl:
            trend_signals.append("Đỉnh-đáy tăng dần ✓")

        assessment['trend'] = " | ".join(trend_signals) if trend_signals else "Chưa xác định"

        # --- Góc nhìn SỨC MẠNH ---
        rs_1d  = last.get('rs_vs_index_1d', 0) or 0
        rs_10d = last.get('rs_vs_index_10d', 0) or 0
        pct_high20 = last.get('pct_of_high20', 0) or 0

        if rs_10d > 0.03 and pct_high20 >= 0.95:
            strength = "SỨC MẠNH CAO - Dẫn dắt thị trường, gần đỉnh. Momentum rất tốt."
        elif rs_10d > 0 and pct_high20 >= 0.90:
            strength = "SỨC MẠNH TRÊN TB - Outperform thị trường, vùng giá tích cực."
        elif rs_10d < -0.03:
            strength = "SỨC MẠNH YẾU - Thua thị trường đáng kể. Cần cẩn trọng."
        else:
            strength = "SỨC MẠNH TRUNG BÌNH - Đi ngang so với thị trường."

        assessment['strength'] = strength

        # --- Ghi chú cho NĐT ---
        if assessment['signal_code'] == 'STRONG':
            assessment['investor_note'] = (
                f"Cổ phiếu đang trong xu hướng tăng mạnh ({s_score:.0f}% tiêu chí mạnh). "
                f"Dòng tiền tích cực, có thể cân nhắc THAM GIA hoặc NẮM GIỮ. "
                f"Lưu ý đặt trailing stop để bảo vệ lợi nhuận."
            )
        elif assessment['signal_code'] == 'ACCUMULATION':
            assessment['investor_note'] = (
                f"Cổ phiếu đang tích lũy ({a_score:.0f}% tiêu chí). "
                f"Đây là giai đoạn chờ breakout. Có thể THEO DÕI và mua khi có tín hiệu phá vỡ vùng kháng cự."
            )
        elif assessment['signal_code'] == 'DISTRIBUTION':
            assessment['investor_note'] = (
                f"⚠️ CẢNH BÁO: Cổ phiếu có dấu hiệu phân phối ({d_score:.0f}% tiêu chí). "
                f"Dòng tiền lớn đang thoát hàng. NÊN CÂN NHẮC CẮT LỖ hoặc GIẢM VỊ THẾ."
            )
        else:
            assessment['investor_note'] = (
                f"Tín hiệu chưa rõ ràng. Mạnh: {s_score:.0f}% | Tích lũy: {a_score:.0f}% | "
                f"Phân phối: {d_score:.0f}%. Nên CHỜ ĐỢI thêm tín hiệu."
            )

        # --- Ghi chú cho Môi giới ---
        assessment['broker_note'] = (
            f"Điểm: Mạnh {s_score:.0f}% | Tích lũy {a_score:.0f}% | Phân phối {d_score:.0f}% | "
            f"Vol Ratio: {vol_ratio:.1f}x | RS(10D): {rs_10d:.4f} | "
            f"Vị trí giá: {pct_high20:.1%} so với High20"
        )

        return assessment

    @staticmethod
    def get_price_summary(df: pd.DataFrame) -> Dict:
        """Tóm tắt giá và volume hiện tại."""
        if df is None or df.empty:
            return {}

        last = df.iloc[-1]
        return {
            'close': last['close'],
            'volume': last['volume'],
            'pct_change_1d': last.get('pct_change_1d', 0),
            'pct_change_5d': last.get('pct_change_5d', 0),
            'pct_change_20d': last.get('pct_change_20d', 0),
            'ma20': last.get('ma20', np.nan),
            'vol_ratio_20': last.get('vol_ratio_20', 0),
            'pct_of_high20': last.get('pct_of_high20', 0),
        }


# ============================================================================
# PHẦN 4B: MARKET ANALYZER - Phân tích sức khỏe VNINDEX
# ============================================================================

class MarketAnalyzer:
    """
    Phân tích sức khỏe thị trường thông qua VNINDEX.

    Cung cấp bối cảnh vĩ mô TRƯỚC KHI đánh giá từng cổ phiếu:
    - Xu hướng VNINDEX (MA, cấu trúc giá)
    - Dòng tiền thị trường (Volume, breadth)
    - Regime detection (Bull / Bear / Sideway)
    - Market breadth (% mã tăng/giảm)
    - Mức độ rủi ro hiện tại

    Góc nhìn NĐT: "Thị trường chung thế nào? Có nên tham gia không?"
    Góc nhìn Môi giới: "Nên tư vấn tấn công hay phòng thủ?"
    """

    def __init__(self, data_provider):
        self.data_provider = data_provider

    def analyze_index(self, index_symbol: str = 'VNINDEX', days: int = 90) -> Dict:
        """
        Phân tích toàn diện VNINDEX.

        Returns:
            Dict chứa:
            - regime: Bull / Bear / Sideway
            - trend_score: 0-100 (điểm xu hướng)
            - money_flow_score: 0-100 (điểm dòng tiền)
            - risk_level: Thấp / Trung bình / Cao
            - summary: Nhận xét tổng hợp
            - detail: Dict chi tiết từng chỉ báo
            - raw_data: DataFrame VNINDEX đã tính features
        """
        # Lấy dữ liệu VNINDEX đầy đủ (OHLCV)
        index_df = self.data_provider.get_ohlcv(index_symbol, days=days)

        if index_df is None or len(index_df) < 20:
            logger.warning(f"Không đủ dữ liệu {index_symbol}")
            return self._empty_result(index_symbol)

        # Tính features cho VNINDEX
        fe = FeatureEngine()
        index_df = fe.compute_features(index_df)

        last = index_df.iloc[-1]
        result = {
            'index_symbol': index_symbol,
            'last_close': last['close'],
            'last_date': str(last.get('time', '')),
            'detail': {},
        }

        # ============================
        # 1. XU HƯỚNG (Trend Analysis)
        # ============================
        trend_points = 0
        trend_detail = {}

        # 1.1 Vị trí giá vs MA
        ma20 = last.get('ma20', np.nan)
        ma50 = last.get('ma50', np.nan)
        close = last['close']

        above_ma20 = bool(not pd.isna(ma20) and close > ma20)
        above_ma50 = bool(not pd.isna(ma50) and close > ma50)
        golden_cross = bool(not pd.isna(ma20) and not pd.isna(ma50) and ma20 > ma50)

        if above_ma20:
            trend_points += 20
        if above_ma50:
            trend_points += 20
        if golden_cross:
            trend_points += 15

        trend_detail['above_ma20'] = above_ma20
        trend_detail['above_ma50'] = above_ma50
        trend_detail['golden_cross'] = golden_cross
        trend_detail['ma20'] = round(ma20, 2) if not pd.isna(ma20) else None
        trend_detail['ma50'] = round(ma50, 2) if not pd.isna(ma50) else None

        # 1.2 Cấu trúc đỉnh-đáy
        hh_hl = fe.detect_higher_highs_higher_lows(index_df, lookback=20)
        if hh_hl:
            trend_points += 20
        trend_detail['higher_highs_higher_lows'] = hh_hl

        # 1.3 % thay đổi các khung thời gian
        pct_5d  = last.get('pct_change_5d', 0) or 0
        pct_10d = last.get('pct_change_10d', 0) or 0
        pct_20d = last.get('pct_change_20d', 0) or 0

        if pct_5d > 0:
            trend_points += 5
        if pct_10d > 0:
            trend_points += 10
        if pct_20d > 0:
            trend_points += 10

        trend_detail['pct_change_5d']  = round(pct_5d * 100, 2)
        trend_detail['pct_change_10d'] = round(pct_10d * 100, 2)
        trend_detail['pct_change_20d'] = round(pct_20d * 100, 2)

        result['trend_score'] = min(trend_points, 100)

        # ============================
        # 2. DÒNG TIỀN (Money Flow)
        # ============================
        mf_points = 0
        mf_detail = {}

        vol_ratio = last.get('vol_ratio_20', 0) or 0
        vol5  = last.get('vol_avg5', 0) or 0
        vol20 = last.get('vol_avg20', 1) or 1

        # 2.1 Volume hôm nay vs trung bình
        if vol_ratio >= 1.5:
            mf_points += 25
        elif vol_ratio >= 1.0:
            mf_points += 15
        elif vol_ratio >= 0.7:
            mf_points += 5

        mf_detail['vol_ratio_today'] = round(vol_ratio, 2)

        # 2.2 Xu hướng volume 5 phiên vs 20 phiên
        vol_trend = vol5 / vol20 if vol20 > 0 else 0
        if vol_trend >= 1.2:
            mf_points += 25  # Volume đang tăng
        elif vol_trend >= 0.8:
            mf_points += 15  # Volume ổn định
        else:
            mf_points += 0   # Volume suy giảm

        mf_detail['vol_5d_vs_20d'] = round(vol_trend, 2)
        mf_detail['avg_vol_5d']  = round(vol5, 0)
        mf_detail['avg_vol_20d'] = round(vol20, 0)

        # 2.3 Volume tăng trong ngày tăng vs ngày giảm (10 phiên)
        recent = index_df.tail(10)
        up_days   = recent[recent['pct_change_1d'] > 0]
        down_days = recent[recent['pct_change_1d'] < 0]

        avg_up_vol   = up_days['volume'].mean() if len(up_days) > 0 else 0
        avg_down_vol = down_days['volume'].mean() if len(down_days) > 0 else 0

        if avg_up_vol > avg_down_vol * 1.2:
            mf_points += 25  # Tiền vào mạnh hơn tiền ra
        elif avg_up_vol > avg_down_vol:
            mf_points += 15
        else:
            mf_points += 0   # Tiền ra mạnh hơn

        mf_detail['avg_up_day_vol']   = round(avg_up_vol, 0)
        mf_detail['avg_down_day_vol'] = round(avg_down_vol, 0)
        mf_detail['up_days_count']    = len(up_days)
        mf_detail['down_days_count']  = len(down_days)

        # 2.4 Tỷ lệ ngày tăng/giảm trong 10 phiên
        up_ratio = len(up_days) / 10
        if up_ratio >= 0.7:
            mf_points += 25
        elif up_ratio >= 0.5:
            mf_points += 15
        else:
            mf_points += 5

        mf_detail['up_day_ratio_10d'] = round(up_ratio * 100, 1)

        result['money_flow_score'] = min(mf_points, 100)

        # ============================
        # 3. BIẾN ĐỘNG (Volatility)
        # ============================
        vol_detail = {}

        # ATR-like: average true range 14 phiên
        if len(index_df) >= 14:
            tr = pd.DataFrame({
                'hl': index_df['high'] - index_df['low'],
                'hc': abs(index_df['high'] - index_df['close'].shift(1)),
                'lc': abs(index_df['low'] - index_df['close'].shift(1))
            })
            index_df['true_range'] = tr.max(axis=1)
            atr_14 = index_df['true_range'].rolling(14).mean().iloc[-1]
            atr_pct = atr_14 / close * 100 if close > 0 else 0
        else:
            atr_14 = 0
            atr_pct = 0

        vol_detail['atr_14'] = round(atr_14, 2)
        vol_detail['atr_pct'] = round(atr_pct, 2)

        # Range 20 ngày
        range_20d = last.get('range_20d', 0) or 0
        vol_detail['range_20d_pct'] = round(range_20d * 100, 2)

        # Vị trí giá trong range
        pct_of_high20 = last.get('pct_of_high20', 0) or 0
        vol_detail['pct_of_high20'] = round(pct_of_high20 * 100, 1)

        # ============================
        # 4. XÁC ĐỊNH REGIME
        # ============================
        t_score = result['trend_score']
        m_score = result['money_flow_score']
        combined = (t_score * 0.6) + (m_score * 0.4)  # Trend nặng hơn

        if combined >= 65:
            regime = 'BULL'
            regime_label = '🟢 BULL - Xu hướng tăng'
        elif combined >= 45:
            regime = 'SIDEWAY'
            regime_label = '🟡 SIDEWAY - Đi ngang'
        else:
            regime = 'BEAR'
            regime_label = '🔴 BEAR - Xu hướng giảm'

        result['regime'] = regime
        result['regime_label'] = regime_label
        result['combined_score'] = round(combined, 1)

        # ============================
        # 5. MỨC ĐỘ RỦI RO
        # ============================
        if regime == 'BULL' and atr_pct < 1.5:
            risk = 'THẤP'
            risk_label = '🟢 Rủi ro THẤP'
        elif regime == 'BEAR' or atr_pct > 2.5:
            risk = 'CAO'
            risk_label = '🔴 Rủi ro CAO'
        else:
            risk = 'TRUNG BÌNH'
            risk_label = '🟡 Rủi ro TRUNG BÌNH'

        result['risk_level'] = risk
        result['risk_label'] = risk_label

        # ============================
        # 6. KHUYẾN NGHỊ CHIẾN LƯỢC
        # ============================
        # Cho NĐT
        if regime == 'BULL':
            investor_strategy = (
                f"Thị trường đang trong xu hướng TĂNG (Score: {combined:.0f}/100). "
                f"VNI {pct_5d*100:+.1f}% tuần | {pct_20d*100:+.1f}% tháng. "
                f"Có thể MUA cổ phiếu mạnh, ưu tiên breakout với volume. "
                f"Tỷ trọng cổ phiếu: 70-100% danh mục."
            )
        elif regime == 'SIDEWAY':
            investor_strategy = (
                f"Thị trường SIDEWAY (Score: {combined:.0f}/100). "
                f"VNI {pct_5d*100:+.1f}% tuần | {pct_20d*100:+.1f}% tháng. "
                f"Chỉ mua cổ phiếu tích lũy sắp breakout. Giữ 30-50% tiền mặt. "
                f"Trading ngắn hạn, chốt lời nhanh ở vùng kháng cự."
            )
        else:
            investor_strategy = (
                f"Thị trường GIẢM (Score: {combined:.0f}/100). "
                f"VNI {pct_5d*100:+.1f}% tuần | {pct_20d*100:+.1f}% tháng. "
                f"HẠN CHẾ MUA MỚI. Giữ 70-100% tiền mặt. "
                f"Chỉ quan sát, chờ tín hiệu đảo chiều (VNI hồi trên MA20 + volume tăng)."
            )

        result['investor_strategy'] = investor_strategy

        # Cho Môi giới
        if regime == 'BULL':
            broker_strategy = (
                f"REGIME: BULL | Score: {combined:.0f} | "
                f"Trend: {t_score}/100 | MF: {m_score}/100 | "
                f"ATR: {atr_pct:.1f}% | Risk: {risk} | "
                f"Chiến lược: Tấn công. Đẩy mạnh tư vấn mua CP mạnh. "
                f"Nâng tỷ trọng danh mục khách hàng."
            )
        elif regime == 'SIDEWAY':
            broker_strategy = (
                f"REGIME: SIDEWAY | Score: {combined:.0f} | "
                f"Trend: {t_score}/100 | MF: {m_score}/100 | "
                f"ATR: {atr_pct:.1f}% | Risk: {risk} | "
                f"Chiến lược: Chọn lọc. Chỉ tư vấn mã tích lũy tốt. "
                f"Ưu tiên bảo toàn vốn. Trading ngắn."
            )
        else:
            broker_strategy = (
                f"REGIME: BEAR | Score: {combined:.0f} | "
                f"Trend: {t_score}/100 | MF: {m_score}/100 | "
                f"ATR: {atr_pct:.1f}% | Risk: {risk} | "
                f"Chiến lược: Phòng thủ. Cảnh báo khách giảm vị thế. "
                f"Không tư vấn mua mới. Theo dõi tín hiệu đảo chiều."
            )

        result['broker_strategy'] = broker_strategy

        # Chi tiết
        result['detail'] = {
            'trend': trend_detail,
            'money_flow': mf_detail,
            'volatility': vol_detail,
        }
        result['raw_data'] = index_df

        return result

    def get_screening_adjustments(self, market_result: Dict) -> Dict:
        """
        Dựa trên regime thị trường, điều chỉnh ngưỡng lọc cổ phiếu.

        Bull → nới lỏng tiêu chí mạnh, chặt tiêu chí phân phối
        Bear → siết tiêu chí mạnh, nới lỏng tiêu chí phân phối
        Sideway → giữ nguyên

        Returns:
            Dict với ngưỡng điểm đề xuất cho mỗi nhóm
        """
        regime = market_result.get('regime', 'SIDEWAY')

        if regime == 'BULL':
            return {
                'strong_threshold': 42.9,   # 3/7 là đủ (nới lỏng)
                'accum_threshold': 57.1,    # 4/7 (bình thường)
                'dist_threshold': 71.4,     # 5/7 mới cảnh báo (siết)
                'note': 'BULL: Nới lỏng tiêu chí mạnh, siết tiêu chí phân phối'
            }
        elif regime == 'BEAR':
            return {
                'strong_threshold': 71.4,   # 5/7 mới được gọi là mạnh (siết)
                'accum_threshold': 57.1,    # 4/7 (bình thường)
                'dist_threshold': 42.9,     # 3/7 đã cảnh báo (nới lỏng)
                'note': 'BEAR: Siết tiêu chí mạnh, nới lỏng cảnh báo phân phối'
            }
        else:
            return {
                'strong_threshold': 57.1,   # 4/7 (mặc định)
                'accum_threshold': 57.1,
                'dist_threshold': 57.1,
                'note': 'SIDEWAY: Giữ nguyên ngưỡng mặc định'
            }

    def format_market_report(self, market_result: Dict) -> str:
        """Tạo báo cáo thị trường dạng text để in ra console hoặc gửi khách."""
        r = market_result
        d = r.get('detail', {})
        t = d.get('trend', {})
        m = d.get('money_flow', {})
        v = d.get('volatility', {})

        report = f"""
{'='*70}
  📊 BÁO CÁO THỊ TRƯỜNG - {r.get('index_symbol', 'VNINDEX')}
  Ngày: {r.get('last_date', 'N/A')} | Giá: {r.get('last_close', 0):,.2f}
{'='*70}

  🏷️  REGIME: {r.get('regime_label', 'N/A')}
  📈  Điểm tổng hợp: {r.get('combined_score', 0)}/100
  ⚠️  {r.get('risk_label', 'N/A')}

{'─'*70}
  📈 XU HƯỚNG (Score: {r.get('trend_score', 0)}/100)
{'─'*70}
  Trên MA20: {'✅' if t.get('above_ma20') else '❌'} (MA20 = {t.get('ma20', 'N/A')})
  Trên MA50: {'✅' if t.get('above_ma50') else '❌'} (MA50 = {t.get('ma50', 'N/A')})
  Golden Cross: {'✅ MA20 > MA50' if t.get('golden_cross') else '❌ MA20 < MA50'}
  Cấu trúc HH-HL: {'✅ Đỉnh-đáy tăng' if t.get('higher_highs_higher_lows') else '❌'}
  Thay đổi: 5D={t.get('pct_change_5d', 0):+.2f}% | 10D={t.get('pct_change_10d', 0):+.2f}% | 20D={t.get('pct_change_20d', 0):+.2f}%

{'─'*70}
  💰 DÒNG TIỀN (Score: {r.get('money_flow_score', 0)}/100)
{'─'*70}
  Volume hôm nay: {m.get('vol_ratio_today', 0):.2f}x so với TB 20 phiên
  Volume 5D vs 20D: {m.get('vol_5d_vs_20d', 0):.2f}x
  Vol ngày tăng: {m.get('avg_up_day_vol', 0):,.0f} | Vol ngày giảm: {m.get('avg_down_day_vol', 0):,.0f}
  Tỷ lệ ngày tăng (10 phiên): {m.get('up_day_ratio_10d', 0):.0f}% ({m.get('up_days_count', 0)} tăng / {m.get('down_days_count', 0)} giảm)

{'─'*70}
  📊 BIẾN ĐỘNG
{'─'*70}
  ATR(14): {v.get('atr_14', 0):,.2f} ({v.get('atr_pct', 0):.2f}%)
  Range 20 ngày: {v.get('range_20d_pct', 0):.2f}%
  Vị trí giá: {v.get('pct_of_high20', 0):.1f}% so với đỉnh 20 phiên

{'─'*70}
  🔔 KHUYẾN NGHỊ CHO NHÀ ĐẦU TƯ
{'─'*70}
  {r.get('investor_strategy', '')}

{'─'*70}
  💼 KHUYẾN NGHỊ CHO MÔI GIỚI
{'─'*70}
  {r.get('broker_strategy', '')}
{'='*70}"""
        return report

    def _empty_result(self, index_symbol: str) -> Dict:
        """Trả về kết quả rỗng khi không đủ dữ liệu."""
        return {
            'index_symbol': index_symbol,
            'last_close': 0,
            'last_date': '',
            'regime': 'UNKNOWN',
            'regime_label': '⚪ KHÔNG XÁC ĐỊNH',
            'trend_score': 0,
            'money_flow_score': 0,
            'combined_score': 0,
            'risk_level': 'UNKNOWN',
            'risk_label': '⚪ Không xác định',
            'investor_strategy': 'Không đủ dữ liệu VNINDEX để đánh giá.',
            'broker_strategy': 'Không đủ dữ liệu VNINDEX.',
            'detail': {},
            'raw_data': pd.DataFrame(),
        }


# ============================================================================
# PHẦN 5: NEWS INTEGRATION - Tích hợp tin tức từ vnstock_news
# ============================================================================

class NewsIntegration:
    """
    Tích hợp tin tức liên quan đến cổ phiếu từ vnstock_news.
    Thu thập tin tức từ CafeF, VnExpress, TheSaigonTimes...
    """

    def __init__(self, sources: List[str] = None):
        """
        Parameters:
            sources: Danh sách nguồn tin (VD: ['cafef', 'vnexpress'])
        """
        self.sources = sources or ['cafef', 'vnexpress']
        self.crawler = None
        self._initialize()

    def _initialize(self):
        """Khởi tạo vnstock_news Crawler."""
        try:
            from vnstock_news.core.crawler import Crawler
            from vnstock_news.core.batch import BatchCrawler
            self.crawlers = {}
            self.batch_crawlers = {}
            for src in self.sources:
                try:
                    self.crawlers[src] = Crawler(site_name=src)
                    self.batch_crawlers[src] = BatchCrawler(site_name=src, request_delay=0.5)
                except Exception as e:
                    logger.warning(f"Cannot init vnstock_news for {src}: {e}")
            logger.info(f"NewsIntegration initialized with sources: {list(self.crawlers.keys())}")
        except ImportError:
            logger.warning(
                "vnstock_news chưa cài đặt. Chạy: pip install vnstock_news\n"
                "Tin tức sẽ bị bỏ qua."
            )
            self.crawlers = {}
            self.batch_crawlers = {}

    def fetch_latest_news(self, limit: int = 20) -> pd.DataFrame:
        """
        Lấy tin tức mới nhất từ tất cả nguồn.

        Returns:
            DataFrame: title, url, publish_time, source
        """
        all_articles = []

        for src, crawler in self.crawlers.items():
            try:
                articles = crawler.get_articles(limit=limit)
                for art in articles:
                    art['source'] = src
                    all_articles.append(art)
            except Exception as e:
                logger.warning(f"Error fetching news from {src}: {e}")

        if not all_articles:
            return pd.DataFrame()

        return pd.DataFrame(all_articles)

    def search_news_for_symbol(self, symbol: str, news_df: pd.DataFrame) -> List[Dict]:
        """
        Tìm tin tức liên quan đến 1 mã cổ phiếu.

        Logic: Kiểm tra tên mã hoặc từ khóa liên quan trong tiêu đề tin.
        """
        if news_df.empty:
            return []

        symbol_upper = symbol.upper()
        matched = []

        for _, row in news_df.iterrows():
            title = str(row.get('title', '')).upper()
            # Tìm theo mã hoặc tên công ty phổ biến
            if symbol_upper in title:
                matched.append({
                    'title': row.get('title', ''),
                    'url': row.get('url', row.get('link', '')),
                    'source': row.get('source', ''),
                    'publish_time': row.get('publish_time', row.get('pubDate', ''))
                })

        return matched

    def fetch_article_content(self, url: str, source: str = 'cafef') -> Optional[Dict]:
        """Lấy nội dung chi tiết 1 bài viết."""
        if source not in self.crawlers:
            return None
        try:
            return self.crawlers[source].get_article_details(url)
        except Exception as e:
            logger.warning(f"Error fetching article detail: {e}")
            return None


# ============================================================================
# PHẦN 6: GOOGLE SHEETS EXPORT
# ============================================================================

class GoogleSheetsExporter:
    """
    Xuất kết quả bộ lọc ra Google Sheets.

    Yêu cầu:
    - File credentials JSON từ Google Cloud Console (Service Account)
    - Share sheet với email của Service Account
    """

    def __init__(self, credentials_file: str = 'credentials.json'):
        """
        Parameters:
            credentials_file: Đường dẫn file JSON credentials
        """
        self.credentials_file = credentials_file
        self.client = None

    def authenticate(self):
        """Xác thực với Google Sheets API."""
        try:
            import gspread
            from oauth2client.service_account import ServiceAccountCredentials

            scope = [
                'https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive'
            ]
            creds = ServiceAccountCredentials.from_json_keyfile_name(
                self.credentials_file, scope
            )
            self.client = gspread.authorize(creds)
            logger.info("Google Sheets authentication successful")
        except FileNotFoundError:
            logger.error(
                f"Không tìm thấy file credentials: {self.credentials_file}\n"
                "Hướng dẫn tạo:\n"
                "1. Vào Google Cloud Console → APIs & Services → Credentials\n"
                "2. Tạo Service Account → Download JSON key\n"
                "3. Enable Google Sheets API & Google Drive API\n"
                "4. Share Google Sheet với email của Service Account"
            )
            raise
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise

    def export_to_sheet(
        self,
        results_df: pd.DataFrame,
        spreadsheet_name: str = 'Stock_Screener_Results',
        worksheet_name: str = None
    ):
        """
        Xuất DataFrame kết quả ra Google Sheet.

        Parameters:
            results_df: DataFrame kết quả screening
            spreadsheet_name: Tên Google Spreadsheet
            worksheet_name: Tên sheet (mặc định: ngày hiện tại)
        """
        if self.client is None:
            self.authenticate()

        if worksheet_name is None:
            worksheet_name = datetime.now().strftime('Scan_%Y%m%d_%H%M')

        try:
            # Mở hoặc tạo spreadsheet
            try:
                spreadsheet = self.client.open(spreadsheet_name)
            except Exception:
                spreadsheet = self.client.create(spreadsheet_name)
                logger.info(f"Created new spreadsheet: {spreadsheet_name}")

            # Tạo worksheet mới
            try:
                worksheet = spreadsheet.add_worksheet(
                    title=worksheet_name,
                    rows=len(results_df) + 5,
                    cols=len(results_df.columns) + 2
                )
            except Exception:
                worksheet = spreadsheet.worksheet(worksheet_name)
                worksheet.clear()

            # Chuẩn bị dữ liệu
            df_export = results_df.copy()

            # Chuyển NaN → ''
            df_export = df_export.fillna('')

            # Convert numeric columns properly
            for col in df_export.columns:
                if df_export[col].dtype in ['float64', 'float32']:
                    df_export[col] = df_export[col].round(4)

            # Ghi header + data
            header = df_export.columns.tolist()
            data = df_export.values.tolist()

            all_values = [header] + data
            worksheet.update(range_name='A1', values=all_values)

            # Format header
            worksheet.format('1:1', {
                'textFormat': {'bold': True, 'fontSize': 11},
                'backgroundColor': {'red': 0.2, 'green': 0.4, 'blue': 0.7},
                'horizontalAlignment': 'CENTER',
                'textFormat': {'bold': True, 'foregroundColor': {'red': 1, 'green': 1, 'blue': 1}}
            })

            # Freeze header row
            worksheet.freeze(rows=1)

            logger.info(
                f"Exported {len(results_df)} rows to "
                f"'{spreadsheet_name}' → '{worksheet_name}'"
            )

            return spreadsheet.url

        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise

    def export_to_excel_fallback(
        self,
        results_df: pd.DataFrame,
        filename: str = None
    ) -> str:
        """
        Xuất ra file Excel nếu không có Google Sheets credentials.
        File Excel này có thể import vào Google Sheets thủ công.
        """
        if filename is None:
            filename = f"quant {datetime.now().strftime('%d-%m-%y')}.xlsx"

        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name='Kết quả lọc', index=False)

            # Auto-adjust column width
            ws = writer.sheets['Kết quả lọc']
            for col_idx, col in enumerate(results_df.columns, 1):
                max_len = max(
                    results_df[col].astype(str).apply(len).max(),
                    len(str(col))
                ) + 2
                ws.column_dimensions[chr(64 + col_idx) if col_idx <= 26 else 'A'].width = min(max_len, 50)

        logger.info(f"Exported to Excel: {filename}")
        return filename


# ============================================================================
# PHẦN 7: MAIN ORCHESTRATOR - Điều phối toàn bộ pipeline
# ============================================================================

class StockScreener:
    """
    Orchestrator chính - Điều phối toàn bộ pipeline lọc cổ phiếu.

    Pipeline: Data → Features → Screening → Analysis → News → Export
    """

    def __init__(
        self,
        source: str = 'KBS',
        enable_news: bool = True,
        news_sources: List[str] = None,
        credentials_file: str = 'credentials.json'
    ):
        self.data_provider = DataProvider(source=source)
        self.feature_engine = FeatureEngine()
        self.screening_engine = ScreeningEngine()
        self.analysis_engine = AnalysisEngine()
        self.market_analyzer = MarketAnalyzer(self.data_provider)

        # News integration (tùy chọn)
        self.enable_news = enable_news
        if enable_news:
            self.news = NewsIntegration(sources=news_sources)
        else:
            self.news = None

        # Google Sheets export
        self.exporter = GoogleSheetsExporter(credentials_file=credentials_file)

        logger.info("StockScreener initialized")

    def scan(
        self,
        symbols: List[str] = None,
        group: str = 'VN30',
        lookback_days: int = 90,
        delay: float = 0.3,
        index_symbol: str = 'VNINDEX'
    ) -> pd.DataFrame:
        """
        Chạy toàn bộ pipeline lọc cổ phiếu.

        Parameters:
            symbols: Danh sách mã cụ thể (nếu None → dùng group)
            group: Nhóm cổ phiếu ('VN30', 'HOSE', 'HNX', 'UPCOM', 'ALL' = toàn bộ HOSE+HNX+UPCOM, ...)
            lookback_days: Số ngày dữ liệu lịch sử
            delay: Thời gian chờ giữa các request (giây)
            index_symbol: Chỉ số so sánh (VNINDEX, VN30, ...)

        Returns:
            DataFrame kết quả tổng hợp với scoring + nhận xét
        """
        # ---- STEP 1: Lấy danh sách mã ----
        if symbols is None:
            symbols = self.data_provider.get_stock_list(group=group)
            logger.info(f"Stock list loaded: {len(symbols)} symbols from {group}")
        else:
            logger.info(f"Using custom symbol list: {len(symbols)} symbols")

        # ---- STEP 2: Lấy dữ liệu VNINDEX ----
        logger.info(f"Fetching index data: {index_symbol}...")
        index_df = self.data_provider.get_index_data(
            index_symbol=index_symbol, days=lookback_days
        )

        # ---- STEP 2B: Phân tích sức khỏe VNINDEX ----
        logger.info(f"Analyzing market health: {index_symbol}...")
        market_result = self.market_analyzer.analyze_index(
            index_symbol=index_symbol, days=lookback_days
        )

        # In báo cáo thị trường
        market_report = self.market_analyzer.format_market_report(market_result)
        print(market_report)

        # Lấy ngưỡng điều chỉnh theo regime
        adjustments = self.market_analyzer.get_screening_adjustments(market_result)
        logger.info(f"Regime adjustments: {adjustments['note']}")

        # ---- STEP 3: Lấy tin tức (nếu bật) ----
        news_df = pd.DataFrame()
        if self.enable_news and self.news:
            logger.info("Fetching latest news...")
            news_df = self.news.fetch_latest_news(limit=50)
            logger.info(f"Fetched {len(news_df)} news articles")

        # ---- STEP 4: Lọc từng mã ----
        results = []
        total = len(symbols)

        for i, symbol in enumerate(symbols):
            logger.info(f"[{i+1}/{total}] Scanning {symbol}...")

            # 4.1: Lấy dữ liệu OHLCV
            df = self.data_provider.get_ohlcv(symbol, days=lookback_days)
            if df is None or len(df) < 20:
                logger.warning(f"  → {symbol}: Không đủ dữ liệu, bỏ qua.")
                continue

            # 4.1b: Lọc thanh khoản tối thiểu (AvgVol20 >= 100,000 cp/phiên)
            avg_vol_20 = df['volume'].tail(20).mean()
            if avg_vol_20 < 100_000:
                logger.info(f"  → {symbol}: Thanh khoản thấp (AvgVol20={avg_vol_20:,.0f}), bỏ qua.")
                continue

            # 4.2: Tính features
            df = self.feature_engine.compute_features(df)

            # 4.3: Tính Relative Strength vs Index
            if index_df is not None:
                df = self.feature_engine.compute_relative_strength(df, index_df)

            # 4.4: Áp dụng 3 bộ tiêu chí
            strong_result = self.screening_engine.screen_strong(df)
            accum_result  = self.screening_engine.screen_accumulation(df)
            dist_result   = self.screening_engine.screen_distribution(df)

            # 4.5: Tổng hợp nhận xét
            assessment = self.analysis_engine.generate_assessment(
                symbol, df, strong_result, accum_result, dist_result
            )

            # 4.6: Tóm tắt giá
            price_summary = self.analysis_engine.get_price_summary(df)

            # 4.7: Tin tức liên quan
            related_news = []
            if self.enable_news and self.news and not news_df.empty:
                related_news = self.news.search_news_for_symbol(symbol, news_df)

            news_titles = " | ".join([n['title'] for n in related_news[:3]]) if related_news else ""

            # ---- Tổng hợp kết quả ----
            row = {
                'Mã CK': symbol,
                'Giá đóng cửa': price_summary.get('close', 0),
                'Thay đổi 1D (%)': round((price_summary.get('pct_change_1d', 0) or 0) * 100, 2),
                'Thay đổi 5D (%)': round((price_summary.get('pct_change_5d', 0) or 0) * 100, 2),
                'Thay đổi 20D (%)': round((price_summary.get('pct_change_20d', 0) or 0) * 100, 2),
                'Vol/AvgVol20': round(price_summary.get('vol_ratio_20', 0) or 0, 2),
                'Vị trí vs High20 (%)': round((price_summary.get('pct_of_high20', 0) or 0) * 100, 1),

                # Scores
                'Điểm MẠNH (%)': strong_result['score'],
                'Đạt/Tổng (Mạnh)': f"{strong_result['met_count']}/{strong_result['total_criteria']}",
                'Điểm TÍCH LŨY (%)': accum_result['score'],
                'Đạt/Tổng (TL)': f"{accum_result['met_count']}/{accum_result['total_criteria']}",
                'Điểm PHÂN PHỐI (%)': dist_result['score'],
                'Đạt/Tổng (PP)': f"{dist_result['met_count']}/{dist_result['total_criteria']}",

                # Overall assessment
                'Tín hiệu': assessment['overall_signal'],
                'Dòng tiền': assessment['money_flow'],
                'Xu hướng': assessment['trend'],
                'Sức mạnh': assessment['strength'],

                # Market context (VNINDEX)
                'VNI Regime': market_result.get('regime_label', 'N/A'),
                'VNI Score': market_result.get('combined_score', 0),
                'VNI Risk': market_result.get('risk_label', 'N/A'),
                'VNINDEX': (
                    f"{market_result.get('last_close', 0):,.2f} | "
                    f"{market_result.get('detail', {}).get('trend', {}).get('pct_change_5d', 0):+.2f}% (5D) | "
                    f"{market_result.get('detail', {}).get('trend', {}).get('pct_change_20d', 0):+.2f}% (20D) | "
                    f"{'Trên' if market_result.get('detail', {}).get('trend', {}).get('above_ma20') else 'Dưới'} MA20 | "
                    f"ATR: {market_result.get('detail', {}).get('volatility', {}).get('atr_pct', 0):.1f}%"
                ),

                # Regime-adjusted signal
                'Tín hiệu (Adj)': self._apply_regime_adjustment(
                    strong_result['score'], accum_result['score'],
                    dist_result['score'], adjustments
                ),

                # Notes
                'Ghi chú NĐT': assessment['investor_note'],
                'Ghi chú Môi giới': assessment['broker_note'],

                # News
                'Tin tức liên quan': news_titles
            }

            results.append(row)
            time.sleep(delay)

        # ---- STEP 5: Tổng hợp & sắp xếp ----
        if not results:
            logger.warning("Không có kết quả nào!")
            return pd.DataFrame()

        results_df = pd.DataFrame(results)

        # Sắp xếp: Mạnh nhất → Tích lũy → Phân phối
        results_df = results_df.sort_values(
            by=['Điểm MẠNH (%)', 'Điểm TÍCH LŨY (%)', 'Điểm PHÂN PHỐI (%)'],
            ascending=[False, False, True]
        ).reset_index(drop=True)

        # Lưu toàn bộ kết quả
        self._full_results = results_df.copy()

        # Lọc Top 70 cổ phiếu tốt nhất (điểm MẠNH hoặc TÍCH LŨY cao nhất)
        top_n = min(70, len(results_df))
        results_df = results_df.head(top_n).reset_index(drop=True)

        logger.info(f"Scan complete: {len(self._full_results)} stocks analyzed → Top {top_n} selected")
        return results_df

    def export(
        self,
        results_df: pd.DataFrame,
        spreadsheet_name: str = 'Bo_Loc_Co_Phieu',
        use_google_sheets: bool = True
    ) -> str:
        """
        Xuất kết quả.

        Parameters:
            results_df: DataFrame kết quả từ scan()
            spreadsheet_name: Tên Google Sheet
            use_google_sheets: True = Google Sheets, False = Excel

        Returns:
            str: URL Google Sheet hoặc đường dẫn file Excel
        """
        if use_google_sheets:
            try:
                url = self.exporter.export_to_sheet(
                    results_df, spreadsheet_name=spreadsheet_name
                )
                return url
            except Exception as e:
                logger.warning(f"Google Sheets export failed: {e}")
                logger.info("Falling back to Excel export...")
                return self.exporter.export_to_excel_fallback(results_df)
        else:
            return self.exporter.export_to_excel_fallback(results_df)

    def quick_scan_vn30(self) -> pd.DataFrame:
        """Scan nhanh VN30 - Dùng cho demo hoặc scan hàng ngày."""
        return self.scan(group='VN30', lookback_days=60, delay=0.3)

    def scan_custom(self, symbols: List[str]) -> pd.DataFrame:
        """Scan danh sách mã tùy chỉnh."""
        return self.scan(symbols=symbols, lookback_days=60, delay=0.3)

    def scan_market(self, index_symbol: str = 'VNINDEX', days: int = 90) -> Dict:
        """
        Chỉ phân tích VNINDEX (không scan cổ phiếu).
        Dùng khi chỉ cần xem sức khỏe thị trường.
        """
        result = self.market_analyzer.analyze_index(index_symbol, days)
        report = self.market_analyzer.format_market_report(result)
        print(report)
        return result

    @staticmethod
    def _apply_regime_adjustment(
        s_score: float, a_score: float, d_score: float,
        adjustments: Dict
    ) -> str:
        """
        Áp dụng ngưỡng điều chỉnh theo regime thị trường.

        Trong bull market: dễ gọi là "mạnh", khó gọi là "phân phối"
        Trong bear market: khó gọi là "mạnh", dễ gọi là "phân phối"
        """
        s_thresh = adjustments.get('strong_threshold', 57.1)
        a_thresh = adjustments.get('accum_threshold', 57.1)
        d_thresh = adjustments.get('dist_threshold', 57.1)

        if s_score >= s_thresh and d_score < d_thresh:
            return '🟢 MẠNH (Adj)'
        elif d_score >= d_thresh:
            return '🔴 PHÂN PHỐI (Adj)'
        elif a_score >= a_thresh and d_score < d_thresh:
            return '🟡 TÍCH LŨY (Adj)'
        elif s_score >= (s_thresh * 0.75):
            return '🟢 THIÊN MẠNH (Adj)'
        elif a_score >= (a_thresh * 0.75):
            return '🟡 THIÊN TL (Adj)'
        else:
            return '⚪ TRUNG LẬP (Adj)'


# ============================================================================
# PHẦN 8: CHẠY CHƯƠNG TRÌNH
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bộ lọc cổ phiếu thông minh')
    parser.add_argument('--mode', choices=['1', '2', '3'],
                        help='1: quét toàn thị trường, 2: chỉ VNINDEX, 3: danh sách tùy chỉnh')
    parser.add_argument('--symbols',
                        help='Danh sách mã cách nhau bằng dấu phẩy, dùng cùng --mode 3')
    args = parser.parse_args()

    print("=" * 70)
    print("  BỘ LỌC CỔ PHIẾU THÔNG MINH - SMART STOCK SCREENER")
    print("  Tích hợp phân tích VNINDEX + Regime Detection")
    print("=" * 70)

    # --- CẤU HÌNH ---
    CONFIG = {
        'group': 'ALL',                  # Quét toàn bộ 3 sàn: HOSE + HNX + UPCOM (~1600+ mã)
        'lookback_days': 90,             # Số ngày dữ liệu lịch sử
        'delay': 0.3,                    # Delay giữa các request
        'enable_news': True,             # Bật/tắt tin tức
        'news_sources': ['cafef'],       # Nguồn tin
        # File Excel `quant ...xlsx` là đầu vào của quant.py.
        'use_google_sheets': False,
        'credentials_file': 'credentials.json',
        'spreadsheet_name': 'Bộ Lọc Cổ Phiếu',   # Tên GG Sheet đã tạo
        'index_symbol': 'VNINDEX',       # Chỉ số thị trường
    }

    # --- KHỞI TẠO ---
    screener = StockScreener(
        source='KBS',
        enable_news=CONFIG['enable_news'],
        news_sources=CONFIG['news_sources'],
        credentials_file=CONFIG['credentials_file']
    )

    # --- CHỌN CHẾ ĐỘ ---
    print("\n  Chọn chế độ:")
    print("  1. Scan đầy đủ (VNINDEX + Cổ phiếu)")
    print("  2. Chỉ phân tích VNINDEX")
    print("  3. Scan danh sách mã tùy chỉnh")

    mode = args.mode
    if mode is None:
        try:
            mode = input("\n  Nhập (1/2/3, Enter = 1): ").strip() or '1'
        except EOFError:
            mode = '1'
            logger.info("Không có bàn phím tương tác; tự động chọn chế độ 1.")

    if mode == '2':
        # Chỉ phân tích VNINDEX
        market_result = screener.scan_market(
            index_symbol=CONFIG['index_symbol'],
            days=CONFIG['lookback_days']
        )
    elif mode == '3':
        # Scan tùy chỉnh
        symbols_input = args.symbols or ''
        if not symbols_input:
            try:
                symbols_input = input("  Nhập mã (cách nhau bởi dấu phẩy, VD: FPT,VNM,HPG): ").strip()
            except EOFError:
                symbols_input = ''
        symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
        if symbols:
            results = screener.scan(
                symbols=symbols,
                lookback_days=CONFIG['lookback_days'],
                delay=CONFIG['delay'],
                index_symbol=CONFIG['index_symbol']
            )
        else:
            print("Không có mã nào được nhập.")
            results = pd.DataFrame()
    else:
        # Scan đầy đủ
        results = screener.scan(
            group=CONFIG['group'],
            lookback_days=CONFIG['lookback_days'],
            delay=CONFIG['delay'],
            index_symbol=CONFIG['index_symbol']
        )

    if mode != '2' and 'results' in dir() and not results.empty:
        # --- HIỂN THỊ KẾT QUẢ ---
        print("\n" + "=" * 70)
        print(f"  TOP 70 CỔ PHIẾU TỐT NHẤT - TOÀN THỊ TRƯỜNG (HOSE + HNX + UPCOM)")
        print(f"  (Lọc từ {len(screener._full_results)} mã có thanh khoản >= 100K cp/phiên)")
        print(f"  Có điều chỉnh theo VNINDEX Regime")
        print("=" * 70)

        # VNINDEX summary
        if 'VNINDEX' in results.columns:
            print(f"\n  📊 VNINDEX: {results['VNINDEX'].iloc[0]}")
            print(f"  🏷️  {results['VNI Regime'].iloc[0]} | Score: {results['VNI Score'].iloc[0]} | {results['VNI Risk'].iloc[0]}")
            print(f"{'─'*70}")

        # Top cổ phiếu mạnh (dùng tín hiệu đã điều chỉnh)
        strong = results[results['Tín hiệu (Adj)'].str.contains('MẠNH', na=False)]
        if not strong.empty:
            print(f"\n🟢 CỔ PHIẾU MẠNH ({len(strong)} mã):")
            for _, r in strong.iterrows():
                print(f"  {r['Mã CK']:>6} | Giá: {r['Giá đóng cửa']:>10,.0f} | "
                      f"Mạnh: {r['Điểm MẠNH (%)']:>5.1f}% | {r['Tín hiệu (Adj)']}")
                print(f"         {r['Dòng tiền'][:60]}")

        # Cổ phiếu tích lũy
        accum = results[results['Tín hiệu (Adj)'].str.contains('TÍCH LŨY|TL', na=False)]
        if not accum.empty:
            print(f"\n🟡 CỔ PHIẾU TÍCH LŨY ({len(accum)} mã):")
            for _, r in accum.iterrows():
                print(f"  {r['Mã CK']:>6} | Giá: {r['Giá đóng cửa']:>10,.0f} | "
                      f"TL: {r['Điểm TÍCH LŨY (%)']:>5.1f}% | {r['Tín hiệu (Adj)']}")

        # Cổ phiếu phân phối
        dist = results[results['Tín hiệu (Adj)'].str.contains('PHÂN PHỐI', na=False)]
        if not dist.empty:
            print(f"\n🔴 CẢNH BÁO PHÂN PHỐI ({len(dist)} mã):")
            for _, r in dist.iterrows():
                print(f"  {r['Mã CK']:>6} | Giá: {r['Giá đóng cửa']:>10,.0f} | "
                      f"PP: {r['Điểm PHÂN PHỐI (%)']:>5.1f}% | {r['Tín hiệu (Adj)']}")

        # Bối cảnh thị trường (nhắc lại)
        print(f"\n{'─'*70}")
        print(f"  📊 BỐI CẢNH: {results['VNI Regime'].iloc[0]} | "
              f"Score: {results['VNI Score'].iloc[0]} | {results['VNI Risk'].iloc[0]}")
        print(f"{'─'*70}")

        # --- XUẤT FILE ---
        output_path = screener.export(
            results,
            spreadsheet_name=CONFIG['spreadsheet_name'],
            use_google_sheets=CONFIG['use_google_sheets']
        )
        print(f"\n📁 Kết quả đã xuất: {output_path}")
    elif mode != '2':
        print("\nKhông có kết quả. Kiểm tra kết nối mạng và danh sách mã.")
