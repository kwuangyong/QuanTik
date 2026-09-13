"""Institutional-style social visuals for the Quant Pipeline V6.

The module consumes the pipeline's report dictionaries and OHLCV frames.  It
never recalculates Monte Carlo, HMM, GARCH, or the pipeline Quant Score.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd

log = logging.getLogger("quant_visuals")

_FONT_DIR = Path(__file__).resolve().parent / "assets" / "fonts"
for _font_file in ("IBMPlexMono-Regular.ttf", "IBMPlexMono-SemiBold.ttf"):
    _font_path = _FONT_DIR / _font_file
    if _font_path.exists():
        font_manager.fontManager.addfont(str(_font_path))
TERMINAL_FONT = "IBM Plex Mono" if (_FONT_DIR / "IBMPlexMono-Regular.ttf").exists() else "DejaVu Sans Mono"

QUANT_THEME = {
    "background": "#F5F7FA", "panel": "#FFFFFF", "text": "#172033",
    "muted": "#687386", "positive": "#087F5B", "negative": "#C92A2A",
    "neutral": "#D18B00", "primary": "#174EA6", "grid": "#DCE2EA",
    "font": "DejaVu Sans", "figure_size": (16, 9), "dpi": 120,
}

TERMINAL_THEME = {
    "background": "#0A0A0C", "panel": "#0A0A0C", "panel_alt": "#111114",
    "text": "#F1F1F1", "muted": "#00D9FF", "positive": "#00E676",
    "negative": "#FF3B3B", "neutral": "#FF9F1C", "primary": "#00D9FF",
    "grid": "#3D3D42", "font": TERMINAL_FONT, "figure_size": (16, 9), "dpi": 140,
}

# New factor normalization is isolated here. Inputs reuse existing pipeline
# scores; equal weights avoid silently overriding AdaptiveScorer's logic.
FACTOR_WEIGHTS = {
    "Trend": {"trend_er": 0.60, "trend_direction": 0.40},
    "Momentum": {"momentum_score": 1.0},
    "Dòng tiền": {"cmf": 1.0},
    "Sức mạnh tương đối": {"relative_strength": 1.0},
    "Chất lượng rủi ro": {"sharpe": 0.35, "sortino": 0.25, "max_dd": 0.25, "volatility": 0.15},
    "Dự báo": {"expected_return": 0.50, "prob_up": 0.25, "agreement": 0.25},
}


def _num(value: Any) -> float | None:
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _pct(value: Any, decimals: int = 1) -> str:
    value = _num(value)
    if value is None:
        return "N/A"
    return "—" if value is None else f"{value:+.{decimals}f}%"


def _percent(value: Any, decimals: int = 1) -> str:
    value = _num(value)
    if value is None:
        return "N/A"
    return "—" if value is None else f"{value:.{decimals}f}%"


def _price(value: Any) -> str:
    value = _num(value)
    if value is None:
        return "N/A"
    return "—" if value is None else f"{value:,.0f}"


def _plain_label(value: Any) -> str:
    s = str(value or "N/A")
    for token in ("🟢", "🔴", "🟡", "📈", "📉", "↔", "⭐", "️"):
        s = s.replace(token, "")
    return " ".join(s.split()).strip()


def _first_passage_prob(paths: np.ndarray | None, upper: float | None, lower: float | None) -> float | None:
    """Probability that the upper barrier is reached before the lower barrier."""
    if paths is None or upper is None or lower is None or upper <= lower:
        return None
    hit_up, hit_down = paths >= upper, paths <= lower
    up_any, down_any = hit_up.any(axis=1), hit_down.any(axis=1)
    up_first = np.where(up_any, hit_up.argmax(axis=1), paths.shape[1] + 1)
    down_first = np.where(down_any, hit_down.argmax(axis=1), paths.shape[1] + 1)
    return float(np.mean(up_first < down_first) * 100)


def _action_color(theme: Mapping[str, Any], action: Any) -> str:
    label = _plain_label(action).upper()
    if any(token in label for token in ("AVOID", "SELL", "TRÁNH", "KHÔNG VÀO")):
        return theme["negative"]
    if any(token in label for token in ("BUY", "MUA", "VÀO NGAY")):
        return theme["positive"]
    return theme["neutral"]


def _valid_price_frame(df: pd.DataFrame | None, minimum: int = 2) -> pd.DataFrame | None:
    if not isinstance(df, pd.DataFrame) or "close" not in df.columns:
        return None
    out = df.copy()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["close"])
    out = out[out["close"] > 0]
    if len(out) < minimum:
        return None
    try: out.index = pd.to_datetime(out.index)
    except (TypeError, ValueError): pass
    return out.sort_index()


class QuantVisualizer:
    def __init__(self, report: Mapping[str, Any], prices: pd.DataFrame, theme: Mapping[str, Any] | None = None):
        self.report = dict(report or {})
        self.ticker = str(self.report.get("symbol", "")).upper().strip()
        if not self.ticker:
            raise ValueError("Report thiếu symbol hợp lệ")
        self.prices = _valid_price_frame(prices)
        if self.prices is None:
            raise ValueError(f"{self.ticker}: thiếu dữ liệu close hợp lệ")
        self.theme = {**QUANT_THEME, **(theme or {})}

    def _figure(self, *args, **kwargs):
        plt.rcParams.update({"font.family": self.theme["font"], "axes.unicode_minus": False})
        fig = plt.figure(figsize=self.theme["figure_size"], dpi=self.theme["dpi"], facecolor=self.theme["background"], *args, **kwargs)
        return fig

    def _style_ax(self, ax, grid=True):
        ax.set_facecolor(self.theme["background"])
        ax.tick_params(colors=self.theme["muted"], labelsize=9)
        for sp in ax.spines.values(): sp.set_visible(False)
        if grid: ax.grid(True, color=self.theme["grid"], linewidth=.7, alpha=.7)

    def _save(self, fig, path: Path, theme: Mapping[str, Any] | None = None, tight: bool = True):
        save_theme = self.theme if theme is None else theme
        path.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {"dpi": save_theme["dpi"], "facecolor": fig.get_facecolor()}
        if tight: save_kwargs["bbox_inches"] = "tight"
        fig.savefig(path, **save_kwargs)
        plt.close(fig)
        # bbox_inches='tight' prevents clipped labels but changes pixel bounds.
        # Letterbox back to the configured 16:9 canvas without stretching.
        try:
            from PIL import Image, ImageOps
            target = tuple(round(v * save_theme["dpi"]) for v in save_theme["figure_size"])
            with Image.open(path) as im:
                fitted = ImageOps.contain(im.convert("RGB"), target)
                canvas = Image.new("RGB", target, save_theme["background"])
                canvas.paste(fitted, ((target[0]-fitted.width)//2, (target[1]-fitted.height)//2))
                canvas.save(path, dpi=(save_theme["dpi"], save_theme["dpi"]), quality=95)
        except ImportError:
            log.warning("Pillow unavailable; %s keeps tight bounding-box dimensions", path)
        return path

    def _paths(self) -> np.ndarray | None:
        paths = self.report.get("fcast", {}).get("_mc_paths")
        if paths is None: return None
        paths = np.asarray(paths, dtype=float)
        if paths.ndim != 2 or min(paths.shape) < 2: return None
        if not np.isfinite(paths).all(): return None
        return paths

    def quant_card(self, path: Path):
        t, r = self.theme, self.report
        score = _num(r.get("rec", {}).get("score"))
        if score is None: raise ValueError("thiếu Quant Score")
        fig = self._figure(); ax = fig.add_axes([0, 0, 1, 1]); ax.set(xlim=(0,1),ylim=(0,1)); ax.axis("off")
        ax.text(.055, .89, "QUANT RESEARCH  •  VIETNAM EQUITIES", color=t["muted"], fontsize=11, weight="bold")
        ax.text(.055, .70, self.ticker, color=t["text"], fontsize=min(64, 72-len(self.ticker)*2), weight="bold")
        company = r.get("company_name") or r.get("company") or r.get("sector", {}).get("name", "")
        if company: ax.text(.058, .64, str(company).upper(), color=t["muted"], fontsize=14)
        ax.text(.058, .49, "ĐIỂM ĐỊNH LƯỢNG", color=t["muted"], fontsize=12, weight="bold")
        ax.text(.055, .31, f"{score:.0f}", color=t["primary"], fontsize=92, weight="bold")
        ax.text(.205, .35, "/ 100", color=t["muted"], fontsize=24)
        action = _plain_label(r.get("action", {}).get("action") or r.get("rec", {}).get("rating"))
        ax.text(.058, .19, action, color=_action_color(t, action), fontsize=20, weight="bold")

        stats, fc, flow, alpha, vol, hmm = r.get("stats",{}), r.get("fcast",{}), r.get("flow",{}), r.get("alpha",{}), r.get("vol",{}), r.get("hmm",{})
        cmf = _num(flow.get("cmf")); rs = _num(alpha.get("cross_sectional",{}).get("rs_20d_pct"))
        metrics = [("LỢI NHUẬN KỲ VỌNG", _pct(fc.get("ensemble_ret_pct"))), ("SHARPE", f"{_num(stats.get('sharpe')):.2f}" if _num(stats.get('sharpe')) is not None else None),
                   ("DÒNG TIỀN", "MẠNH" if cmf is not None and cmf > .1 else "YẾU" if cmf is not None and cmf < -.1 else "TRUNG TÍNH" if cmf is not None else None),
                   ("SỨC MẠNH TƯƠNG ĐỐI", _pct(rs) if rs is not None else None), ("RỦI RO", _plain_label(vol.get("regime")) if vol.get("regime") else None),
                   ("TRẠNG THÁI HMM", _plain_label(hmm.get("current")) if hmm.get("current") else None)]
        metrics = [(a,b) for a,b in metrics if b not in (None,"—")][:6]
        for i, (label, value) in enumerate(metrics):
            x=.46+(i%2)*.25; y=.78-(i//2)*.17
            ax.text(x,y,label,color=t["muted"],fontsize=10,weight="bold")
            ax.text(x,y-.065,value,color=t["text"],fontsize=20,weight="bold")
        levels = r.get("sl",{}); vals = [("ENTRY",levels.get("entry")),("STOP LOSS",levels.get("sl_swing")),("TP1",levels.get("tp1")),("TP2",levels.get("tp2",levels.get("tp_optimal")))]
        for i,(label,val) in enumerate((x for x in vals if _num(x[1]) is not None)):
            x=.43+i*.14; ax.text(x,.17,label,color=t["muted"],fontsize=9,weight="bold"); ax.text(x,.105,_price(val),color=t["text"],fontsize=16,weight="bold")
        ax.plot([.40,.40],[.08,.88],color=t["grid"],lw=1)
        return self._save(fig,path)

    def probability_cone(self, path: Path):
        sims = self._paths()
        if sims is None: raise ValueError("thiếu ma trận Monte Carlo gốc")
        t=self.theme; hist=self.prices.tail(80)["close"]; current=float(hist.iloc[-1]); h=sims.shape[1]
        q10,q25,q50,q75,q90=np.percentile(sims,[10,25,50,75,90],axis=0)
        xh=np.arange(len(hist)); xf=np.arange(len(hist),len(hist)+h)
        fig=self._figure(); ax=fig.add_axes([.07,.14,.68,.72]); self._style_ax(ax)
        ax.plot(xh,hist.values,color=t["text"],lw=1.8,label="Giá lịch sử")
        ax.fill_between(xf,q10,q90,color=t["primary"],alpha=.10,label="P10–P90")
        ax.fill_between(xf,q25,q75,color=t["primary"],alpha=.22,label="P25–P75")
        ax.plot(np.r_[len(hist)-1,xf],np.r_[current,q50],color=t["primary"],lw=2.5,label="Trung vị P50")
        ax.scatter(len(hist)-1,current,s=40,color=t["text"],zorder=4)
        levels=self.report.get("sl",{})
        for key,color,label in [("entry",t["primary"],"Entry"),("sl_swing",t["negative"],"SL"),("tp1",t["positive"],"TP1"),("tp2",t["positive"],"TP2")]:
            v=_num(levels.get(key, levels.get("tp_optimal") if key=="tp2" else None))
            if v is not None: ax.axhline(v,color=color,lw=1,ls="--",alpha=.75); ax.text(len(hist)+h-.2,v,f" {label}",color=color,va="center",fontsize=9)
        ax.set_xticks([0,len(hist)-1,len(hist)+h-1]); ax.set_xticklabels([str(hist.index[0])[:10],"HIỆN TẠI",f"+{h} PHIÊN"]); ax.legend(frameon=False,ncol=3,loc="upper left")
        fig.text(.07,.94,f"{self.ticker} — XÁC SUẤT {h} PHIÊN",fontsize=24,weight="bold",color=t["text"]); fig.text(.07,.90,f"Monte Carlo | {len(sims):,} mô phỏng",fontsize=11,color=t["muted"])
        terminal=sims[:,-1]; summary=[("Lợi nhuận trung vị",(np.median(terminal)/current-1)*100),("P(Lợi nhuận > 0)",np.mean(terminal>current)*100),("P(Lỗ > 3%)",np.mean(terminal<current*.97)*100)]
        tp1=_num(levels.get("tp1")); sl=_num(levels.get("sl_swing"))
        if tp1 is not None: summary.append(("P(Chạm TP1)",np.mean(np.max(sims,axis=1)>=tp1)*100))
        if sl is not None: summary.append(("P(Chạm SL)",np.mean(np.min(sims,axis=1)<=sl)*100))
        for i,(lab,val) in enumerate(summary): fig.text(.79,.73-i*.12,lab,color=t["muted"],fontsize=10); fig.text(.79,.68-i*.12,_pct(val) if i == 0 else _percent(val),color=t["text"],fontsize=17,weight="bold")
        return self._save(fig,path)

    def return_distribution(self, path: Path):
        sims=self._paths()
        if sims is None: raise ValueError("thiếu ma trận Monte Carlo gốc")
        t=self.theme; current=float(self.prices["close"].iloc[-1]); ret=sims[:,-1]/current-1
        mean,median=np.mean(ret),np.median(ret); var=np.percentile(ret,5); cvar=np.mean(ret[ret<=var])
        fig=self._figure(); ax=fig.add_axes([.08,.16,.66,.68]); self._style_ax(ax)
        bins=np.histogram_bin_edges(ret,bins="fd"); ax.hist(ret[ret<0],bins=bins,color=t["negative"],alpha=.65); ax.hist(ret[ret>=0],bins=bins,color=t["positive"],alpha=.65)
        for val,color,label in [(0,t["text"],"0%"),(median,t["primary"],"Trung vị"),(var,t["negative"],"VaR 95%"),(cvar,t["negative"],"CVaR 95%")]: ax.axvline(val,color=color,lw=1.6,ls="--",label=label)
        ax.xaxis.set_major_formatter(lambda x,pos:f"{x*100:.0f}%"); ax.set_ylabel("SỐ MÔ PHỎNG"); ax.legend(frameon=False,ncol=2)
        h=sims.shape[1]; fig.text(.08,.93,f"{self.ticker} — PHÂN PHỐI LỢI NHUẬN",fontsize=24,weight="bold",color=t["text"]); fig.text(.08,.89,f"Monte Carlo terminal return | {h} phiên | {len(ret):,} mô phỏng",fontsize=11,color=t["muted"])
        expected=_num(self.report.get("fcast",{}).get("ensemble_ret_pct")); items=[("Lợi nhuận kỳ vọng",expected),("Trung vị",median*100),("Xác suất > 0",np.mean(ret>0)*100),("VaR 95%",var*100),("CVaR 95%",cvar*100)]
        for i,(lab,val) in enumerate(items): fig.text(.79,.73-i*.115,lab,color=t["muted"],fontsize=10); fig.text(.79,.685-i*.115,_percent(val) if lab == "Xác suất > 0" else _pct(val),color=t["text"],fontsize=17,weight="bold")
        return self._save(fig,path)

    def regime_chart(self, path: Path):
        hmm=self.report.get("hmm",{}); labels=hmm.get("_state_labels"); dates=hmm.get("_state_dates")
        if not labels or not dates: raise ValueError("thiếu chuỗi state HMM theo thời gian")
        t=self.theme; s=pd.Series(labels,index=pd.to_datetime(dates)); close=self.prices["close"].reindex(s.index).dropna(); s=s.reindex(close.index)
        if close.empty: raise ValueError("state HMM không khớp lịch sử giá")
        fig=self._figure(); ax=fig.add_axes([.07,.14,.86,.72]); self._style_ax(ax); ax.plot(close.index,close.values,color=t["text"],lw=1.8,zorder=3)
        colors={"BULL":t["positive"],"BEAR":t["negative"],"SIDEWAY":t["neutral"]}
        start=0
        for i in range(1,len(s)+1):
            if i==len(s) or s.iloc[i]!=s.iloc[start]: ax.axvspan(s.index[start],s.index[i-1],color=colors.get(s.iloc[start],t["muted"]),alpha=.10,lw=0); start=i
        ax.xaxis.set_major_locator(mdates.AutoDateLocator()); ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        current=_plain_label(hmm.get("current")); fig.text(.07,.93,f"{self.ticker} — HMM MARKET REGIME",fontsize=24,weight="bold",color=t["text"]); fig.text(.73,.79,"TRẠNG THÁI HIỆN TẠI",fontsize=10,color=t["muted"],weight="bold"); fig.text(.73,.72,current,fontsize=25,color=colors.get(current,t["text"]),weight="bold")
        prob=_num(hmm.get("prob_pct"));
        if prob is not None: fig.text(.73,.67,f"Xác suất regime: {prob:.1f}%",fontsize=11,color=t["muted"])
        return self._save(fig,path)

    def risk_profile(self, path: Path):
        close=self.prices["close"]; dd=close/close.cummax()-1
        if len(dd)<20: raise ValueError("cần tối thiểu 20 phiên giá")
        t=self.theme; fig=self._figure(); ax=fig.add_axes([.07,.16,.66,.68]); self._style_ax(ax); ax.plot(dd.index,dd*100,color=t["negative"],lw=1.7); ax.fill_between(dd.index,dd*100,0,color=t["negative"],alpha=.13); ax.scatter(dd.idxmin(),dd.min()*100,color=t["negative"],s=45,zorder=4); ax.annotate(f"Max DD {dd.min()*100:.1f}%",(dd.idxmin(),dd.min()*100),xytext=(8,-18),textcoords="offset points",color=t["negative"],weight="bold"); ax.set_ylabel("DRAWDOWN (%)")
        fig.text(.07,.93,f"{self.ticker} — HỒ SƠ RỦI RO",fontsize=24,weight="bold",color=t["text"]); stats=self.report.get("stats",{}); cs=self.report.get("alpha",{}).get("cross_sectional",{}); vals=[("Max Drawdown",stats.get("max_dd_pct",dd.min()*100),"pct"),("Volatility",stats.get("ann_vol_pct"),"pct"),("VaR 95%",stats.get("VaR_95"),"pct"),("CVaR 95%",stats.get("CVaR_95"),"pct"),("Beta",cs.get("beta"),"num")]
        vals=[x for x in vals if _num(x[1]) is not None]
        for i,(lab,val,kind) in enumerate(vals): fig.text(.78,.74-i*.115,lab,color=t["muted"],fontsize=10); fig.text(.78,.695-i*.115,(_percent(val) if lab == "Volatility" else _pct(val)) if kind=="pct" else f"{float(val):.2f}",color=t["text"],fontsize=17,weight="bold")
        return self._save(fig,path)

    @staticmethod
    def _scale(x, low, high, inverse=False):
        x=_num(x)
        if x is None: return None
        z=float(np.clip((x-low)/(high-low),0,1)); return 100*(1-z if inverse else z)

    def _factor_scores(self):
        r=self.report; tr=r.get("trend",{}); al=r.get("alpha",{}); st=r.get("stats",{}); fc=r.get("fcast",{}); flow=r.get("flow",{}); cs=al.get("cross_sectional",{})
        inputs={
            "trend_er":self._scale(tr.get("er"),0,1), "trend_direction":self._scale(tr.get("slope_pct"),-10,10),
            "momentum_score":self._scale(al.get("momentum",{}).get("score"),-1,1), "cmf":self._scale(flow.get("cmf"),-.35,.35),
            "relative_strength":self._scale(cs.get("rs_20d_pct"),-15,15), "sharpe":self._scale(st.get("sharpe"),-1,2.5),
            "sortino":self._scale(st.get("sortino"),-1,3.5), "max_dd":self._scale(st.get("max_dd_pct"),-40,0),
            "volatility":self._scale(st.get("ann_vol_pct"),15,65,inverse=True), "expected_return":self._scale(fc.get("ensemble_ret_pct"),-10,10),
            "prob_up":self._scale(fc.get("mc",{}).get("prob_up"),20,80), "agreement":self._scale(fc.get("agreement_pct"),0,100),
        }
        out={}
        for factor,weights in FACTOR_WEIGHTS.items():
            present=[(inputs.get(k),w) for k,w in weights.items() if inputs.get(k) is not None]
            if present: out[factor]=sum(v*w for v,w in present)/sum(w for _,w in present)
        return out

    def radar(self, path: Path):
        factors=self._factor_scores()
        if len(factors)<3: raise ValueError("không đủ tối thiểu 3 factor để chuẩn hóa radar")
        t=self.theme; labels=list(factors); vals=np.array(list(factors.values())); angles=np.linspace(0,2*np.pi,len(vals),endpoint=False); angles=np.r_[angles,angles[0]]; vals=np.r_[vals,vals[0]]
        fig=self._figure(); ax=fig.add_axes([.12,.12,.62,.72],projection="polar"); ax.set_facecolor(t["background"]); ax.plot(angles,vals,color=t["primary"],lw=2.5); ax.fill(angles,vals,color=t["primary"],alpha=.17); ax.set_ylim(0,100); ax.set_yticks([25,50,75,100]); ax.set_yticklabels(["25","50","75","100"],color=t["muted"],fontsize=8); ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels,color=t["text"],fontsize=10); ax.grid(color=t["grid"],lw=.8); ax.spines["polar"].set_visible(False)
        score=_num(self.report.get("rec",{}).get("score")); fig.text(.07,.93,f"{self.ticker} — HỒ SƠ NHÂN TỐ",fontsize=24,weight="bold",color=t["text"]); fig.text(.78,.62,"ĐIỂM ĐỊNH LƯỢNG",fontsize=10,color=t["muted"],weight="bold"); fig.text(.78,.50,f"{score:.0f}" if score is not None else "—",fontsize=62,color=t["primary"],weight="bold"); fig.text(.90,.51,"/ 100",fontsize=16,color=t["muted"]); fig.text(.78,.39,"Các factor đã chuẩn hóa 0–100",fontsize=10,color=t["muted"])
        return self._save(fig,path)

    def _decision_metrics(self):
        r, paths = self.report, self._paths()
        fc, sl, stats = r.get("fcast", {}), r.get("sl", {}), r.get("stats", {})
        entry, stop = _num(sl.get("entry")), _num(sl.get("sl_swing"))
        tp1 = _num(sl.get("tp1")); tp2 = _num(sl.get("tp2", sl.get("tp_optimal")))
        risk = entry - stop if entry is not None and stop is not None else None
        upside1 = (tp1 / entry - 1) * 100 if entry and tp1 else None
        upside2 = (tp2 / entry - 1) * 100 if entry and tp2 else None
        downside = (stop / entry - 1) * 100 if entry and stop else None
        rr = (tp2 - entry) / risk if risk and tp2 else None
        p_tp_before_sl = _first_passage_prob(paths, tp1, stop)
        terminal = paths[:, -1] if paths is not None else None
        current = float(self.prices["close"].iloc[-1])
        mc_median = (np.median(terminal) / current - 1) * 100 if terminal is not None else None
        return {
            "entry": entry, "stop": stop, "tp1": tp1, "tp2": tp2,
            "upside1": upside1, "upside2": upside2, "downside": downside, "rr": rr,
            "p_tp_before_sl": p_tp_before_sl, "mc_median": mc_median,
            "ensemble": _num(fc.get("ensemble_ret_pct")), "prob_up": _num(fc.get("mc", {}).get("prob_up")),
            "agreement": _num(fc.get("agreement_pct", fc.get("confidence"))),
            "horizon": int(fc.get("horizon", paths.shape[1] if paths is not None else 0) or 0),
            "simulations": len(paths) if paths is not None else 0,
            "ev_r": _num(sl.get("ev_r_per_trade")), "pwin_cal": _num(sl.get("p_win_blended")),
            "max_dd": _num(stats.get("max_dd_pct")), "vol": _num(stats.get("ann_vol_pct")),
            "var": _num(stats.get("VaR_95")), "cvar": _num(stats.get("CVaR_95")),
        }

    def _terminal_dashboard_legacy(self, path: Path):
        """Dense, audit-friendly terminal view for internal quant analysis."""
        t = dict(TERMINAL_THEME)
        r, m, close, paths = self.report, self._decision_metrics(), self.prices["close"].tail(120), self._paths()
        plt.rcParams.update({"font.family": t["font"], "axes.unicode_minus": False})
        fig = plt.figure(figsize=t["figure_size"], dpi=t["dpi"], facecolor=t["background"])
        gs = fig.add_gridspec(20, 24, left=.025, right=.985, top=.925, bottom=.035, hspace=.72, wspace=.55)

        def panel(spec, title):
            ax = fig.add_subplot(spec); ax.set_facecolor(t["panel"])
            for sp in ax.spines.values(): sp.set_color(t["grid"]); sp.set_linewidth(.65)
            ax.tick_params(colors=t["muted"], labelsize=6.5, length=2, width=.5); ax.grid(color=t["grid"], alpha=.45, lw=.45)
            ax.set_title(f" {title} ", loc="left", color=t["background"], backgroundcolor=t["primary"], fontsize=7.5, weight="bold", pad=2)
            return ax

        score = _num(r.get("rec", {}).get("score")); action = _plain_label(r.get("action", {}).get("action") or r.get("rec", {}).get("rating"))
        nav = fig.add_axes([.015,.944,.97,.042]); nav.set_facecolor(t["panel_alt"]); nav.set_xticks([]); nav.set_yticks([])
        for sp in nav.spines.values(): sp.set_color(t["primary"]); sp.set_linewidth(.65)
        nav.text(.008,.55,"QNT <GO>",color=t["neutral"],fontsize=10,weight="bold",va="center")
        nav.text(.095,.55,f"{self.ticker} EQUITY  |  MONITOR  |  RISK  |  FORECAST  |  EXECUTION",color=t["neutral"],fontsize=8,weight="bold",va="center")
        nav.text(.99,.55,f"{r.get('date', str(close.index[-1])[:16])}  VN",color=t["primary"],fontsize=7,ha="right",va="center")

        kpi_ax = panel(gs[0:3, :], "DECISION / MARKET SNAPSHOT"); kpi_ax.set_xticks([]); kpi_ax.set_yticks([])
        kpis = [
            ("CALL", action or "N/A", _action_color(t, action)),
            ("QUANT SCORE", f"{score:.0f}/100" if score is not None else "N/A", t["neutral"]),
            (f"ENSEMBLE +{m['horizon']}D", _pct(m["ensemble"]), t["positive"] if (m["ensemble"] or 0) >= 0 else t["negative"]),
            ("MC MEDIAN", _pct(m["mc_median"]), t["positive"] if (m["mc_median"] or 0) >= 0 else t["negative"]),
            ("P(RETURN > 0)", _percent(m["prob_up"]), t["neutral"]),
            ("MODEL AGREEMENT", _percent(m["agreement"]), t["neutral"]),
            ("R:R TO TP2", f"{m['rr']:.2f}x" if m["rr"] is not None else "N/A", t["neutral"]),
            ("P(TP1 BEFORE SL)", _percent(m["p_tp_before_sl"]), t["neutral"]),
        ]
        for i, (lab, val, color) in enumerate(kpis):
            x = .012 + i / len(kpis)
            if i: kpi_ax.plot([x-.012,x-.012],[.08,.88],transform=kpi_ax.transAxes,color=t["grid"],lw=.45)
            kpi_ax.text(x, .69, lab, transform=kpi_ax.transAxes, color=t["muted"], fontsize=6.2, weight="bold")
            kpi_ax.text(x, .21, val, transform=kpi_ax.transAxes, color=color, fontsize=11, weight="bold")

        price_ax = panel(gs[3:11, :15], "1  PRICE / MONTE CARLO PROBABILITY CONE")
        xh = np.arange(len(close)); price_ax.plot(xh, close.values, color=t["neutral"], lw=1.25, label="History")
        if paths is not None:
            q10, q25, q50, q75, q90 = np.percentile(paths, [10,25,50,75,90], axis=0); xf = np.arange(len(close), len(close)+paths.shape[1])
            price_ax.fill_between(xf, q10, q90, color=t["primary"], alpha=.10, label="P10-P90")
            price_ax.fill_between(xf, q25, q75, color=t["primary"], alpha=.25, label="P25-P75")
            price_ax.plot(np.r_[len(close)-1, xf], np.r_[close.iloc[-1], q50], color=t["primary"], lw=1.8, label="MC P50")
        for key, color, lab in [("entry",t["primary"],"ENTRY"),("stop",t["negative"],"SL"),("tp1",t["positive"],"TP1"),("tp2",t["positive"],"TP2")]:
            v=m[key]
            if v is not None: price_ax.axhline(v,color=color,ls="--",lw=.8,alpha=.85); price_ax.text(.995,v,lab,transform=price_ax.get_yaxis_transform(),ha="right",va="bottom",color=color,fontsize=7)
        price_ax.legend(loc="upper left", ncol=4, frameon=False, fontsize=7, labelcolor=t["muted"])
        price_ax.set_ylabel("VND", color=t["muted"], fontsize=7)

        risk_ax = panel(gs[3:11, 15:], "2  RISK MONITOR / DRAWDOWN")
        dd = close / close.cummax() - 1; risk_ax.plot(dd.index, dd*100, color=t["negative"], lw=1.2); risk_ax.fill_between(dd.index, dd*100, 0, color=t["negative"], alpha=.18)
        risk_ax.set_ylabel("Drawdown %", color=t["muted"], fontsize=7)
        risk_items=[("ANN VOL",m["vol"]),("MAX DD",m["max_dd"]),("VaR 95",m["var"]),("CVaR 95",m["cvar"])]
        for i,(lab,val) in enumerate(risk_items): risk_ax.text(.03+i*.245,.06,f"{lab}\n{_percent(val) if lab=='ANN VOL' else _pct(val)}",transform=risk_ax.transAxes,color=t["neutral"],fontsize=8,weight="bold")

        factor_ax = panel(gs[11:20, :8], "3  FACTOR EXPOSURE [0-100]")
        factors=self._factor_scores()
        terminal_factor_labels = ["TREND", "MOMENTUM", "MONEY FLOW", "RELATIVE STRENGTH", "RISK QUALITY", "FORECAST"]
        labels=terminal_factor_labels[:len(factors)]; vals=list(factors.values()); y=np.arange(len(labels)); factor_ax.barh(y,vals,color=[t["positive"] if v>=60 else t["neutral"] if v>=40 else t["negative"] for v in vals],height=.55)
        factor_ax.set_yticks(y, labels, fontsize=7); factor_ax.set_xlim(0,100); factor_ax.invert_yaxis(); factor_ax.axvline(50,color=t["muted"],ls="--",lw=.7)

        model_ax = panel(gs[11:20, 8:16], "4  MODEL / FORECAST AUDIT"); model_ax.set_xticks([]); model_ax.set_yticks([])
        fc=r.get("fcast",{}); audit=[("Horizon",f"{m['horizon']} sessions"),("Simulations",f"{m['simulations']:,}"),("Ensemble ER",_pct(m["ensemble"])),("MC median",_pct(m["mc_median"])),("MC vol source",str(fc.get("mc",{}).get("vol_source","N/A"))),("Agreement (not P-win)",_percent(m["agreement"])),("Calibrated P-win",_percent(m["pwin_cal"]) if m["pwin_cal"] is not None else "N/A - no OOS calibration")]
        for i,(lab,val) in enumerate(audit):
            y=.88-i*.125; model_ax.axhline(y-.045,color=t["grid"],lw=.35,alpha=.7)
            model_ax.text(.035,y,lab,color=t["muted"],fontsize=6.8); model_ax.text(.47,y,val,color=t["neutral"],fontsize=7.2,weight="bold")

        trade_ax = panel(gs[11:20, 16:], "5  EXECUTION / PAYOFF"); trade_ax.set_xticks([]); trade_ax.set_yticks([])
        trade=[("ENTRY",_price(m["entry"])),("STOP",f"{_price(m['stop'])}  ({_pct(m['downside'])})"),("TP1",f"{_price(m['tp1'])}  ({_pct(m['upside1'])})"),("TP2",f"{_price(m['tp2'])}  ({_pct(m['upside2'])})"),("R:R TP2",f"{m['rr']:.2f}x" if m["rr"] is not None else "N/A"),("EV / TRADE",f"{m['ev_r']:+.3f}R" if m["ev_r"] is not None else "N/A - no calibrated P-win")]
        for i,(lab,val) in enumerate(trade):
            y=.86-i*.145; trade_ax.axhline(y-.05,color=t["grid"],lw=.35,alpha=.7)
            trade_ax.text(.035,y,lab,color=t["muted"],fontsize=6.8); trade_ax.text(.40,y,val,color=t["neutral"],fontsize=7.8,weight="bold")
        fig.text(.025,.012,"QNT SYSTEM // MODEL OUTPUT — NOT A GUARANTEE // AGREEMENT != PROBABILITY OF PROFIT",color=t["muted"],fontsize=5.8)
        return self._save(fig,path,t)

    def _terminal_dashboard_cards(self, path: Path):
        """Three-column legacy terminal: tables, status blocks and sparklines only."""
        t = dict(TERMINAL_THEME); r = self.report; m = self._decision_metrics(); paths = self._paths()
        close = self.prices["close"].tail(100); score = _num(r.get("rec",{}).get("score"))
        action = _plain_label(r.get("action",{}).get("action") or r.get("rec",{}).get("rating"))
        hmm = _plain_label(r.get("hmm",{}).get("current")) or "N/A"
        hmm_prob = _num(r.get("hmm",{}).get("prob_pct")); factors = self._factor_scores()
        plt.rcParams.update({"font.family":t["font"],"axes.unicode_minus":False})
        fig=plt.figure(figsize=t["figure_size"],dpi=t["dpi"],facecolor=t["background"])

        def box(rect, title, code=""):
            ax=fig.add_axes(rect); ax.set_facecolor(t["background"]); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_color(t["grid"]); sp.set_linewidth(.7)
            ax.text(.025,.965,title,color=t["neutral"],fontsize=9,weight="bold",va="top",transform=ax.transAxes)
            if code: ax.text(.975,.965,code,color="#687078",fontsize=6.5,ha="right",va="top",transform=ax.transAxes)
            ax.plot([.02,.98],[.90,.90],transform=ax.transAxes,color=t["grid"],lw=.55)
            ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_autoscale_on(False)
            return ax

        # Legacy terminal chrome and command line.
        top=fig.add_axes([.012,.945,.976,.04]); top.set_facecolor("#050507"); top.set_xticks([]); top.set_yticks([])
        for sp in top.spines.values(): sp.set_color(t["grid"]); sp.set_linewidth(.6)
        top.text(.012,.52,"NAIZY",color=t["neutral"],fontsize=10,weight="bold",va="center")
        top.text(.075,.52,"QUANT TERMINAL — HOSE MONITOR",color="#687078",fontsize=8,va="center")
        top.text(.985,.52,f"ICT {str(r.get('date',''))[11:16] or 'LIVE'}",color=t["primary"],fontsize=8,ha="right",va="center")
        cmd=fig.add_axes([.012,.895,.976,.047]); cmd.set_facecolor(t["neutral"]); cmd.set_xticks([]); cmd.set_yticks([])
        for sp in cmd.spines.values(): sp.set_visible(False)
        cmd.text(.012,.52,f"> GO    {self.ticker} <Equity>  QNT <GO>",color="#111111",fontsize=9,weight="bold",va="center")
        cmd.text(.985,.52,"HMM · GARCH · MC · SCORE",color="#111111",fontsize=7,weight="bold",ha="right",va="center")

        # Ticker tape: current instrument snapshots, no graphical bars.
        tape=fig.add_axes([.012,.853,.976,.038]); tape.set_facecolor("#050507"); tape.set_xticks([]); tape.set_yticks([])
        for sp in tape.spines.values(): sp.set_color(t["grid"]); sp.set_linewidth(.5)
        last=float(close.iloc[-1]); day=(close.iloc[-1]/close.iloc[-2]-1)*100 if len(close)>1 else 0
        tape_items=[(self.ticker,last,day),("ENTRY",m["entry"],None),("STOP",m["stop"],m["downside"]),("TP1",m["tp1"],m["upside1"]),("TP2",m["tp2"],m["upside2"]),("R:R",m["rr"],None)]
        for i,(lab,val,ch) in enumerate(tape_items):
            x=.018+i*.163; tape.text(x,.56,lab,color=t["text"],fontsize=7.5,weight="bold",va="center")
            display=f"{val:,.0f}" if val is not None and lab!="R:R" else f"{val:.2f}x" if val is not None else "N/A"
            tape.text(x+.047,.56,display,color=t["neutral"],fontsize=7.5,weight="bold",va="center")
            if ch is not None: tape.text(x+.108,.56,f"{ch:+.1f}%",color=t["positive"] if ch>=0 else t["negative"],fontsize=7,weight="bold",va="center")

        left=box([.012,.30,.38,.545],f"{self.ticker} SIGNAL & EXECUTION","TRADE<GO>")
        left.text(.035,.835,"CALL",color="#687078",fontsize=6.2); left.text(.20,.835,action,color=_action_color(t,action),fontsize=14,weight="bold")
        left.text(.48,.835,"SCORE",color="#687078",fontsize=6.2); left.text(.64,.835,f"{score:.0f}/100" if score is not None else "N/A",color=t["neutral"],fontsize=12,weight="bold")
        left.text(.035,.765,"LAST / 1D",color="#687078",fontsize=6.2); left.text(.20,.765,f"{last:,.0f}",color=t["neutral"],fontsize=8.5,weight="bold"); left.text(.34,.765,f"{day:+.1f}%",color=t["positive"] if day>=0 else t["negative"],fontsize=8,weight="bold")
        left.text(.48,.765,"HORIZON",color="#687078",fontsize=6.2); left.text(.64,.765,f"{m['horizon']} SESS",color=t["text"],fontsize=7.5,weight="bold")
        costs=r.get("costs",{}); pos=r.get("pos",{}); atr=_num(r.get("sl",{}).get("atr"))
        rows=[("ENTRY",m["entry"],None),("STOP LOSS",m["stop"],m["downside"]),("TARGET 1",m["tp1"],m["upside1"]),("TARGET 2",m["tp2"],m["upside2"]),("R:R TO TP2",m["rr"],None),("P(TP1 BEFORE SL)",m["p_tp_before_sl"],"pct"),("ATR / ROUNDTRIP",atr,"atr"),("POSITION SHARES",_num(pos.get("shares",pos.get("qty"))),"shares")]
        for i,(lab,val,change) in enumerate(rows):
            y=.67-i*.061; left.plot([.03,.97],[y-.024,y-.024],transform=left.transAxes,color=t["grid"],lw=.35)
            left.text(.035,y,lab,color=t["primary"],fontsize=6.2,va="center")
            if lab=="R:R TO TP2": value=f"{val:.2f}x" if val is not None else "N/A"
            elif change=="pct": value=_percent(val)
            elif change=="atr": value=f"{_price(val)} / {_percent(costs.get('roundtrip_cost_pct'))}"
            elif change=="shares": value=f"{val:,.0f}" if val is not None else "N/A"
            else: value=_price(val)
            left.text(.48,y,value,color=t["neutral"],fontsize=7.5,weight="bold",va="center")
            if isinstance(change,(int,float)): left.text(.72,y,_pct(change),color=t["positive"] if change>=0 else t["negative"],fontsize=7.2,weight="bold",va="center")

        mid=box([.397,.30,.285,.545],"REGIME & RISK","HMM<GO>")
        regime_color=t["positive"] if "BULL" in hmm else t["negative"] if "BEAR" in hmm else t["neutral"]
        mid.text(.50,.83,hmm,color=regime_color,fontsize=18,weight="bold",ha="center")
        mid.text(.50,.775,f"P(REGIME) {_percent(hmm_prob)}  |  METHOD {str(r.get('hmm',{}).get('method','HMM')).upper()}",color=t["text"],fontsize=6.2,ha="center")
        st=r.get("stats",{}); cs=r.get("alpha",{}).get("cross_sectional",{})
        risk_rows=[("ANN VOL",m["vol"],"pct"),("MAX DD",m["max_dd"],"sign"),("VaR 95",m["var"],"sign"),("CVaR 95",m["cvar"],"sign"),("SHARPE",st.get("sharpe"),"num"),("SORTINO",st.get("sortino"),"num"),("BETA",cs.get("beta"),"num")]
        for i,(lab,val,kind) in enumerate(risk_rows):
            col=i%2; row=i//2; x=.045+col*.50; y=.69-row*.075
            mid.text(x,y,lab,color=t["primary"],fontsize=6.1); shown=_percent(val) if kind=="pct" else _pct(val) if kind=="sign" else f"{float(val):.2f}" if _num(val) is not None else "N/A"
            mid.text(x+.42,y,shown,color=t["negative"] if kind=="sign" else t["neutral"],fontsize=7.2,weight="bold",ha="right")
        spark=mid.inset_axes([.045,.08,.91,.255]); spark.set_facecolor(t["background"]); dd=close/close.cummax()-1
        spark.plot(np.arange(len(dd)),dd*100,color=t["negative"],lw=1); spark.axhline(0,color=t["grid"],lw=.4); spark.set_xticks([]); spark.set_yticks([])
        for sp in spark.spines.values(): sp.set_visible(False)
        mid.text(.045,.35,"DRAWDOWN HISTORY / 100 SESSIONS",color="#687078",fontsize=5.8)

        right=box([.687,.30,.301,.545],"FORECAST & MODEL","MC<GO>")
        mc=r.get("fcast",{}).get("mc",{}); forecast_rows=[("ENSEMBLE ER",_pct(m["ensemble"])),("MC MEDIAN",_pct(m["mc_median"])),("P(RETURN > 0)",_percent(m["prob_up"])),("MODEL AGREEMENT",_percent(m["agreement"])),("SIMULATIONS",f"{m['simulations']:,}"),("VOL SOURCE",str(mc.get("vol_source","N/A")))]
        for i,(lab,val) in enumerate(forecast_rows):
            y=.835-i*.052; right.text(.045,y,lab,color=t["primary"],fontsize=5.9); right.text(.95,y,val,color=t["neutral"],fontsize=6.9,weight="bold",ha="right")
        # Old-terminal probability bars: fixed-width blocks, not modern rounded cards.
        probs=[("UP",m["prob_up"],t["positive"]),("TP1",m["p_tp_before_sl"],t["positive"]),("LOSS>3",_num(r.get("fcast",{}).get("lock_risk",{}).get("prob_loss_gt_3pct")),t["negative"])]
        for i,(lab,val,col) in enumerate(probs):
            y=.50-i*.055; right.text(.045,y,lab,color=t["primary"],fontsize=5.8,va="center"); right.add_patch(Rectangle((.19,y-.012),.56,.022,transform=right.transAxes,facecolor="#1B1B20",edgecolor="none")); width=.56*np.clip((val or 0)/100,0,1); right.add_patch(Rectangle((.19,y-.012),width,.022,transform=right.transAxes,facecolor=col,edgecolor="none")); right.text(.95,y,_percent(val),color=t["neutral"],fontsize=6.4,weight="bold",ha="right",va="center")
        fan=right.inset_axes([.045,.065,.91,.245]); fan.set_facecolor(t["background"])
        if paths is not None:
            current=float(close.iloc[-1]); q10,q25,q50,q75,q90=np.percentile(paths,[10,25,50,75,90],axis=0); x=np.arange(paths.shape[1]+1)
            fan.fill_between(x,np.r_[current,q10],np.r_[current,q90],color=t["primary"],alpha=.09)
            fan.fill_between(x,np.r_[current,q25],np.r_[current,q75],color=t["primary"],alpha=.19)
            fan.plot(x,np.r_[current,q50],color=t["neutral"],lw=.9,drawstyle="steps-mid")
            terminal=paths[:,-1]; labels=[("P10",np.percentile(terminal,10)),("P25",np.percentile(terminal,25)),("P50",np.percentile(terminal,50)),("P75",np.percentile(terminal,75)),("P90",np.percentile(terminal,90))]
            for i,(lab,val) in enumerate(labels): fan.text(.02+i*.195,.92,f"{lab} {val:,.0f}",transform=fan.transAxes,color=t["primary"] if lab!="P50" else t["neutral"],fontsize=4.8,va="top")
        fan.set_xticks([]); fan.set_yticks([])
        for sp in fan.spines.values(): sp.set_visible(False)
        right.text(.045,.325,"MC FAN / TERMINAL PERCENTILES",color="#687078",fontsize=5.8)

        factors_ax=box([.012,.095,.976,.195],"FACTOR SCOREBOARD — NUMERIC VIEW","SCORE<GO>")
        factor_labels=["TREND","MOMENTUM","MONEY FLOW","REL STRENGTH","RISK QUALITY","FORECAST"]
        for i,(lab,val) in enumerate(zip(factor_labels,list(factors.values()))):
            x=.025+i*.162; factors_ax.text(x,.66,lab,color=t["primary"],fontsize=6.8,weight="bold")
            factors_ax.text(x,.30,f"{val:05.1f}",color=t["positive"] if val>=60 else t["neutral"] if val>=40 else t["negative"],fontsize=13,weight="bold")
        logax=fig.add_axes([.012,.035,.976,.045]); logax.set_facecolor("#050507"); logax.set_xticks([]); logax.set_yticks([])
        for sp in logax.spines.values(): sp.set_color(t["grid"]); sp.set_linewidth(.5)
        logax.text(.012,.55,"LOG",color=t["primary"],fontsize=7,weight="bold",va="center")
        logax.text(.055,.55,f"DATA {r.get('range','N/A')}  ::  GARCH-MC OK  ::  HMM {hmm}  ::  OOS CALIBRATION {'OK' if m['pwin_cal'] is not None else 'PENDING'}",color="#687078",fontsize=6.5,va="center")
        logax.text(.985,.55,"MODEL OUTPUT · NOT A GUARANTEE",color=t["neutral"],fontsize=6.5,ha="right",va="center")
        return self._save(fig,path,t,tight=False)

    def _terminal_dashboard_dense(self, path: Path):
        """Dense classic research screen: quote, models, risk and distributions."""
        t=dict(TERMINAL_THEME); r=self.report; m=self._decision_metrics(); paths=self._paths()
        df=self.prices.tail(120); close=df["close"].astype(float); last=float(close.iloc[-1]); score=_num(r.get("rec",{}).get("score"))
        action=_plain_label(r.get("action",{}).get("action") or r.get("rec",{}).get("rating")); factors=self._factor_scores()
        plt.rcParams.update({"font.family":t["font"],"axes.unicode_minus":False})
        fig=plt.figure(figsize=t["figure_size"],dpi=t["dpi"],facecolor=t["background"])

        def panel(rect,title,code):
            ax=fig.add_axes(rect); ax.set_facecolor(t["background"]); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_autoscale_on(False)
            for sp in ax.spines.values(): sp.set_color(t["grid"]); sp.set_linewidth(.55)
            ax.text(.015,.975,title,color=t["neutral"],fontsize=7.2,weight="bold",va="top",transform=ax.transAxes)
            ax.text(.985,.975,code,color="#65656D",fontsize=5.4,ha="right",va="top",transform=ax.transAxes)
            ax.plot([.01,.99],[.925,.925],transform=ax.transAxes,color=t["grid"],lw=.45)
            return ax

        # Compact terminal chrome.
        top=fig.add_axes([.01,.952,.98,.035]); top.set_facecolor("#050507"); top.set_xticks([]); top.set_yticks([])
        for sp in top.spines.values(): sp.set_color(t["grid"]); sp.set_linewidth(.5)
        top.text(.01,.52,"NAIZY",color=t["neutral"],fontsize=8.5,weight="bold",va="center"); top.text(.065,.52,"QUANT RESEARCH TERMINAL / VN EQUITIES",color="#65656D",fontsize=6.5,va="center"); top.text(.99,.52,f"ICT {str(r.get('date',''))[11:16] or 'LIVE'}",color=t["primary"],fontsize=6.5,ha="right",va="center")
        cmd=fig.add_axes([.01,.907,.98,.041]); cmd.set_facecolor(t["neutral"]); cmd.set_xticks([]); cmd.set_yticks([]); [sp.set_visible(False) for sp in cmd.spines.values()]
        cmd.text(.012,.52,f"> GO   {self.ticker} <EQUITY>   QNT <GO>   ANALYZE <GO>",color="#0A0A0C",fontsize=8,weight="bold",va="center"); cmd.text(.988,.52,"QUOTE · PRICE · HMM · GARCH · MONTE CARLO · RISK",color="#0A0A0C",fontsize=6,weight="bold",ha="right",va="center")
        tape=fig.add_axes([.01,.866,.98,.037]); tape.set_facecolor("#050507"); tape.set_xticks([]); tape.set_yticks([])
        for sp in tape.spines.values(): sp.set_color(t["grid"]); sp.set_linewidth(.45)
        day=(close.iloc[-1]/close.iloc[-2]-1)*100; tape_data=[("LAST",last,day),("ENTRY",m["entry"],None),("STOP",m["stop"],m["downside"]),("TP1",m["tp1"],m["upside1"]),("TP2",m["tp2"],m["upside2"]),("R:R",m["rr"],None),("SCORE",score,None)]
        for i,(lab,val,ch) in enumerate(tape_data):
            x=.014+i*.14; tape.text(x,.55,lab,color=t["primary"],fontsize=5.8,va="center"); shown=f"{val:,.0f}" if val is not None and lab not in ("R:R","SCORE") else f"{val:.2f}x" if lab=="R:R" and val is not None else f"{val:.0f}/100" if val is not None else "N/A"; tape.text(x+.047,.55,shown,color=t["neutral"],fontsize=6.8,weight="bold",va="center");
            if ch is not None: tape.text(x+.095,.55,f"{ch:+.1f}%",color=t["positive"] if ch>=0 else t["negative"],fontsize=5.8,weight="bold",va="center")

        # LEFT: dominant price/volume workspace, with data table beneath chart.
        pleft=panel([.01,.365,.475,.495],f"{self.ticker} PRICE / VOLUME / STRUCTURE","GP<GO>")
        chart=pleft.inset_axes([.055,.31,.91,.57]); chart.set_facecolor(t["background"]); x=np.arange(len(close)); chart.plot(x,close.values,color=t["neutral"],lw=1.05)
        for val,col,lab in [(m["entry"],t["primary"],"ENTRY"),(m["stop"],t["negative"],"SL"),(m["tp1"],t["positive"],"TP1"),(m["tp2"],t["positive"],"TP2")]:
            if val is not None: chart.axhline(val,color=col,lw=.45,ls="--"); chart.text(len(x)-1,val,lab,color=col,fontsize=4.5,ha="right",va="bottom")
        chart.grid(color=t["grid"],lw=.3,alpha=.8); chart.tick_params(colors=t["primary"],labelsize=4.8,length=2); [sp.set_color(t["grid"]) for sp in chart.spines.values()]
        if "volume" in df:
            vol=chart.twinx(); vv=pd.to_numeric(df["volume"],errors="coerce").fillna(0).values; vol.bar(x,vv,color=t["primary"],alpha=.13,width=.8); vol.set_yticks([]); [sp.set_visible(False) for sp in vol.spines.values()]
        sr=r.get("sr",{}); trend=r.get("trend",{}); flow=r.get("flow",{}); liq=r.get("liquidity",{})
        quote_rows=[("RANGE",r.get("range","N/A")),("TREND / ER",f"{_plain_label(trend.get('label'))} / {_num(trend.get('er')) or 0:.3f}"),("CMF / FLOW",f"{_num(flow.get('cmf')) or 0:+.3f} / {_plain_label(flow.get('label'))}"),("SUPPORTS",", ".join(_price(v) for v in sr.get("supports",[])[:3]) or "N/A"),("RESISTANCES",", ".join(_price(v) for v in sr.get("resistances",[])[:3]) or "N/A"),("LIQUIDITY",_plain_label(liq.get("label",liq.get("status","N/A"))))]
        for i,(lab,val) in enumerate(quote_rows):
            col=i%3; row=i//3; xx=.035+col*.325; yy=.245-row*.105; pleft.text(xx,yy,lab,color=t["primary"],fontsize=5.2); pleft.text(xx,yy-.047,str(val),color=t["neutral"],fontsize=5.8,weight="bold")

        # CENTER: dense model matrix plus compact MC fan.
        pmodel=panel([.49,.365,.245,.495],"MODEL MATRIX / FORECAST","MOD<GO>")
        fc=r.get("fcast",{}); mc=fc.get("mc",{}); mr=fc.get("mr",{}); mom=fc.get("mom",{}); garch=r.get("garch",{}); hmm=r.get("hmm",{})
        model_rows=[("HMM STATE",_plain_label(hmm.get("current"))), ("HMM PROB",_percent(hmm.get("prob_pct"))), ("GARCH MODEL",garch.get("model","N/A")), ("VOL SOURCE",mc.get("vol_source","N/A")), ("ENSEMBLE ER",_pct(m["ensemble"])), ("MC MEDIAN",_pct(m["mc_median"])), ("MC P(UP)",_percent(m["prob_up"])), ("MEAN REV Z",f"{_num(mr.get('z')) or 0:+.3f}"), ("MOM PROJ",_pct(mom.get("proj_pct"))), ("AGREEMENT",_percent(m["agreement"])), ("SIMS / HORIZON",f"{m['simulations']:,} / {m['horizon']}D")]
        for i,(lab,val) in enumerate(model_rows):
            y=.875-i*.047; pmodel.text(.035,y,lab,color=t["primary"],fontsize=4.8); pmodel.text(.965,y,str(val),color=t["neutral"],fontsize=5.5,weight="bold",ha="right")
        fan=pmodel.inset_axes([.055,.075,.89,.255]); fan.set_facecolor(t["background"])
        if paths is not None:
            q=np.percentile(paths,[10,25,50,75,90],axis=0); xx=np.arange(paths.shape[1]+1); cur=float(close.iloc[-1]); fan.fill_between(xx,np.r_[cur,q[0]],np.r_[cur,q[4]],color=t["primary"],alpha=.09); fan.fill_between(xx,np.r_[cur,q[1]],np.r_[cur,q[3]],color=t["primary"],alpha=.20); fan.step(xx,np.r_[cur,q[2]],where="mid",color=t["neutral"],lw=.75)
            terminal=paths[:,-1]; labs=[10,25,50,75,90]
            for i,qv in enumerate(labs): pmodel.text(.035+i*.19,.355,f"P{qv} {np.percentile(terminal,qv):,.0f}",color=t["neutral"] if qv==50 else t["primary"],fontsize=4.5)
        fan.set_xticks([]); fan.set_yticks([]); [sp.set_color(t["grid"]) for sp in fan.spines.values()]

        # RIGHT: recommendation, execution and complete risk table.
        prisk=panel([.74,.365,.25,.495],"TRADE / RISK / PAYOFF","RISK<GO>")
        st=r.get("stats",{}); cs=r.get("alpha",{}).get("cross_sectional",{}); sl=r.get("sl",{}); pos=r.get("pos",{}); costs=r.get("costs",{})
        prisk.text(.035,.875,"CALL",color=t["primary"],fontsize=5.2); prisk.text(.24,.875,action,color=_action_color(t,action),fontsize=11,weight="bold"); prisk.text(.68,.875,"SCORE",color=t["primary"],fontsize=5.2); prisk.text(.965,.875,f"{score:.0f}" if score is not None else "N/A",color=t["neutral"],fontsize=9,weight="bold",ha="right")
        risk_rows=[("ENTRY",_price(m["entry"])),("STOP / DOWN",f"{_price(m['stop'])} / {_pct(m['downside'])}"),("TP1 / UP",f"{_price(m['tp1'])} / {_pct(m['upside1'])}"),("TP2 / UP",f"{_price(m['tp2'])} / {_pct(m['upside2'])}"),("R:R NET COST",f"{_num(sl.get('rr_net_cost',m['rr'])) or 0:.2f}x"),("P(TP1<SL)",_percent(m["p_tp_before_sl"])),("ATR / COST",f"{_price(sl.get('atr'))} / {_percent(costs.get('roundtrip_cost_pct'))}"),("POSITION",f"{_num(pos.get('shares',pos.get('qty'))) or 0:,.0f} sh"),("ANN VOL",_percent(m["vol"])),("MAX DD",_pct(m["max_dd"])),("VaR / CVaR",f"{_pct(m['var'])} / {_pct(m['cvar'])}"),("SHARPE / SORTINO",f"{_num(st.get('sharpe')) or 0:.2f} / {_num(st.get('sortino')) or 0:.2f}"),("BETA / RS20",f"{_num(cs.get('beta')) or 0:.2f} / {_pct(cs.get('rs_20d_pct'))}"),("EV / TRADE",f"{_num(sl.get('ev_r_per_trade')):+.3f}R" if _num(sl.get('ev_r_per_trade')) is not None else "N/A — NO OOS CAL")]
        for i,(lab,val) in enumerate(risk_rows):
            y=.805-i*.052; prisk.text(.035,y,lab,color=t["primary"],fontsize=4.8); color=t["negative"] if any(k in lab for k in ("STOP","MAX DD","VaR")) else t["neutral"]; prisk.text(.965,y,val,color=color,fontsize=5.6,weight="bold",ha="right"); prisk.plot([.03,.97],[y-.021,y-.021],transform=prisk.transAxes,color=t["grid"],lw=.25)

        # Bottom-left: factor and diagnostics matrix, no oversized score strip.
        pf=panel([.01,.075,.49,.282],"FACTOR / DIAGNOSTICS MATRIX","FAC<GO>")
        flabels=["TREND","MOMENTUM","MONEY FLOW","REL STRENGTH","RISK QUALITY","FORECAST"]
        fvals=list(factors.values()); diagnostics=[("SKEW",st.get("skew")),("KURT",st.get("kurtosis")),("AUTOCORR",r.get("ac",{}).get("avg_short")),("GARCH AIC",garch.get("aic")),("HMM BIC",hmm.get("bic")),("DATA N",r.get("n"))]
        for i,(lab,val) in enumerate(zip(flabels,fvals)):
            row=i//3; col=i%3; x0=.035+col*.32; y0=.78-row*.31; c=t["positive"] if val>=60 else t["neutral"] if val>=40 else t["negative"]; pf.text(x0,y0,lab,color=t["primary"],fontsize=5.2); pf.text(x0+.22,y0,f"{val:05.1f}",color=c,fontsize=8,weight="bold",ha="right"); dl,dv=diagnostics[i]; pf.text(x0,y0-.10,dl,color="#65656D",fontsize=4.7); pf.text(x0+.22,y0-.10,f"{_num(dv):.2f}" if _num(dv) is not None else "N/A",color=t["text"],fontsize=5.2,ha="right"); pf.plot([x0,x0+.25],[y0-.16,y0-.16],transform=pf.transAxes,color=t["grid"],lw=.3)

        # Bottom-right: terminal-return histogram and tail statistics.
        pdist=panel([.505,.075,.485,.282],"MONTE CARLO TERMINAL DISTRIBUTION / TAIL","DIST<GO>")
        histax=pdist.inset_axes([.045,.18,.65,.68]); histax.set_facecolor(t["background"])
        if paths is not None:
            ret=paths[:,-1]/last-1; bins=np.histogram_bin_edges(ret,bins=28); neg=ret[ret<0]; posret=ret[ret>=0]; histax.hist(neg,bins=bins,color=t["negative"],alpha=.75); histax.hist(posret,bins=bins,color=t["positive"],alpha=.75); histax.axvline(0,color=t["primary"],lw=.5); histax.tick_params(colors=t["primary"],labelsize=4.2,length=2); histax.xaxis.set_major_formatter(lambda x,p:f"{x*100:.0f}%"); [sp.set_color(t["grid"]) for sp in histax.spines.values()]
            var=np.percentile(ret,5); cvar=np.mean(ret[ret<=var]); drows=[("P>0",np.mean(ret>0)*100),("P<-3",np.mean(ret<-.03)*100),("MEDIAN",np.median(ret)*100),("MEAN",np.mean(ret)*100),("VaR95",var*100),("CVaR95",cvar*100)]
            for i,(lab,val) in enumerate(drows): y=.80-i*.115; pdist.text(.735,y,lab,color=t["primary"],fontsize=4.9); pdist.text(.965,y,_percent(val) if lab.startswith("P") else _pct(val),color=t["neutral"] if val>=0 else t["negative"],fontsize=5.8,weight="bold",ha="right")
        logax=fig.add_axes([.01,.025,.98,.04]); logax.set_facecolor("#050507"); logax.set_xticks([]); logax.set_yticks([]); [sp.set_color(t["grid"]) for sp in logax.spines.values()]
        logax.text(.012,.55,"LOG",color=t["primary"],fontsize=5.8,weight="bold",va="center"); logax.text(.055,.55,f"DATA {r.get('range','N/A')} :: SOURCE OK :: HMM {_plain_label(hmm.get('current'))} :: GARCH {garch.get('model','N/A')} :: OOS CAL {'OK' if m['pwin_cal'] is not None else 'PENDING'}",color="#65656D",fontsize=5.2,va="center"); logax.text(.985,.55,"MODEL OUTPUT · NOT A GUARANTEE",color=t["neutral"],fontsize=5.2,ha="right",va="center")
        return self._save(fig,path,t,tight=False)

    def terminal_dashboard(self, path: Path):
        """Six-panel research terminal, with model and review status separated."""
        from terminal_research_layout import render_research_terminal
        return render_research_terminal(self, Path(path))

    def _terminal_dashboard_reference(self, path: Path):
        """Institutional one-page research sheet with icon-led diagnostics."""
        t=dict(TERMINAL_THEME); r=self.report; m=self._decision_metrics(); paths=self._paths(); df=self.prices.tail(180)
        close=df["close"].astype(float); last=float(close.iloc[-1]); score=_num(r.get("rec",{}).get("score")); factors=self._factor_scores()
        action=_plain_label(r.get("action",{}).get("action") or r.get("rec",{}).get("rating")); sector=r.get("sector",{}).get("name","N/A")
        plt.rcParams.update({"font.family":t["font"],"axes.unicode_minus":False})
        fig=plt.figure(figsize=t["figure_size"],dpi=t["dpi"],facecolor=t["background"])

        def panel(rect,title,subtitle=""):
            ax=fig.add_axes(rect); ax.set_facecolor(t["background"]); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_autoscale_on(False); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_color(t["grid"]); sp.set_linewidth(.55)
            ax.text(.015,.975,title,color=t["neutral"],fontsize=6.7,weight="bold",va="top",transform=ax.transAxes)
            if subtitle: ax.text(.985,.975,subtitle,color="#66666F",fontsize=4.8,ha="right",va="top",transform=ax.transAxes)
            ax.plot([.01,.99],[.92,.92],transform=ax.transAxes,color=t["grid"],lw=.4)
            return ax

        def icon(ax,x,y,symbol,color=t["primary"]):
            ax.add_patch(plt.Circle((x,y),.022,transform=ax.transAxes,fill=False,ec=color,lw=.7))
            ax.text(x,y-.001,symbol,color=color,fontsize=6.5,ha="center",va="center",fontfamily="DejaVu Sans",weight="bold",transform=ax.transAxes)

        # Research header and identity.
        head=fig.add_axes([.008,.955,.984,.034]); head.set_facecolor("#050507"); head.set_xticks([]); head.set_yticks([]); [sp.set_color(t["grid"]) for sp in head.spines.values()]
        head.text(.012,.52,"QUANT RESEARCH TERMINAL / VN EQUITIES",color="#92929B",fontsize=6.8,weight="bold",va="center"); head.text(.988,.52,f"DATA: EOD {str(r.get('date',''))[:16]} ICT",color="#92929B",fontsize=5.5,ha="right",va="center")
        ident=fig.add_axes([.008,.895,.984,.057]); ident.set_facecolor("#070709"); ident.set_xticks([]); ident.set_yticks([]); ident.set_xlim(0,1); ident.set_ylim(0,1); ident.set_autoscale_on(False); [sp.set_color(t["grid"]) for sp in ident.spines.values()]
        ident.text(.012,.58,f"{self.ticker} <EQUITY>",color=t["neutral"],fontsize=20,weight="bold",va="center"); ident.text(.185,.68,str(r.get("company_name") or r.get("company") or sector),color="#A7A7AF",fontsize=7,va="center"); ident.text(.185,.32,f"Sector: {sector}  |  Exchange: {r.get('exchange','HOSE')}",color="#777780",fontsize=5.8,va="center")
        ident.text(.82,.68,"QUANT SCORE",color="#92929B",fontsize=5.5,weight="bold"); ident.text(.87,.48,f"{score:.0f}" if score is not None else "N/A",color=t["neutral"],fontsize=15,weight="bold",ha="center"); ident.text(.90,.48,"/100",color="#92929B",fontsize=8); ident.plot([.925,.925],[.16,.84],color=t["grid"],lw=.6); ident.text(.958,.68,"RATING",color="#92929B",fontsize=5.5,ha="center"); ident.text(.958,.36,action,color=_action_color(t,action),fontsize=8,weight="bold",ha="center")

        strip=fig.add_axes([.008,.842,.79,.049]); strip.set_facecolor("#070709"); strip.set_xticks([]); strip.set_yticks([]); strip.set_xlim(0,1); strip.set_ylim(0,1); strip.set_autoscale_on(False); [sp.set_color(t["grid"]) for sp in strip.spines.values()]
        strip_vals=[("ALERT",last), ("ENTRY",m["entry"]),("STOP LOSS",m["stop"]),("TP1",m["tp1"]),("TP2",m["tp2"]),("RISK / REWARD",m["rr"]),("HOLDING PERIOD",m["horizon"]),("POSITION SIZE",_num(r.get("pos",{}).get("shares",r.get("pos",{}).get("qty"))))]
        for i,(lab,val) in enumerate(strip_vals):
            x=.012+i*.123; strip.text(x+.055,.74,lab,color="#777780",fontsize=4.7,ha="center"); shown=f"{val:.2f}x" if lab=="RISK / REWARD" and val is not None else f"{val:.0f} sess" if lab=="HOLDING PERIOD" and val is not None else f"{val:,.0f} sh" if lab=="POSITION SIZE" and val is not None else _price(val); col=t["negative"] if lab=="STOP LOSS" else t["positive"] if lab in ("ENTRY","TP1","TP2") else t["neutral"]; strip.text(x+.055,.28,shown,color=col,fontsize=6.8,weight="bold",ha="center");
            if i<7: strip.plot([x+.119,x+.119],[.15,.82],color=t["grid"],lw=.5)
        summary=fig.add_axes([.803,.842,.189,.049]); summary.set_facecolor("#070709"); summary.set_xticks([]); summary.set_yticks([]); [sp.set_color(t["grid"]) for sp in summary.spines.values()]
        summary.text(.5,.68,"SIGNAL STATUS",color=t["neutral"],fontsize=6,weight="bold",ha="center"); summary.text(.5,.28,"FORMING — NEEDS CONFIRMATION" if "AVOID" not in action.upper() else "EDGE NOT SUFFICIENT",color=t["neutral"],fontsize=5.3,weight="bold",ha="center")

        summary.add_patch(Rectangle((0,.02),1,.48,transform=summary.transAxes,fc="#070709",ec="none",zorder=3))
        weak_payoff = (m["rr"] is not None and m["rr"] < 1.25) or (m["prob_up"] is not None and m["prob_up"] < 55) or (m["agreement"] is not None and m["agreement"] < 60)
        status = "LOW CONVICTION - CHECK PAYOFF" if "BUY" in action.upper() and weak_payoff else "CONFIRMED SETUP" if action.upper() == "BUY_NOW" else "WAITING FOR CONFIRMATION" if "AVOID" not in action.upper() else "EDGE NOT SUFFICIENT"
        summary.text(.5,.28,status,color=_action_color(t,action),fontsize=5.3,weight="bold",ha="center",zorder=4)

        # 1. Price / probability / structure.
        p1=panel([.008,.475,.44,.362],"1. PRICE / PROBABILITY / STRUCTURE",f"Monte Carlo {m['horizon']} sessions | {m['simulations']:,} paths")
        ax=p1.inset_axes([.055,.12,.77,.75]); ax.set_facecolor(t["background"]); hist=close.tail(100); xh=np.arange(len(hist)); ax.plot(xh,hist.values,color=t["text"],lw=.9,label="History")
        if paths is not None:
            q10,q25,q50,q75,q90=np.percentile(paths,[10,25,50,75,90],axis=0); xf=np.arange(len(hist),len(hist)+paths.shape[1]); ax.fill_between(xf,q10,q90,color="#B8CCE0",alpha=.30,label="P10-P90"); ax.fill_between(xf,q25,q75,color="#5681AC",alpha=.55,label="P25-P75"); ax.plot(np.r_[len(hist)-1,xf],np.r_[hist.iloc[-1],q50],color="#2F7DFF",lw=1.1,label="P50")
        for val,col,lab in [(m["entry"],t["primary"],"ENTRY"),(m["stop"],t["negative"],"SL"),(m["tp1"],t["positive"],"TP1"),(m["tp2"],t["positive"],"TP2")]:
            if val is not None: ax.axhline(val,color=col,lw=.45,ls="--"); ax.text(len(hist)+(paths.shape[1] if paths is not None else 0)-1,val,lab,color=col,fontsize=4.2,ha="right")
        ax.grid(color=t["grid"],lw=.28,alpha=.7); ax.tick_params(colors="#8B8B94",labelsize=4.2,length=2); [sp.set_color(t["grid"]) for sp in ax.spines.values()]; ax.legend(loc="upper left",fontsize=4.5,frameon=False,labelcolor="#A8A8B0",ncol=4)
        side=[("Median return",_pct(m["mc_median"])),("P(Return > 0)",_percent(m["prob_up"])),("P(Loss > 3%)",_percent(r.get("fcast",{}).get("lock_risk",{}).get("prob_loss_gt_3pct"))),("P(TP1 first)",_percent(m["p_tp_before_sl"]))]
        for i,(lab,val) in enumerate(side): p1.text(.845,.78-i*.17,lab,color="#8B8B94",fontsize=4.8); p1.text(.845,.71-i*.17,val,color=t["neutral"] if i==0 else t["text"],fontsize=8.2,weight="bold")

        # 2. Factor matrix with icons retained.
        p2=panel([.452,.475,.21,.362],"2. FACTOR / DIAGNOSTICS MATRIX")
        flabels=[("↝","TREND"),("↗","MOMENTUM"),("$","MONEY FLOW"),("▥","RELATIVE STRENGTH"),("P","RISK QUALITY"),("◎","FORECAST")]; fvals=list(factors.values())
        flabels=[("T","TREND"),("M","MOMENTUM"),("$","MONEY FLOW"),("R","RELATIVE STRENGTH"),("Q","RISK QUALITY"),("F","FORECAST")]
        for i,((sym,lab),val) in enumerate(zip(flabels,fvals)):
            y=.84-i*.102; icon(p2,.055,y,sym); p2.text(.10,y,lab,color=t["primary"],fontsize=5.6,va="center"); p2.text(.965,y,f"{val:.1f} / 100",color=t["positive"] if val>=60 else t["neutral"] if val>=40 else t["negative"],fontsize=6.2,weight="bold",ha="right",va="center"); p2.plot([.02,.98],[y-.047,y-.047],color=t["grid"],lw=.35)
        st=r.get("stats",{}); cs=r.get("alpha",{}).get("cross_sectional",{}); small=[("AnnRet",st.get("ann_return_pct")),("Alpha",cs.get("alpha_ann_pct")),("Sharpe",st.get("sharpe")),("Beta",cs.get("beta")),("Daily PF",st.get("profit_factor")),("GARCH Pers.",r.get("garch",{}).get("persistence"))]
        for i,(lab,val) in enumerate(small): row=i//2; col=i%2; y=.19-row*.075; x=.035+col*.50; p2.text(x,y,lab,color="#8A8A93",fontsize=4.6); shown=_pct(val) if lab in ("AnnRet","Alpha") else f"{_num(val):.2f}" if _num(val) is not None else "N/A"; p2.text(x+.43,y,shown,color=t["neutral"] if _num(val) is None or _num(val)>=0 else t["negative"],fontsize=5.3,weight="bold",ha="right")

        # 3. Historical HMM shading and current regime.
        p3=panel([.666,.475,.326,.362],"3. HMM MARKET REGIME")
        rg=p3.inset_axes([.045,.12,.75,.75]); rg.set_facecolor(t["background"]); labels=r.get("hmm",{}).get("_state_labels")
        hclose=close.tail(150); hx=np.arange(len(hclose)); rg.plot(hx,hclose.values,color=t["text"],lw=.8)
        if labels:
            aligned=list(labels)[-len(hclose):]; aligned=([aligned[0]]*(len(hclose)-len(aligned))+aligned) if aligned else ["N/A"]*len(hclose); ss=pd.Series(aligned,index=hx); colors={"BULL":t["positive"],"BEAR":t["negative"],"SIDEWAY":"#D6A617","TRANSITION":"#9AA6B2"}; start=0
            for i in range(1,len(ss)+1):
                if i==len(ss) or ss.iloc[i]!=ss.iloc[start]: rg.axvspan(ss.index[start],ss.index[i-1],color=colors.get(ss.iloc[start],"#777"),alpha=.28,lw=0); start=i
        ticks=np.linspace(0,len(hclose)-1,5,dtype=int); rg.set_xticks(ticks,[str(hclose.index[i])[:7] for i in ticks]); rg.grid(color=t["grid"],lw=.25); rg.tick_params(colors="#8B8B94",labelsize=4.1,length=2); [sp.set_color(t["grid"]) for sp in rg.spines.values()]
        hmm=_plain_label(r.get("hmm",{}).get("current")); hp=_num(r.get("hmm",{}).get("prob_pct")); p3.text(.825,.68,"CURRENT REGIME",color=t["neutral"],fontsize=5.2,weight="bold"); p3.text(.825,.53,hmm,color=t["positive"] if "BULL" in hmm else t["negative"] if "BEAR" in hmm else t["neutral"],fontsize=13,weight="bold"); p3.text(.825,.38,"Probability",color="#8B8B94",fontsize=5); p3.text(.825,.27,_percent(hp),color=t["neutral"],fontsize=11,weight="bold")

        # 4. Monte Carlo terminal distribution.
        p4=panel([.008,.083,.36,.386],"4. MONTE CARLO DISTRIBUTION / TAIL",f"{m['horizon']} sessions | {m['simulations']:,} paths")
        da=p4.inset_axes([.055,.15,.73,.70]); da.set_facecolor(t["background"])
        if paths is not None:
            ret=paths[:,-1]/last-1; bins=np.histogram_bin_edges(ret,bins=30); da.hist(ret[ret<0],bins=bins,color=t["negative"],alpha=.78); da.hist(ret[ret>=0],bins=bins,color=t["positive"],alpha=.78); med=np.median(ret); var=np.percentile(ret,5); cvar=np.mean(ret[ret<=var]); da.axvline(0,color=t["text"],lw=.55,ls="--"); da.axvline(med,color="#498BFF",lw=.65,ls="--"); da.axvline(var,color=t["negative"],lw=.65,ls="--"); da.axvline(cvar,color=t["negative"],lw=.65,ls=":"); da.tick_params(colors="#8B8B94",labelsize=4.2,length=2); da.xaxis.set_major_formatter(lambda x,p:f"{x*100:.0f}%"); da.grid(color=t["grid"],lw=.25); [sp.set_color(t["grid"]) for sp in da.spines.values()]
            dside=[("Expected",m["ensemble"]),("Median",med*100),("P > 0",np.mean(ret>0)*100),("VaR95",var*100),("CVaR95",cvar*100)]
            for i,(lab,val) in enumerate(dside): p4.text(.81,.77-i*.135,lab,color="#8B8B94",fontsize=4.8); p4.text(.81,.70-i*.135,_percent(val) if lab=="P > 0" else _pct(val),color=t["positive"] if val>=0 else t["negative"],fontsize=8,weight="bold")

        # 5. Drawdown risk profile and risk notes.
        p5=panel([.372,.083,.414,.386],"5. RISK PROFILE")
        ra=p5.inset_axes([.055,.16,.62,.70]); ra.set_facecolor(t["background"]); dd=close/close.cummax()-1; ra.plot(dd.index,dd*100,color=t["negative"],lw=.8); ra.fill_between(dd.index,dd*100,0,color=t["negative"],alpha=.24); ra.tick_params(colors="#8B8B94",labelsize=4.2,length=2); ra.grid(color=t["grid"],lw=.25); [sp.set_color(t["grid"]) for sp in ra.spines.values()]
        notes=[("Max Drawdown",_pct(m["max_dd"])),("Volatility",_percent(m["vol"])),("VaR 95%",_pct(m["var"])),("CVaR 95%",_pct(m["cvar"])),("Beta",f"{_num(cs.get('beta')) or 0:.2f}")]
        for i,(lab,val) in enumerate(notes): p5.text(.71,.77-i*.12,lab,color="#8B8B94",fontsize=4.8); p5.text(.96,.70-i*.12,val,color=t["negative"] if i in (0,2,3) else t["text"],fontsize=7.5,weight="bold",ha="right")
        p5.text(.71,.18,"RISK INFORMATION",color=t["text"],fontsize=5.2,weight="bold"); p5.text(.71,.12,f"• GARCH: {r.get('fcast',{}).get('mc',{}).get('vol_source','N/A')}\n• Distribution: {_plain_label(r.get('dist',{}).get('model_rec','N/A'))}\n• Settlement: T+2",color="#92929B",fontsize=4.6,linespacing=1.7,va="top")

        # Decision/action block with retained icons.
        p6=panel([.79,.083,.202,.386],"DECISION / ACTION")
        p6.add_patch(FancyBboxPatch((.03,.59),.94,.27,boxstyle="round,pad=.012,rounding_size=.025",transform=p6.transAxes,fc="none",ec="#9C6500",lw=.8)); p6.text(.50,.80,f"CALL: {action}",color=_action_color(t,action),fontsize=12,weight="bold",ha="center")
        decisions=[("◎","SCORE",f"{score:.0f}/100" if score is not None else "N/A"),("▧","10D FORECAST",_pct(m["ensemble"])),("∞","MODEL AGREEMENT",_percent(m["agreement"])),("⚙","MODEL",str(r.get("fcast",{}).get("mc",{}).get("vol_source","N/A"))),("!","PRIORITY",_plain_label(r.get("action",{}).get("priority","MEDIUM")))]
        decisions=[("S","SCORE",f"{score:.0f}/100" if score is not None else "N/A"),("E",f"{m['horizon']}D FORECAST",_pct(m["ensemble"])),("A","MODEL AGREEMENT",_percent(m["agreement"])),("V","VOL MODEL",str(r.get("fcast",{}).get("mc",{}).get("vol_source","N/A"))),("!","PRIORITY",_plain_label(r.get("action",{}).get("priority","MEDIUM")))]
        for i,(sym,lab,val) in enumerate(decisions): y=.70-i*.10; icon(p6,.075,y,sym,t["neutral"] if i in (0,4) else t["primary"]); p6.text(.12,y,lab,color="#92929B",fontsize=4.8,va="center"); p6.text(.94,y,val,color=t["neutral"] if i in (0,4) else t["positive"] if i==2 else t["text"],fontsize=5.7,weight="bold",ha="right",va="center")
        p6.text(.04,.17,"INTERPRETATION",color=t["neutral"],fontsize=5.3,weight="bold"); reason="; ".join(map(str,r.get("action",{}).get("reason_codes",[])[:3])) or "Quant signal requires confirmation from price, flow and regime."
        p6.text(.04,.12,reason,color="#A0A0A8",fontsize=4.7,wrap=True,va="top")

        p6.add_patch(Rectangle((.025,.025),.95,.16,transform=p6.transAxes,fc=t["background"],ec="none",zorder=3))
        p6.text(.04,.17,"EVIDENCE / PAYOFF",color=t["neutral"],fontsize=5.3,weight="bold",zorder=4)
        rr_text=f"{m['rr']:.2f}x" if m["rr"] is not None else "N/A"
        evidence=(f"TP1 {_pct(m['upside1'])} | TP2 {_pct(m['upside2'])} | Stop {_pct(m['downside'])}\n"
                  f"P(up) {_percent(m['prob_up'])} | P(TP1 first) {_percent(m['p_tp_before_sl'])} | R:R {rr_text}")
        p6.text(.04,.12,evidence,color="#A0A0A8",fontsize=4.6,linespacing=1.65,va="top",zorder=4)

        log=fig.add_axes([.008,.025,.984,.048]); log.set_facecolor("#050507"); log.set_xticks([]); log.set_yticks([]); [sp.set_color(t["grid"]) for sp in log.spines.values()]
        log.text(.012,.55,"LOG",color=t["primary"],fontsize=5.4,weight="bold",va="center"); log.text(.055,.55,f"{str(r.get('date',''))[:16]}  |  SOURCE OK  |  MODEL OK  |  DATA QUALITY {r.get('data_quality',{}).get('status','N/A')}  |  LAST RUN ICT",color="#777780",fontsize=4.8,va="center"); log.text(.988,.55,"FOR INVESTMENT REFERENCE — NOT A BUY/SELL GUARANTEE",color=t["neutral"],fontsize=4.8,ha="right",va="center")
        return self._save(fig,path,t,tight=False)

    def client_recommendation(self, path: Path):
        """Readable one-page recommendation sheet with explicit assumptions."""
        t, r, m = self.theme, self.report, self._decision_metrics()
        fig=self._figure(); ax=fig.add_axes([0,0,1,1]); ax.set(xlim=(0,1),ylim=(0,1)); ax.axis("off")
        score=_num(r.get("rec",{}).get("score")); action=_plain_label(r.get("action",{}).get("action") or r.get("rec",{}).get("rating"))
        call_color=_action_color(t, action)
        ax.text(.05,.93,"QUANT CLIENT BRIEF",color=t["muted"],fontsize=10,weight="bold"); ax.text(.95,.93,str(r.get("date", ""))[:16],ha="right",color=t["muted"],fontsize=9)
        ax.text(.05,.82,self.ticker,color=t["text"],fontsize=44,weight="bold"); ax.text(.23,.835,action,color=call_color,fontsize=20,weight="bold")
        ax.text(.05,.755,f"Quant Score  {score:.0f}/100" if score is not None else "Quant Score  N/A",color=t["primary"],fontsize=17,weight="bold")
        ax.text(.95,.82,f"KHUNG NẮM GIỮ  {m['horizon']} PHIÊN",ha="right",color=t["muted"],fontsize=10,weight="bold")
        ax.plot([.05,.95],[.715,.715],color=t["grid"],lw=1)

        cards=[("ĐIỂM MUA",_price(m["entry"]),t["primary"]),("CẮT LỖ",f"{_price(m['stop'])}\n{_pct(m['downside'])}",t["negative"]),("MỤC TIÊU 1",f"{_price(m['tp1'])}\n{_pct(m['upside1'])}",t["positive"]),("MỤC TIÊU 2",f"{_price(m['tp2'])}\n{_pct(m['upside2'])}",t["positive"]),("R:R",f"{m['rr']:.2f}x" if m["rr"] is not None else "N/A",t["text"])]
        for i,(lab,val,color) in enumerate(cards):
            x=.05+i*.183; ax.add_patch(FancyBboxPatch((x,.56),.165,.12,boxstyle="round,pad=.008,rounding_size=.008",fc=t["panel"],ec=t["grid"],lw=.8)); ax.text(x+.012,.645,lab,color=t["muted"],fontsize=8,weight="bold"); ax.text(x+.012,.585,val,color=color,fontsize=14,weight="bold",va="center")

        ax.text(.05,.505,"KẾT QUẢ MÔ HÌNH",color=t["text"],fontsize=11,weight="bold")
        model=[("Ensemble kỳ vọng",_pct(m["ensemble"])),("Monte Carlo trung vị",_pct(m["mc_median"])),("P(lợi nhuận > 0)",_percent(m["prob_up"])),("P(TP1 trước SL)",_percent(m["p_tp_before_sl"])),("Đồng thuận model",_percent(m["agreement"]))]
        for i,(lab,val) in enumerate(model): ax.text(.05,.455-i*.058,lab,color=t["muted"],fontsize=9); ax.text(.25,.455-i*.058,val,color=t["text"],fontsize=11,weight="bold")

        ax.text(.52,.505,"RỦI RO & ĐỘ TIN CẬY",color=t["text"],fontsize=11,weight="bold")
        risks=[("Biến động năm",_percent(m["vol"])),("Max Drawdown",_pct(m["max_dd"])),("VaR 95% / ngày",_pct(m["var"])),("CVaR 95% / ngày",_pct(m["cvar"])),("Expected value",f"{m['ev_r']:+.3f}R/trade" if m["ev_r"] is not None else "N/A — chưa calibration OOS")]
        for i,(lab,val) in enumerate(risks): ax.text(.52,.455-i*.058,lab,color=t["muted"],fontsize=9); ax.text(.72,.455-i*.058,val,color=t["text"],fontsize=11,weight="bold")

        hmm=_plain_label(r.get("hmm",{}).get("current")); regime=_plain_label(r.get("rec",{}).get("vni_regime")); timing=_plain_label(r.get("fcast",{}).get("timing"))
        ax.add_patch(FancyBboxPatch((.05,.105),.90,.105,boxstyle="round,pad=.01,rounding_size=.008",fc=t["panel"],ec=t["grid"],lw=.8))
        ax.text(.065,.178,"BỐI CẢNH",color=t["muted"],fontsize=8,weight="bold"); ax.text(.15,.178,f"HMM: {hmm or 'N/A'}   |   VNINDEX: {regime or 'N/A'}",color=t["text"],fontsize=10,weight="bold")
        ax.text(.065,.138,"THỜI ĐIỂM",color=t["muted"],fontsize=8,weight="bold"); ax.text(.15,.138,timing or "N/A",color=call_color,fontsize=9,weight="bold")
        ax.text(.05,.055,f"Dữ liệu: {r.get('range','N/A')}  |  Horizon: {m['horizon']} phiên  |  Monte Carlo: {m['simulations']:,} mô phỏng",color=t["muted"],fontsize=8)
        ax.text(.95,.025,"Thông tin định lượng tham khảo; không bảo đảm lợi nhuận. Model agreement không phải xác suất thắng.",ha="right",color=t["muted"],fontsize=7.5)
        return self._save(fig,path)

    def generate_all(self, output_dir="output", formats=("png",)):
        folder=Path(output_dir)/self.ticker; generated=[]; skipped={}
        jobs=[("01", "Quant_Card",self.quant_card),("02","Probability_Cone",self.probability_cone),("03","Return_Distribution",self.return_distribution),("04","Regime",self.regime_chart),("05","Risk_Profile",self.risk_profile),("06","Radar",self.radar),("07","Terminal_Dashboard",self.terminal_dashboard)]
        for n,name,fn in jobs:
            for fmt in formats:
                if fmt.lower() not in {"png","pdf","svg"}: skipped[f"{name}.{fmt}"]="format không hỗ trợ"; continue
                target=folder/f"{n}_{self.ticker}_{name}.{fmt.lower()}"
                try: generated.append(fn(target)); log.info("Created %s",target)
                except (ValueError,KeyError,TypeError) as exc: skipped[name]=str(exc); log.warning("Skip %s %s: %s",self.ticker,name,exc); break
                except Exception as exc: skipped[name]=f"plot error: {exc}"; log.exception("Cannot create %s",target); break
        return {"ticker":self.ticker,"generated":generated,"skipped":skipped}


def make_risk_return_map(df: pd.DataFrame, output_path="output/07_Risk_Return_Map.png", theme=None, top_n=20, risk_col=None, return_col=None, score_col=None):
    """Create a universe map; cutoffs are cross-sectional medians."""
    t={**QUANT_THEME,**(theme or {})}; risk_col=risk_col or next((c for c in ["Volatility","ann_vol_pct","AnnVol","volatility","MaxDD","max_drawdown"] if c in df),None); return_col=return_col or next((c for c in ["EnsRet%","expected_return","NetFc%","AnnRet"] if c in df),None); score_col=score_col or next((c for c in ["Score","quant_score"] if c in df),None)
    symbol_col=next((c for c in ["Symbol","Ticker","symbol","ticker"] if c in df),None)
    if not risk_col or not return_col or not symbol_col: raise ValueError("Thiếu cột ticker/risk/expected return")
    d=df[[symbol_col,risk_col,return_col]+([score_col] if score_col else [])].copy(); d[risk_col]=pd.to_numeric(d[risk_col],errors="coerce"); d[return_col]=pd.to_numeric(d[return_col],errors="coerce"); d=d.replace([np.inf,-np.inf],np.nan).dropna(subset=[risk_col,return_col]);
    if risk_col in {"MaxDD","max_drawdown"}: d[risk_col]=d[risk_col].abs()
    if len(d)<4: raise ValueError("Cần ít nhất 4 cổ phiếu cho risk-return map")
    rc,ec=d[risk_col].median(),d[return_col].median(); fig,ax=plt.subplots(figsize=t["figure_size"],dpi=t["dpi"],facecolor=t["background"]); ax.set_facecolor(t["background"]); color=d[score_col] if score_col else t["primary"]; sc=ax.scatter(d[risk_col],d[return_col],c=color,cmap="RdYlGn",s=65,alpha=.85,edgecolor="white",linewidth=.6); ax.axvline(rc,color=t["muted"],ls="--"); ax.axhline(ec,color=t["muted"],ls="--"); ax.grid(color=t["grid"],alpha=.7); [sp.set_visible(False) for sp in ax.spines.values()]; rank=(d[return_col].rank(pct=True)-d[risk_col].rank(pct=True)).nlargest(min(top_n,len(d))).index
    for i in rank: ax.annotate(str(d.loc[i,symbol_col]),(d.loc[i,risk_col],d.loc[i,return_col]),xytext=(4,4),textcoords="offset points",fontsize=8,color=t["text"])
    ax.text(.02,.96,"ATTRACTIVE",transform=ax.transAxes,color=t["positive"],va="top",weight="bold"); ax.text(.75,.96,"AGGRESSIVE",transform=ax.transAxes,color=t["muted"],va="top"); ax.text(.02,.04,"DEFENSIVE",transform=ax.transAxes,color=t["muted"]); ax.text(.82,.04,"AVOID",transform=ax.transAxes,color=t["negative"],weight="bold"); ax.set_xlabel("RỦI RO / VOLATILITY"); ax.set_ylabel("LỢI NHUẬN KỲ VỌNG"); ax.set_title("BẢN ĐỒ RỦI RO — LỢI NHUẬN",loc="left",fontsize=22,weight="bold",color=t["text"]); Path(output_path).parent.mkdir(parents=True,exist_ok=True); fig.savefig(output_path,dpi=t["dpi"],bbox_inches="tight",facecolor=fig.get_facecolor()); plt.close(fig); return Path(output_path)


def generate_quant_visuals(ticker, output_dir="output", formats=("png",), report=None, price_data=None, theme=None):
    """Generate one ticker from supplied pipeline output, or fetch/analyze real data."""
    ticker=str(ticker).upper().strip()
    if not ticker: raise ValueError("Ticker không hợp lệ")
    if report is None or price_data is None:
        from quant import AdaptiveScorer, QuantPipeline, ScreenerBridge
        bridge=ScreenerBridge(); data=bridge.fetch_ohlcv([ticker],days=252)
        if ticker not in data: raise ValueError(f"Không có dữ liệu thật cho {ticker}")
        idx=bridge.fetch_index("VNINDEX",days=252); ex=bridge.fetch_exchange_map([ticker]); reports=QuantPipeline().batch(data,idx_df=idx,exchange_map=ex)
        report=next((x for x in reports if x.get("symbol")==ticker),None); price_data=data[ticker]
        if report is None or "error" in report: raise ValueError(f"Pipeline không phân tích được {ticker}")
    return QuantVisualizer(report,price_data,theme).generate_all(output_dir,formats)


def generate_batch_visuals(reports: Sequence[Mapping[str,Any]], price_data: Mapping[str,pd.DataFrame], summary_df=None, output_dir="output", formats=("png",), theme=None):
    results=[]
    for report in reports:
        ticker=str(report.get("symbol","")).upper(); df=price_data.get(ticker)
        try: results.append(generate_quant_visuals(ticker,output_dir,formats,report,df,theme))
        except Exception as exc: log.warning("Skip visuals %s: %s",ticker,exc); results.append({"ticker":ticker,"generated":[],"skipped":{"all":str(exc)}})
    if isinstance(summary_df,pd.DataFrame) and not summary_df.empty:
        try: make_risk_return_map(summary_df,Path(output_dir)/"07_Risk_Return_Map.png",theme=theme)
        except ValueError as exc: log.warning("Skip risk-return map: %s",exc)
    return results


def make_quant_card(report, prices, output_path, theme=None): return QuantVisualizer(report,prices,theme).quant_card(Path(output_path))
def make_probability_cone(report, prices, output_path, theme=None): return QuantVisualizer(report,prices,theme).probability_cone(Path(output_path))
def make_return_distribution(report, prices, output_path, theme=None): return QuantVisualizer(report,prices,theme).return_distribution(Path(output_path))
def make_regime_chart(report, prices, output_path, theme=None): return QuantVisualizer(report,prices,theme).regime_chart(Path(output_path))
def make_risk_profile(report, prices, output_path, theme=None): return QuantVisualizer(report,prices,theme).risk_profile(Path(output_path))
def make_quant_radar(report, prices, output_path, theme=None): return QuantVisualizer(report,prices,theme).radar(Path(output_path))
def make_terminal_dashboard(report, prices, output_path, theme=None): return QuantVisualizer(report,prices,theme).terminal_dashboard(Path(output_path))
def make_client_recommendation(report, prices, output_path, theme=None): return QuantVisualizer(report,prices,theme).client_recommendation(Path(output_path))
