
import os
import math
import time
import json
from datetime import date, datetime
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, mean_absolute_error, r2_score
from xgboost import XGBRegressor, XGBClassifier

# --- Optional: pybaseball for Statcast (batter data & park factors)

import logging
try:
    from pybaseball import playerid_lookup, statcast_batter
    run_expectancy_matrix = None  # removed in pybaseball >=2.1
    park_factors = None           # removed in pybaseball >=2.1
except Exception as e:
    logging.warning(f"⚠️ PyBaseball import issue: {e}")
    playerid_lookup = statcast_batter = run_expectancy_matrix = park_factors = None

# =============================
# Constants & small utilities
# =============================
TEAM_LIST = [
    'NEUTRAL','ARI','ATL','BAL','BOS','CHC','CIN','CLE','COL','CWS','DET','HOU','KC','LAA','LAD','MIA','MIL',
    'MIN','NYM','NYY','OAK','PHI','PIT','SD','SEA','SF','STL','TB','TEX','TOR','WSH'
]

HIT_EVENTS = {'single','double','triple','home_run'}
PA_EVENTS  = HIT_EVENTS | {'walk', 'intent_walk', 'hit_by_pitch', 'sac_fly', 'sac_bunt',
                           'field_out','strikeout','double_play','force_out','field_error','catcher_interf'}

PITCH_RESULT_FILTERS = {
    'Any': 'ANY',
    'Balls In Play': 'BIP',
    'Called Strikes': 'CALLED',
    'Swinging Strikes': 'SWING',
    'Fouls': 'FOUL',
    'Balls': 'BALL'
}

COMMON_PITCH_TYPES = [
    'FF','FT','SI','FC','FS','SL','CU','KC','CH','KN','ST','PO','EP','SC','FO',
    '4-Seam Fastball','2-Seam Fastball','Sinker','Cutter','Splitter','Slider','Curveball','Knuckle Curve','Changeup','Knuckleball','Sweeper','Eephus','Screwball','Forkball'
]

OUTCOME_LABELS: List[str] = ['OUT','BB','K','1B','2B','3B','HR']
EVENT_TO_OUTCOME = {
    'walk':'BB','intent_walk':'BB','hit_by_pitch':'BB',
    'strikeout':'K',
    'single':'1B','double':'2B','triple':'3B','home_run':'HR',
    'force_out':'OUT','field_out':'OUT','double_play':'DP','sac_fly':'SF','sac_bunt':'OUT','field_error':'OUT'
}
PA_FALLBACKS = set(['field_out','double_play','force_out','field_error','sac_fly','sac_bunt','catcher_interf'])

def bin_velocity(v):
    try:
        v = float(v)
    except Exception:
        return 'Unknown'
    if v < 90: return '<90'
    if v < 95: return '90–94'
    if v < 100: return '95–99'
    return '100+'

# =============================
# Caching helpers
# =============================
@st.cache_data(show_spinner=False)
def cached_player_lookup(first: str, last: str) -> Optional[int]:
    if playerid_lookup is None: return None
    df = playerid_lookup(last=last, first=first)
    if df.empty: return None
    df = df.sort_values('mlbam_id', ascending=False)
    try:
        return int(df.iloc[0]['mlbam_id'])
    except Exception:
        return None

def resolve_player_id(name: str) -> Optional[int]:
    name = (name or '').strip()
    if not name:
        return None
    parts = name.split()
    if len(parts) >= 2:
        return cached_player_lookup(parts[0], parts[-1])
    else:
        return cached_player_lookup('', parts[0])

@st.cache_data(show_spinner=True)
def cached_statcast_batter(mlbam_id: int, start_date: str, end_date: str) -> pd.DataFrame:
    if statcast_batter is None:
        return pd.DataFrame()
    df = statcast_batter(start_date, end_date, mlbam_id)
    needed = ['on_1b','on_2b','on_3b','p_throws','stand','events','description','type',
              'outs_when_up','balls','strikes','inning','inning_topbot','release_speed',
              'estimated_woba_using_speedangle','estimated_ba_using_speedangle','estimated_slg_using_speedangle',
              'game_year','home_team','pitcher','home_score','away_score','venue_name',
              'pitch_type','pitch_name','launch_angle','launch_speed','hit_distance_sc','game_date','player_name']
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan
    df['events'] = df['events'].fillna('')
    return df

@st.cache_data(show_spinner=False)
def cached_re_matrix():
    if run_expectancy_matrix is None:
        # Fallback: minimal 24 table approximation
        cols = ['outs','on1','on2','on3','RE']
        return pd.DataFrame(columns=cols).set_index(['outs','on1','on2','on3'])
    return run_expectancy_matrix()

@st.cache_data(show_spinner=False)
def cached_park_factors():
    if park_factors is None:
        return None
    try:
        return park_factors()
    except Exception:
        return None

# =============================
# Filters, stats, models
# =============================
def apply_pitch_result_filter(df: pd.DataFrame, code: str) -> pd.DataFrame:
    if code == 'ANY': return df
    if code == 'BIP': return df[df['type'] == 'X']
    if code == 'CALLED': return df[df['description'] == 'called_strike']
    if code == 'SWING': return df[df['description'].isin(['swinging_strike','swinging_strike_blocked'])]
    if code == 'FOUL': return df[df['description'] == 'foul']
    if code == 'BALL': return df[df['description'] == 'ball']
    return df

def filter_by_situation(df: pd.DataFrame,
                        outs: Optional[int],
                        runners_on: Tuple[Optional[bool], Optional[bool], Optional[bool]],
                        pitcher_hand: Optional[str],
                        batter_stand: Optional[str],
                        balls: Optional[int],
                        strikes: Optional[int],
                        topbot: Optional[str],
                        inn_min: Optional[int],
                        inn_max: Optional[int],
                        pitcher_id: Optional[int],
                        pitch_types: Optional[list],
                        balls_min: Optional[int],
                        balls_max: Optional[int],
                        strikes_min: Optional[int],
                        strikes_max: Optional[int],
                        velo_min: Optional[float],
                        velo_max: Optional[float],
                        pitch_result_code: str) -> pd.DataFrame:
    f = df.copy()
    if outs is not None:
        f = f[f['outs_when_up'] == outs]
    on1, on2, on3 = runners_on
    if on1 is not None:
        f = f[(pd.notna(f['on_1b'])) == bool(on1)]
    if on2 is not None:
        f = f[(pd.notna(f['on_2b'])) == bool(on2)]
    if on3 is not None:
        f = f[(pd.notna(f['on_3b'])) == bool(on3)]
    if pitcher_hand:
        f = f[f['p_throws'].fillna('') == pitcher_hand.upper()]
    if batter_stand:
        f = f[f['stand'].fillna('') == batter_stand.upper()]
    if balls is not None:
        f = f[f['balls'] == balls]
    if strikes is not None:
        f = f[f['strikes'] == strikes]
    if topbot:
        f = f[f['inning_topbot'].str.lower() == topbot.lower()]
    if inn_min is not None:
        f = f[f['inning'] >= inn_min]
    if inn_max is not None:
        f = f[f['inning'] <= inn_max]
    if pitcher_id is not None:
        f = f[f['pitcher'] == pitcher_id]
    if pitch_types:
        f = f[(f['pitch_type'].isin(pitch_types)) | (f['pitch_name'].isin(pitch_types))]
    if balls_min is not None:
        f = f[f['balls'] >= balls_min]
    if balls_max is not None:
        f = f[f['balls'] <= balls_max]
    if strikes_min is not None:
        f = f[f['strikes'] >= strikes_min]
    if strikes_max is not None:
        f = f[f['strikes'] <= strikes_max]
    if velo_min is not None:
        f = f[f['release_speed'] >= velo_min]
    if velo_max is not None:
        f = f[f['release_speed'] <= velo_max]
    f = apply_pitch_result_filter(f, pitch_result_code)
    return f.reset_index(drop=True)

def rate_stats(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty: return {}
    pa_df = df[df['events'].isin(PA_EVENTS)]
    pa = max(len(pa_df), 1)
    hits = (df['events'].isin(HIT_EVENTS)).sum()
    singles = (df['events'] == 'single').sum()
    doubles = (df['events'] == 'double').sum()
    triples = (df['events'] == 'triple').sum()
    hr = (df['events'] == 'home_run').sum()
    bb = (df['events'].isin({'walk','intent_walk'})).sum()
    hbp = (df['events'] == 'hit_by_pitch').sum()
    so = (df['events'] == 'strikeout').sum()
    ab = max(pa - bb - hbp, 1)
    tb = singles + 2*doubles + 3*triples + 4*hr
    avg = hits / ab
    obp = (hits + bb + hbp) / pa
    slg = tb / ab
    iso = slg - avg
    contact_rate = 1 - (so / pa)
    xba = df['estimated_ba_using_speedangle'].dropna().mean()
    xslg = df['estimated_slg_using_speedangle'].dropna().mean()
    xwoba = df['estimated_woba_using_speedangle'].dropna().mean()
    return {
        'PA': pa, 'AVG': round(avg,3), 'OBP': round(obp,3), 'SLG': round(slg,3), 'ISO': round(iso,3),
        'HR%': round(hr/pa,3), 'BB%': round(bb/pa,3), 'K%': round(so/pa,3), 'Contact%': round(contact_rate,3),
        'xBA (mean)': None if pd.isna(xba) else round(xba,3),
        'xSLG (mean)': None if pd.isna(xslg) else round(xslg,3),
        'xwOBA (mean)': None if pd.isna(xwoba) else round(xwoba,3),
    }

def get_dynamic_park_scalers(year: int, team_abbr: Optional[str], pf_table: Optional[pd.DataFrame]) -> Dict[str, float]:
    scalers = {'run': 1.0, 'hr': 1.0}
    if pf_table is None or team_abbr is None or team_abbr == 'NEUTRAL':
        return scalers
    try:
        year_col = 'season' if 'season' in pf_table.columns else ('year' if 'year' in pf_table.columns else None)
        team_col = None
        for c in ['team','home_team','Tm','HomeTeam','home']:
            if c in pf_table.columns:
                team_col = c; break
        if year_col is None or team_col is None:
            return scalers
        row = pf_table[(pf_table[year_col]==year) & (pf_table[team_col].str.upper()==team_abbr.upper())]
        if row.empty: return scalers
        row = row.iloc[0]
        run_cols = [c for c in pf_table.columns if 'run' in c.lower() or 'r_' in c.lower() or c.lower().endswith('r')]
        hr_cols  = [c for c in pf_table.columns if 'hr' in c.lower() or 'home_run' in c.lower()]
        if run_cols:
            val = row[run_cols[0]]
            if pd.notna(val) and float(val)>0: scalers['run'] = float(val)
        if hr_cols:
            val = row[hr_cols[0]]
            if pd.notna(val) and float(val)>0: scalers['hr'] = float(val)
    except Exception:
        pass
    scalers['run'] = max(0.7, min(1.3, scalers['run']))
    scalers['hr']  = max(0.7, min(1.4, scalers['hr']))
    return scalers

def apply_park_adjustments(stats: Dict[str,float], scalers: Dict[str,float]) -> Dict[str,float]:
    if not stats: return {}
    run_s, hr_s = scalers.get('run',1.0), scalers.get('hr',1.0)
    adj = stats.copy()
    if adj.get('xwOBA (mean)') is not None:
        adj['xwOBA (park-adj)'] = round(adj['xwOBA (mean)'] * run_s, 3)
    if adj.get('xSLG (mean)') is not None:
        adj['xSLG (park-adj)'] = round(adj['xSLG (mean)'] * hr_s, 3)
    if adj.get('xBA (mean)') is not None:
        adj['xBA (park-adj)'] = round(adj['xBA (mean)'] * (run_s ** 0.5), 3)
    return adj

def label_outcome(row) -> Optional[str]:
    ev = str(row.get('events') or ''); desc = str(row.get('description') or '')
    if ev in EVENT_TO_OUTCOME: return EVENT_TO_OUTCOME[ev]
    if ev == '' and row.get('type') in ('B','S'): return None
    if 'strikeout' in desc: return 'K'
    if ev in PA_FALLBACKS or ev != '': return 'OUT'
    return None

@st.cache_resource(show_spinner=False)
def cached_outcome_model(train_df: pd.DataFrame):
    m = train_df.copy()
    m['outcome'] = m.apply(label_outcome, axis=1)
    m = m[m['outcome'].notna()].copy()
    feat_num = ['balls','strikes','release_speed']
    feat_cat = ['p_throws','stand','pitch_type','pitch_name']
    keep = feat_num + feat_cat + ['outcome']
    m = m[keep].dropna(subset=['balls','strikes'])
    if m['outcome'].nunique() < 3 or len(m) < 300:
        return None
    X = m[feat_num + feat_cat]; y = m['outcome']
    pre = ColumnTransformer([('num','passthrough',feat_num),
                             ('cat',OneHotEncoder(handle_unknown='ignore', sparse_output=False), feat_cat)])
    clf = XGBClassifier(n_estimators=400, learning_rate=0.07, max_depth=5, subsample=0.9, colsample_bytree=0.9,
                        objective='multi:softprob', num_class=len(OUTCOME_LABELS), n_jobs=2, eval_metric='mlogloss', verbosity=0)
    pipe = Pipeline([('pre',pre), ('clf', clf)])
    pipe.fit(X, y)
    return pipe

def predict_outcome_probs(pipe, context: dict) -> Optional[pd.Series]:
    if pipe is None: return None
    X = pd.DataFrame([{'balls': context.get('balls',0), 'strikes': context.get('strikes',0), 'release_speed': context.get('release_speed',93.0),
                       'p_throws': context.get('p_throws','R'), 'stand': context.get('stand','R'),
                       'pitch_type': context.get('pitch_type', None), 'pitch_name': context.get('pitch_name', None)}])
    proba = pipe.predict_proba(X)[0]
    # classes are alphabetical; map back to fixed order
    classes = pipe.named_steps['clf'].classes_
    s = pd.Series(proba, index=classes)
    return s.reindex(OUTCOME_LABELS).fillna(0.0)

@st.cache_resource(show_spinner=False)
def cached_xgb_on_contact(train_df: pd.DataFrame):
    bb = train_df[(train_df['type']=='X') & train_df['launch_speed'].notna() & train_df['launch_angle'].notna() & train_df['estimated_woba_using_speedangle'].notna()].copy()
    if len(bb) < 200: return None, None, None, None
    X = bb[['launch_speed','launch_angle']].values; y = bb['estimated_woba_using_speedangle'].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(n_estimators=350, learning_rate=0.05, max_depth=4, subsample=0.9, colsample_bytree=0.9,
                         reg_alpha=0.0, reg_lambda=1.0, objective='reg:squarederror', n_jobs=2, verbosity=0)
    model.fit(Xtr, ytr); yhat = model.predict(Xte)
    return model, float(mean_absolute_error(yte, yhat)), float(r2_score(yte, yhat)), ['launch_speed','launch_angle']

def predict_xwoba_on_contact(model, df_like: pd.DataFrame):
    if model is None or df_like.empty: return None
    d = df_like[(df_like['type']=='X') & df_like['launch_speed'].notna() & df_like['launch_angle'].notna()].copy()
    if d.empty: return None
    X = d[['launch_speed','launch_angle']].values; preds = model.predict(X)
    return float(np.mean(preds)) if len(preds)>0 else None

# =============================
# Win Probability engine
# =============================
def runs_per_half_inning_default() -> float:
    return 0.50

def remaining_halves(inning: int, topbot: str) -> Tuple[int,int]:
    full_inns_left = max(9 - inning, 0)
    if topbot.lower() == 'top': return (full_inns_left, 1 + full_inns_left)
    else: return (full_inns_left, full_inns_left)

def skellam_normal_winprob(d0: int, mu_b: float, mu_o: float, continuity: float = 0.5) -> float:
    mean = mu_b - mu_o; var = mu_b + mu_o
    if var <= 1e-9: return 1.0 if d0 > 0 else (0.5 if d0 == 0 else 0.0)
    z = ((-d0 + continuity) - mean) / np.sqrt(var)
    from math import erf, sqrt
    Phi = 0.5 * (1 + erf(z / sqrt(2)))
    return float(1 - Phi)

def estimate_preplay_wp(outs: int, on1: int, on2: int, on3: int, inning: int, topbot: str,
                        score_diff: int, run_scaler: float, re_matrix) -> Dict[str,float]:
    try: re_now = float(re_matrix.loc[(outs, on1, on2, on3), 'RE'])
    except KeyError: re_now = 0.0
    b_rem, o_rem = remaining_halves(inning, topbot)
    rph = runs_per_half_inning_default() * run_scaler
    mu_b = re_now + b_rem * rph; mu_o = o_rem * rph
    if topbot.lower() == 'bot': mu_b += 0.05
    wp = skellam_normal_winprob(d0=score_diff, mu_b=mu_b, mu_o=mu_o)
    return {'RE_now': re_now, 'mu_b': mu_b, 'mu_o': mu_o, 'WP_pre': wp}

def transition_state(outs:int, b1:int, b2:int, b3:int, outcome:str):
    o, on1, on2, on3 = outs, b1, b2, b3; runs_scored = 0
    if outcome == 'OUT':
        o = min(2, o+1)
    elif outcome == 'BB':
        if on1 and on2 and on3: runs_scored += 1
        on3 = on2; on2 = on1; on1 = 1
    elif outcome == '1B':
        runs_scored += 1 if on3 else 0; on3 = on2; on2 = on1; on1 = 1
    elif outcome == '2B':
        runs_scored += (1 if on3 else 0) + (1 if on2 else 0); on3 = on1; on2 = 1; on1 = 0
    elif outcome == '3B':
        runs_scored += (1 if on3 else 0) + (1 if on2 else 0) + (1 if on1 else 0); on3 = 1; on2 = 0; on1 = 0
    elif outcome == 'HR':
        runs_scored += 1 + on1 + on2 + on3; on1 = on2 = on3 = 0
    elif outcome == 'DP':
        o = min(2, o+2); on1 = on2 = on3 = 0
    elif outcome == 'SF':
        runs_scored += 1 if on3 else 0; on3 = 0; o = min(2, o+1)
    return o, on1, on2, on3, runs_scored

def wp_post_for_outcome(current_wp: float, score_diff:int, inning:int, topbot:str,
                        outs:int, on1:int, on2:int, on3:int, outcome:str, run_scaler:float, re_matrix):
    o2, b1, b2, b3, runs_scored = transition_state(outs, on1, on2, on3, outcome)
    try: re2 = float(re_matrix.loc[(o2, b1, b2, b3), 'RE'])
    except KeyError: re2 = 0.0
    b_rem, o_rem = remaining_halves(inning, topbot)
    rph = runs_per_half_inning_default() * run_scaler
    mu_b2 = re2 + b_rem * rph; mu_o2 = o_rem * rph
    if topbot.lower() == 'bot': mu_b2 += 0.05
    d0_new = score_diff + runs_scored
    wp_post = skellam_normal_winprob(d0=d0_new, mu_b=mu_b2, mu_o=mu_o2)
    return wp_post, (wp_post - current_wp)

# =============================
# Live data (MLB stats API)
# =============================
@st.cache_data(ttl=30, show_spinner=False)
def get_today_live_games():
    try:
        today = date.today().strftime('%Y-%m-%d')
        url = f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}&languageCode=en'
        j = requests.get(url, timeout=10).json()
        games = []
        for d in j.get('dates', []):
            for g in d.get('games', []):
                if g.get('status', {}).get('abstractGameState') in ['Live', 'In Progress'] or g.get('status', {}).get('codedGameState') in ['I','P','L']:
                    desc = f"{g['teams']['away']['team']['name']} @ {g['teams']['home']['team']['name']} (gamePk {g['gamePk']})"
                    games.append((desc, g['gamePk']))
        return games
    except Exception:
        return []

@st.cache_data(ttl=5, show_spinner=False)
def get_game_feed(gamepk: int):
    try:
        url = f'https://statsapi.mlb.com/api/v1.1/game/{gamepk}/feed/live'
        return requests.get(url, timeout=10).json()
    except Exception:
        return None

def parse_live_state(feed):
    linescore = feed.get('liveData', {}).get('linescore', {})
    current = feed.get('liveData', {}).get('plays', {}).get('currentPlay', {})
    count = current.get('count', {})
    inning = int(linescore.get('currentInning', current.get('about', {}).get('inning', 1)))
    half = linescore.get('inningState', current.get('about', {}).get('halfInning', 'top')).title()
    outs = int(linescore.get('outs', 0))
    bs = linescore.get('offense', {})
    on1, on2, on3 = bool(bs.get('first')), bool(bs.get('second')), bool(bs.get('third'))
    home = feed['gameData']['teams']['home']['name']; away = feed['gameData']['teams']['away']['name']
    home_score = linescore.get('teams', {}).get('home', {}).get('runs', 0) or 0
    away_score = linescore.get('teams', {}).get('away', {}).get('runs', 0) or 0
    bat = current.get('matchup', {}).get('batter', {}).get('fullName', '')
    pit = current.get('matchup', {}).get('pitcher', {}).get('fullName', '')
    st_live = {
        'inning': inning, 'half': 'Top' if half.lower().startswith('top') else 'Bot',
        'outs': outs, 'balls': int(count.get('balls', 0)), 'strikes': int(count.get('strikes', 0)),
        'on1b': on1, 'on2b': on2, 'on3b': on3,
        'home': home, 'away': away, 'home_score': home_score, 'away_score': away_score,
        'batter': bat, 'pitcher': pit, 'batting_home': (half.lower().startswith('bot')),
        'playId': current.get('playId', None),
        'current': current
    }
    return st_live

def map_event_to_outcome(current_play) -> str:
    res = (current_play.get('result', {}) or {})
    etype = (res.get('eventType') or '').lower()
    ev = (res.get('event') or '').lower()
    for key in [etype, ev]:
        if key in EVENT_TO_OUTCOME: return EVENT_TO_OUTCOME[key]
    if 'strikeout' in etype or 'strikeout' in ev: return 'K'
    if 'home_run' in etype or 'home run' in ev: return 'HR'
    if 'double_play' in etype or 'double play' in ev: return 'DP'
    if 'sac_fly' in etype or 'sac fly' in ev: return 'SF'
    return 'OUT'

# =============================
# UI PAGES
# =============================
st.set_page_config(page_title="MLB Outcome Simulator + Live Tracker", layout="wide")

st.sidebar.title("⚾ MLB Dashboard")
page = st.sidebar.radio("Navigate", ["Live Game Tracker", "Player Situation Simulator", "Advanced Settings"], index=0)
st.sidebar.markdown("---")

# Session state for live mode
if "play_log" not in st.session_state:
    st.session_state.play_log = pd.DataFrame(columns=['ts','Inning','Half','Batter','Pitcher','Count','Result','WPA Δ','WP (Bat)','WP (Opp)','home_score','away_score','launch_speed','launch_angle','hc_x','hc_y','pitch_type','pitch_name'])
if "wp_series" not in st.session_state:
    st.session_state.wp_series = []  # list of dicts
if "last_state" not in st.session_state:
    st.session_state.last_state = None
if "last_play_id" not in st.session_state:
    st.session_state.last_play_id = None

# Common cached data
RE24 = cached_re_matrix()
PF_TABLE = cached_park_factors()

# ----------- PAGE 1: LIVE GAME TRACKER -----------
if page == "Live Game Tracker":
    st.header("🟢 Live Game Tracker")

    colA, colB, colC = st.columns([2,1,1])
    with colA:
        games = get_today_live_games()
        game_sel = st.selectbox("Select a live MLB game", options=games if games else [("No live games found", None)], format_func=lambda x: x[0] if isinstance(x, tuple) else x)
    with colB:
        refresh_s = st.number_input("Refresh (seconds)", min_value=3, max_value=60, step=1, value=10)
    with colC:
        auto = st.toggle("Auto-refresh", value=True)
        if auto:
            st_autorefresh = st.rerun  # Updated for Streamlit 1.50+
            st.query_params = {"_": str(int(time.time()))}


    if game_sel and isinstance(game_sel, tuple) and game_sel[1]:
        gamepk = game_sel[1]
        feed = get_game_feed(gamepk)
        if feed:
            live = parse_live_state(feed)
            # Status header
            st.markdown(f"**{live['away']} {live['away_score']} @ {live['home']} {live['home_score']}** — **{live['half']} {live['inning']}**, {live['outs']} out(s), count {live['balls']}-{live['strikes']} | Runners: {''.join([b for b,flag in zip(['1','2','3'], [live['on1b'],live['on2b'],live['on3b']]) if flag]) or '—'}")

            # Compute WPA for new play against last snapshot
            if live.get('playId') and live['playId'] != st.session_state.last_play_id:
                # Pre-WP from last state
                if st.session_state.last_state is not None and not RE24.empty:
                    score_diff = (live['home_score'] if live['batting_home'] else live['away_score']) - (live['away_score'] if live['batting_home'] else live['home_score'])
                    pre = estimate_preplay_wp(
                        outs=st.session_state.last_state['outs'],
                        on1=int(st.session_state.last_state['on1b']),
                        on2=int(st.session_state.last_state['on2b']),
                        on3=int(st.session_state.last_state['on3b']),
                        inning=st.session_state.last_state['inning'],
                        topbot=st.session_state.last_state['half'],
                        score_diff=score_diff,
                        run_scaler=1.0,
                        re_matrix=RE24
                    )['WP_pre']
                    outcome = map_event_to_outcome(live['current'])
                    wp_post, wpa = wp_post_for_outcome(
                        current_wp=pre,
                        score_diff=score_diff,
                        inning=st.session_state.last_state['inning'],
                        topbot=st.session_state.last_state['half'],
                        outs=st.session_state.last_state['outs'],
                        on1=int(st.session_state.last_state['on1b']), on2=int(st.session_state.last_state['on2b']), on3=int(st.session_state.last_state['on3b']),
                        outcome=outcome,
                        run_scaler=1.0, re_matrix=RE24
                    )
                    # Append to timeline
                    st.session_state.wp_series.append({'n': len(st.session_state.wp_series)+1, 'wp_bat': wp_post, 'wp_opp': 1.0 - wp_post, 'desc': live.get('current',{}).get('result',{}).get('description','')})
                    # Append to play log
                    ev = live.get('current', {}).get('hitData', {}) or {}
                    coords = ev.get('coordinates', {}) if isinstance(ev, dict) else {}
                    row = {
                        'ts': datetime.utcnow().isoformat(),
                        'Inning': st.session_state.last_state['inning'],
                        'Half': st.session_state.last_state['half'],
                        'Batter': live.get('current',{}).get('matchup',{}).get('batter',{}).get('fullName',''),
                        'Pitcher': live.get('current',{}).get('matchup',{}).get('pitcher',{}).get('fullName',''),
                        'Count': f"{live.get('current',{}).get('count',{}).get('balls',0)}-{live.get('current',{}).get('count',{}).get('strikes',0)}",
                        'Result': live.get('current',{}).get('result',{}).get('description',''),
                        'WPA Δ': wpa,
                        'WP (Bat)': wp_post,
                        'WP (Opp)': 1.0 - wp_post,
                        'home_score': live['home_score'], 'away_score': live['away_score'],
                        'launch_speed': ev.get('launchSpeed', None),
                        'launch_angle': ev.get('launchAngle', None),
                        'hc_x': coords.get('coordX', None),
                        'hc_y': coords.get('coordY', None),
                        'pitch_type': live.get('current',{}).get('details',{}).get('type',{}).get('code', None),
                        'pitch_name': live.get('current',{}).get('details',{}).get('type',{}).get('description', None),
                    }
                    st.session_state.play_log = pd.concat([st.session_state.play_log, pd.DataFrame([row])], ignore_index=True)

                # update last play id and snapshot of state (pre next play)
                st.session_state.last_play_id = live['playId']
                st.session_state.last_state = {
                    'inning': live['inning'], 'half': live['half'], 'outs': live['outs'],
                    'on1b': live['on1b'], 'on2b': live['on2b'], 'on3b': live['on3b']
                }

            # --------- Charts: Dual WP timeline
            st.subheader("Win Probability Timeline (Both Teams)")
            if st.session_state.wp_series:
                df = pd.DataFrame(st.session_state.wp_series)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['n'], y=df['wp_bat'], mode='lines+markers', name='Batting WP'))
                fig.add_trace(go.Scatter(x=df['n'], y=-df['wp_opp'], mode='lines+markers', name='Opponent WP (mirrored)'))
                for i in range(len(df)):
                    if i % 2 == 0:
                        fig.add_vrect(x0=i-0.5, x1=i+0.5, fillcolor='lightgrey', opacity=0.1, line_width=0)
                fig.update_layout(yaxis=dict(tickformat='.0%'), xaxis_title='Play #', yaxis_title='WP (opp mirrored)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No plays captured yet. Waiting for first update...")

            # --------- Play-by-Play log with spray thumbnails
            st.subheader("Play-by-Play (Live)")
            if not st.session_state.play_log.empty:
                view = st.session_state.play_log.tail(25).copy()
                for _, row in view.iterrows():
                    st.markdown(f"**{row['Half']} {int(row['Inning'])}** — {row['Batter']} vs {row['Pitcher']} | Count {row['Count']} | _{row['Result']}_ | WPA Δ {row['WPA Δ']:+.2%} | WP {row['WP (Bat)']:.1%}/{row['WP (Opp)']:.1%} | Score {int(row['away_score'])}-{int(row['home_score'])}")
                    if pd.notna(row['hc_x']) and pd.notna(row['hc_y']):
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=[row['hc_x']], y=[row['hc_y']], mode='markers',
                                                 hovertext=[f"EV {row['launch_speed']} | LA {row['launch_angle']} | {row['pitch_name'] or row['pitch_type'] or ''}"]))
                        fig.update_layout(width=420, height=240, margin=dict(l=10,r=10,t=10,b=10), xaxis_title='', yaxis_title='')
                        st.plotly_chart(fig, use_container_width=False)
                    st.markdown("---")
            else:
                st.write("_Waiting for plays..._")

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Clear Log"):
                    st.session_state.play_log = st.session_state.play_log.iloc[0:0].copy()
                    st.session_state.wp_series = []
                    st.session_state.last_state = None
                    st.session_state.last_play_id = None
                    st.success("Cleared live log and timeline.")

# ----------- PAGE 2: PLAYER SITUATION SIMULATOR -----------
elif page == "Player Situation Simulator":
    st.header("🔍 Player Situation Simulator")

    with st.sidebar.expander("Player & Date Range", expanded=True):
        batter_name = st.text_input("Batter", "Shohei Ohtani")
        pitcher_name = st.text_input("Pitcher (optional)", "")
        h2h = st.checkbox("Head-to-Head only", value=False)
        start_date = st.text_input("Start date (YYYY-MM-DD)", "2022-03-01")
        end_date = st.text_input("End date (YYYY-MM-DD)", str(date.today()))

    with st.sidebar.expander("Situation & Park"):
        outs = st.selectbox("Outs", options=[0,1,2], index=0)
        on1 = st.selectbox("1B", options=["Any","Empty","Runner"], index=0)
        on2 = st.selectbox("2B", options=["Any","Empty","Runner"], index=0)
        on3 = st.selectbox("3B", options=["Any","Empty","Runner"], index=0)
        topbot = st.selectbox("Inning Half", options=["Any","Top","Bot"], index=0)
        inn_min = st.number_input("Inning ≥", min_value=1, max_value=20, value=1)
        inn_max = st.number_input("Inning ≤", min_value=1, max_value=20, value=9)
        park = st.selectbox("Park (for park factors)", options=TEAM_LIST, index=0)

    with st.sidebar.expander("Advanced Filters"):
        balls_eq = st.selectbox("Balls (exact)", options=[None,0,1,2,3], index=0)
        strikes_eq = st.selectbox("Strikes (exact)", options=[None,0,1,2], index=0)
        balls_min = st.selectbox("Balls ≥", options=[None,0,1,2,3], index=0)
        balls_max = st.selectbox("Balls ≤", options=[None,0,1,2,3], index=0)
        strikes_min = st.selectbox("Strikes ≥", options=[None,0,1,2], index=0)
        strikes_max = st.selectbox("Strikes ≤", options=[None,0,1,2], index=0)
        velo_min = st.number_input("Velo ≥ (mph)", min_value=0.0, max_value=110.0, value=0.0, step=0.5)
        velo_max = st.number_input("Velo ≤ (mph)", min_value=0.0, max_value=110.0, value=110.0, step=0.5)
        pitch_result = st.selectbox("Pitch Result filter", options=list(PITCH_RESULT_FILTERS.keys()), index=0)
        pitch_types = st.multiselect("Pitch Types", options=COMMON_PITCH_TYPES)

    with st.sidebar.expander("Heatmap Settings"):
        show_heatmap = st.checkbox("Show Heatmap", value=True)
        heatmap_metric = st.selectbox("Heatmap Metric", options=["Expected WPA","HR%","xwOBA"], index=0)

    # Resolve player ids
    batter_id = resolve_player_id(batter_name) if playerid_lookup else None
    pitcher_id = resolve_player_id(pitcher_name) if (h2h and pitcher_name.strip() and playerid_lookup) else None

    if playerid_lookup is None or statcast_batter is None:
        st.error("pybaseball is required on the server to load Statcast data. Please ensure it is installed.")
    else:
        if batter_id is None:
            st.warning("Batter not found.")
        else:
            df = cached_statcast_batter(batter_id, start_date, end_date)
            if df.empty:
                st.info("No Statcast rows for that range.")
            else:
                filt = filter_by_situation(
                    df,
                    outs=outs,
                    runners_on=(None if on1=='Any' else (on1=='Runner'),
                                None if on2=='Any' else (on2=='Runner'),
                                None if on3=='Any' else (on3=='Runner')),
                    pitcher_hand=None, batter_stand=None,
                    balls=balls_eq, strikes=strikes_eq,
                    topbot=None if topbot=='Any' else topbot,
                    inn_min=inn_min, inn_max=inn_max,
                    pitcher_id=(pitcher_id if (h2h and pitcher_id is not None) else None),
                    pitch_types=pitch_types if pitch_types else None,
                    balls_min=balls_min, balls_max=balls_max,
                    strikes_min=strikes_min, strikes_max=strikes_max,
                    velo_min=(velo_min if velo_min>0 else None),
                    velo_max=(velo_max if velo_max<110 else None),
                    pitch_result_code=PITCH_RESULT_FILTERS[pitch_result]
                )

                st.markdown(f"**Rows matched:** {len(filt)}")
                if len(filt)==0:
                    st.stop()

                # Rate stats + park
                stats = rate_stats(filt)
                try:
                    yr = int(pd.to_numeric(filt['game_year'].dropna(), errors='coerce').median())
                except Exception:
                    yr = int(start_date[:4])
                scalers = get_dynamic_park_scalers(yr, None if park=='NEUTRAL' else park, PF_TABLE)
                stats_adj = apply_park_adjustments(stats, scalers)

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("AVG", stats.get('AVG'))
                    st.metric("OBP", stats.get('OBP'))
                    st.metric("SLG", stats.get('SLG'))
                with c2:
                    st.metric("HR%", stats.get('HR%'))
                    st.metric("BB%", stats.get('BB%'))
                    st.metric("K%", stats.get('K%'))
                with c3:
                    st.metric("xwOBA", stats.get('xwOBA (mean)'))
                    if 'xwOBA (park-adj)' in stats_adj:
                        st.metric("xwOBA (adj)", stats_adj.get('xwOBA (park-adj)'))

                # Models
                outcome_pipe = cached_outcome_model(df)
                # WP context (simple default without scoreboard inputs here)
                re = RE24
                outcome_wpas = None
                if not re.empty and None not in (on1, on2, on3):
                    outs0 = outs
                    on1v = int(on1=='Runner') if on1!='Any' else 0
                    on2v = int(on2=='Runner') if on2!='Any' else 0
                    on3v = int(on3=='Runner') if on3!='Any' else 0
                    pre = estimate_preplay_wp(outs0, on1v, on2v, on3v, max(1, inn_min), 'Top', 0, scalers.get('run',1.0), re)
                    # simple WPA table for outcomes from neutral score
                    outcomes_for_wp = ['OUT','BB','1B','2B','3B','HR','DP','SF']
                    wpas_map = {}
                    for oc in outcomes_for_wp:
                        wp_post, wpa = wp_post_for_outcome(pre['WP_pre'], 0, max(1, inn_min), 'Top', outs0, on1v, on2v, on3v, oc, scalers.get('run',1.0), re)
                        wpas_map[oc] = wpa
                    outcome_wpas = wpas_map

                # Per-pitch-type × velocity bin table
                comp_rows = []
                if outcome_pipe is not None:
                    g = filt.copy()
                    g['pitch_display'] = g['pitch_name'].fillna(g['pitch_type']).fillna('Unknown')
                    g['velo_bin'] = g['release_speed'].apply(bin_velocity)
                    pt_counts = g['pitch_display'].value_counts()
                    valid_pts = [pt for pt,c in pt_counts.items() if c >= 40]
                    for pt in valid_pts:
                        sub_pt = g[g['pitch_display']==pt]
                        pitch_type_val = sub_pt['pitch_type'].dropna().mode().iloc[0] if sub_pt['pitch_type'].notna().any() else None
                        pitch_name_val = sub_pt['pitch_name'].dropna().mode().iloc[0] if sub_pt['pitch_name'].notna().any() else None
                        base_ctx = {
                            'balls': sub_pt['balls'].dropna().median() if sub_pt['balls'].notna().any() else 0,
                            'strikes': sub_pt['strikes'].dropna().median() if sub_pt['strikes'].notna().any() else 0,
                            'release_speed': sub_pt['release_speed'].dropna().median() if sub_pt['release_speed'].notna().any() else 93.0,
                            'p_throws': sub_pt['p_throws'].dropna().mode().iloc[0] if sub_pt['p_throws'].notna().any() else (filt['p_throws'].dropna().mode().iloc[0] if filt['p_throws'].notna().any() else 'R'),
                            'stand': sub_pt['stand'].dropna().mode().iloc[0] if sub_pt['stand'].notna().any() else (filt['stand'].dropna().mode().iloc[0] if filt['stand'].notna().any() else 'R'),
                            'pitch_type': pitch_type_val,
                            'pitch_name': pitch_name_val
                        }
                        for vb, sub_bin in sub_pt.groupby('velo_bin'):
                            if len(sub_bin) < 25: continue
                            vb_mid = {'<90': 88, '90–94': 92, '95–99': 97, '100+': 101}.get(vb, base_ctx['release_speed'])
                            ctx = base_ctx.copy(); ctx['release_speed'] = vb_mid
                            probs = predict_outcome_probs(outcome_pipe, ctx)
                            if probs is None: continue
                            exp_wpa = None
                            if outcome_wpas is not None:
                                wpa_map = outcome_wpas.copy()
                                if 'K' not in wpa_map and 'OUT' in wpa_map:
                                    wpa_map['K'] = wpa_map['OUT']
                                exp_wpa = float((probs * pd.Series(wpa_map)).sum())
                            sub_bin_bip = sub_bin[(sub_bin['type']=='X') & sub_bin['estimated_woba_using_speedangle'].notna()]
                            xwoba_emp = float(sub_bin_bip['estimated_woba_using_speedangle'].mean()) if not sub_bin_bip.empty else None
                            comp_rows.append({'Pitch Type': pt, 'Velo Bin': vb, 'HR%': float(probs.get('HR',0.0)),
                                              'Expected WPA': exp_wpa, 'xwOBA': xwoba_emp, 'Samples': int(len(sub_bin))})

                if comp_rows:
                    comp_df = pd.DataFrame(comp_rows)
                    comp_df['X'] = comp_df['Pitch Type'] + ' (' + comp_df['Velo Bin'] + ')'
                    order = (comp_df.groupby('X')['Samples'].sum().sort_values(ascending=False).index.tolist())
                    # Grouped bar (HR%)
                    fig_grouped = go.Figure()
                    fig_grouped.add_trace(go.Bar(name='HR%', x=order, y=[comp_df.loc[comp_df['X']==x, 'HR%'].mean() for x in order]))
                    fig_grouped.update_layout(barmode='group', yaxis=dict(tickformat='.0%'), title='HR% by Pitch Type × Velocity Bin')
                    st.plotly_chart(fig_grouped, use_container_width=True)

                    # Heatmap
                    if show_heatmap:
                        metric = heatmap_metric
                        z_col = 'Expected WPA' if metric == 'Expected WPA' else ('HR%' if metric == 'HR%' else 'xwOBA')
                        pivot = comp_df.pivot_table(index='Pitch Type', columns='Velo Bin', values=z_col, aggfunc='mean')
                        cols_order = ['<90','90–94','95–99','100+']
                        pivot = pivot.reindex(columns=[c for c in cols_order if c in pivot.columns])
                        fig_heat = go.Figure(data=go.Heatmap(z=pivot.values, x=list(pivot.columns), y=list(pivot.index),
                                                             coloraxis='coloraxis',
                                                             hovertemplate='Pitch: %{y}<br>Velo: %{x}<br>'+metric+': %{z}<extra></extra>'))
                        fig_heat.update_layout(title=f'{metric} Heatmap — Pitch Type × Velocity Bin', coloraxis_colorscale='RdBu')
                        st.plotly_chart(fig_heat, use_container_width=True)
                else:
                    st.info("Not enough balanced data to build per-pitch-type visuals. Widen the date range.")

# ----------- PAGE 3: ADVANCED SETTINGS -----------
else:
    st.header("⚙️ Advanced Settings")
    st.write("Caching speeds up repeated lookups and model training.")
    if st.button("Clear all Streamlit caches"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cleared cache.")
    st.markdown("**Dependencies present:**")
    st.code("\n".join(sorted(['streamlit','plotly','pybaseball','pandas','numpy','xgboost','scikit-learn','requests'])), language='text')
