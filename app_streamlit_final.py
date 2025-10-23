import logging
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import streamlit as st
from datetime import date

# =============================================================
# Safe PyBaseball imports (v2.1.0 compatible)
# =============================================================
import logging
try:
    import pybaseball
    from pybaseball import playerid_lookup, statcast_batter
    run_expectancy_matrix = None
    park_factors = None
    st.write("✅ PyBaseball loaded successfully!")  # quick verification
except Exception as e:
    logging.error(f"❌ PyBaseball import failed: {e}")
    playerid_lookup = statcast_batter = run_expectancy_matrix = park_factors = None

# =============================================================
# Streamlit App Setup
# =============================================================
st.set_page_config(page_title="MLB Outcome Simulator", layout="wide")
st.title("⚾ MLB Outcome Simulator Dashboard")

# Sidebar navigation
page = st.sidebar.radio("Navigate", ["Live Game Tracker", "Player Simulator"], index=0)

# =============================================================
# Helper Functions
# =============================================================
def get_live_games():
    try:
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date.today().strftime('%Y-%m-%d')}"
        j = requests.get(url, timeout=10).json()
        games = []
        for d in j.get('dates', []):
            for g in d.get('games', []):
                if g.get('status', {}).get('abstractGameState') in ['Live', 'In Progress']:
                    desc = f"{g['teams']['away']['team']['name']} @ {g['teams']['home']['team']['name']}"
                    games.append((desc, g['gamePk']))
        return games
    except Exception as e:
        logging.warning(f"Error fetching live games: {e}")
        return []

def get_game_feed(gamepk):
    try:
        url = f"https://statsapi.mlb.com/api/v1.1/game/{gamepk}/feed/live"
        return requests.get(url, timeout=10).json()
    except Exception as e:
        logging.warning(e)
        return None

# =============================================================
# Live Game Tracker Page
# =============================================================
if page == "Live Game Tracker":
    st.header("🟢 Live Game Tracker")

    games = get_live_games()
    if not games:
        st.warning("No live games found right now.")
    else:
        game = st.selectbox("Select a Live Game", games, format_func=lambda x: x[0])
        if game:
            feed = get_game_feed(game[1])
            if feed:
                linescore = feed.get('liveData', {}).get('linescore', {})
                inning = linescore.get('currentInning', 1)
                state = linescore.get('inningState', 'Top')
                outs = linescore.get('outs', 0)
                home_score = linescore.get('teams', {}).get('home', {}).get('runs', 0)
                away_score = linescore.get('teams', {}).get('away', {}).get('runs', 0)

                st.markdown(f"### {state} {inning} — Outs: {outs}")
                st.markdown(f"### Score: {away_score} - {home_score}")

                plays = feed.get("liveData", {}).get("plays", {}).get("allPlays", [])
                if plays:
                    df = pd.DataFrame([{
                        "Inning": p.get("about", {}).get("inning"),
                        "Batter": p.get("matchup", {}).get("batter", {}).get("fullName"),
                        "Pitcher": p.get("matchup", {}).get("pitcher", {}).get("fullName"),
                        "Result": p.get("result", {}).get("event"),
                        "Description": p.get("result", {}).get("description")
                    } for p in plays])
                    st.dataframe(df.tail(20))
                else:
                    st.info("Waiting for play-by-play data...")
                    
# =============================================================
# Updated PlayerID lookup patch (new Chadwick CSV structure)
# =============================================================
import io

def safe_playerid_lookup(first, last):
    """
    Updated version — works with new Chadwick 'people.csv'
    which now has a single 'name_display_first_last' column.
    """
    try:
        url = "https://raw.githubusercontent.com/chadwickbureau/register/master/data/people.csv"
        data = requests.get(url, timeout=10).content
        df = pd.read_csv(io.BytesIO(data))
        df.columns = df.columns.str.lower()

        # Try to find by combined display name
        if "name_display_first_last" in df.columns:
            mask = df["name_display_first_last"].str.contains(f"{first}.*{last}", case=False, na=False)
        else:
            # fallback if they re-add separate columns
            mask = (
                df.get("name_first", pd.Series("", index=df.index))
                .str.contains(first, case=False, na=False)
            ) & (
                df.get("name_last", pd.Series("", index=df.index))
                .str.contains(last, case=False, na=False)
            )

        result = df.loc[mask, ["key_mlbam", "name_display_first_last"]].dropna(subset=["key_mlbam"])
        return result
    except Exception as e:
        st.warning(f"⚠️ Custom player lookup failed: {e}")
        return pd.DataFrame(columns=["key_mlbam", "name_display_first_last"])

# =============================================================
# Player Situation Simulator Page
# =============================================================
if page == "Player Simulator":
    st.header("🔍 Player Situation Simulator")

    batter_name = st.text_input("Enter Batter Name", "Shohei Ohtani")
    start_date = st.text_input("Start Date", "2022-03-01")
    end_date = st.text_input("End Date", str(date.today()))

    if playerid_lookup and statcast_batter:
        try:
            first, last = batter_name.split()[0], batter_name.split()[-1]
            df_lookup = safe_playerid_lookup(first, last)


            if df_lookup.empty:
                st.warning("No player found with that name.")
            else:
                bid = int(df_lookup.iloc[0]['key_mlbam'])
                df = statcast_batter(start_date, end_date, bid)
                st.success(f"Loaded {len(df)} Statcast rows for {batter_name}.")

                # Basic visualizations
                if not df.empty and "launch_angle" in df.columns and "launch_speed" in df.columns:
                    fig = px.scatter(
                        df, x="launch_angle", y="launch_speed",
                        color="events", title="Batted Ball Profile",
                        hover_data=["pitch_name", "description"]
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    heatmap_data = df.pivot_table(index="pitch_name", columns="events", values="launch_speed", aggfunc="mean")
                    fig2 = px.imshow(heatmap_data, text_auto=True, color_continuous_scale="Viridis", title="Average EV by Pitch Type and Event")
                    st.plotly_chart(fig2, use_container_width=True)

        except Exception as e:
            st.error(f"⚠️ Error loading Statcast data: {e}")
    else:
        st.error("pybaseball is required on the server to load Statcast data. Please ensure it is installed.")

st.sidebar.markdown("---")
st.sidebar.caption("Built with ❤️ by ChatGPT + Streamlit + pybaseball")
