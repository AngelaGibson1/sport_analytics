import pandas as pd
import pybaseball as pyb
import requests
import pickle
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import numpy as np

from .config import Settings
from .utils import team_normalizer

# Enable pybaseball caching
pyb.cache.enable()

class DataManager:
    """
    Centralized data collection and management class.
    Handles all interactions with external APIs (API-Sports) and data sources (pybaseball).
    """
    verbose = False  # Set to True for debug output



    def __init__(self, settings: Settings):
        """Initialize DataManager with configuration and API session."""
        self.settings = settings
        
        # Initialize session for API-Sports requests
        self.session = requests.Session()
        
        # Direct API-Sports authentication (not RapidAPI)
        self.session.headers.update({
            'x-apisports-key': self.settings.API_SPORTS_KEY
        })
        
        # Team name mapping from abbreviations to full names for Statcast matching
        self.team_name_mapping = {
            'ARI': 'Arizona Diamondbacks', 'ATL': 'Atlanta Braves', 'BAL': 'Baltimore Orioles',
            'BOS': 'Boston Red Sox', 'CHC': 'Chicago Cubs', 'CHW': 'Chicago White Sox',
            'CIN': 'Cincinnati Reds', 'CLE': 'Cleveland Guardians', 'COL': 'Colorado Rockies',
            'DET': 'Detroit Tigers', 'HOU': 'Houston Astros', 'KCR': 'Kansas City Royals',
            'LAA': 'Los Angeles Angels', 'LAD': 'Los Angeles Dodgers', 'MIA': 'Miami Marlins',
            'MIL': 'Milwaukee Brewers', 'MIN': 'Minnesota Twins', 'NYM': 'New York Mets',
            'NYY': 'New York Yankees', 'OAK': 'Oakland Athletics', 'PHI': 'Philadelphia Phillies',
            'PIT': 'Pittsburgh Pirates', 'SDP': 'San Diego Padres', 'SEA': 'Seattle Mariners',
            'SFG': 'San Francisco Giants', 'STL': 'St. Louis Cardinals', 'TBR': 'Tampa Bay Rays',
            'TEX': 'Texas Rangers', 'TOR': 'Toronto Blue Jays', 'WSN': 'Washington Nationals',
            # Alternative abbreviations
            'TB': 'Tampa Bay Rays', 'SD': 'San Diego Padres', 'SF': 'San Francisco Giants',
            'KC': 'Kansas City Royals', 'LA': 'Los Angeles Dodgers', 'WAS': 'Washington Nationals',
            'CWS': 'Chicago White Sox', 'WSH': 'Washington Nationals'
        }
        
        # FIXED: Deterministic reverse mapping to avoid collisions
        self.full_name_to_abbrev = {
            'Arizona Diamondbacks': 'ARI', 'Atlanta Braves': 'ATL', 'Baltimore Orioles': 'BAL',
            'Boston Red Sox': 'BOS', 'Chicago Cubs': 'CHC', 'Chicago White Sox': 'CHW',
            'Cincinnati Reds': 'CIN', 'Cleveland Guardians': 'CLE', 'Colorado Rockies': 'COL',
            'Detroit Tigers': 'DET', 'Houston Astros': 'HOU', 'Kansas City Royals': 'KCR',
            'Los Angeles Angels': 'LAA', 'Los Angeles Dodgers': 'LAD', 'Miami Marlins': 'MIA',
            'Milwaukee Brewers': 'MIL', 'Minnesota Twins': 'MIN', 'New York Mets': 'NYM',
            'New York Yankees': 'NYY', 'Oakland Athletics': 'OAK', 'Philadelphia Phillies': 'PHI',
            'Pittsburgh Pirates': 'PIT', 'San Diego Padres': 'SDP', 'Seattle Mariners': 'SEA',
            'San Francisco Giants': 'SFG', 'St. Louis Cardinals': 'STL', 'Tampa Bay Rays': 'TBR',
            'Texas Rangers': 'TEX', 'Toronto Blue Jays': 'TOR', 'Washington Nationals': 'WSN'
        }
        
        # Initialize enhanced normalizer with pitcher roster
        self._initialize_enhanced_normalizer()
        
        print("✅ DataManager initialized with API-Sports integration.")
        print(f"    Using endpoint: {self.settings.API_SPORTS_BASE_URL}")
        print(f"    API Key configured: {'✓' if self.settings.API_SPORTS_KEY else '✗'}")
        
        # Show normalizer stats
        if hasattr(self, 'normalizer') and self.normalizer:
            stats = self.normalizer.get_stats()
            print(f"    Enhanced normalizer: {stats['pitcher_stats']['total_known_pitchers']} known pitchers")

    def _initialize_enhanced_normalizer(self):
        """Initialize the enhanced normalizer with pitcher roster data."""
        try:
            # Load the pitcher roster data
            pitcher_roster_data = self._load_pitcher_roster()
            
            # Import the enhanced normalizer
            from .utils import EnhancedNormalizer, team_normalizer
            
            # Initialize with roster data
            self.normalizer = EnhancedNormalizer(pitcher_roster_data)
            
            # Keep backward compatibility
            self.team_normalizer = team_normalizer
            
            print(f"✅ Enhanced normalizer initialized")
            stats = self.normalizer.get_stats()
            print(f"   Pitcher database: {stats['pitcher_stats']['total_known_pitchers']} known pitchers")
            print(f"   Lookup entries: {stats['pitcher_stats']['total_lookup_entries']} variations")
            
        except Exception as e:
            print(f"⚠️ Could not initialize enhanced normalizer: {e}")
            # Fall back to basic normalizer
            try:
                from .utils import team_normalizer
                self.normalizer = None
                self.team_normalizer = team_normalizer
                print(f"   Using basic team normalizer only")
            except Exception as e2:
                print(f"❌ Could not initialize any normalizer: {e2}")
                self.normalizer = None
                self.team_normalizer = None

    def _load_pitcher_roster(self) -> pd.DataFrame:
        """Load pitcher roster from embedded data."""
        # This is the complete data structure from paste-2.txt
        data = {
            "American League": {
                "AL East": {
                    "Baltimore Orioles": [
                        "Kyle Bradish", "Corbin Burnes", "Grayson Rodriguez", "John Means", "Dean Kremer",
                        "Félix Bautista", "Yennier Cano", "Danny Coulombe", "Cionel Pérez", "Tyler Wells"
                    ],
                    "Boston Red Sox": [
                        "Brayan Bello", "Lucas Giolito", "Chris Sale", "Garrett Whitlock", "Tanner Houck",
                        "Kenley Jansen", "Chris Martin", "Josh Winckowski", "Brennan Bernardino", "Joely Rodríguez"
                    ],
                    "New York Yankees": [
                        "Gerrit Cole", "Carlos Rodón", "Nestor Cortes", "Clarke Schmidt", "Marcus Stroman",
                        "Clay Holmes", "Jonathan Loáisiga", "Tommy Kahnle", "Ian Hamilton", "Victor González"
                    ],
                    "Tampa Bay Rays": [
                        "Zach Eflin", "Shane McClanahan", "Tyler Glasnow", "Aaron Civale", "Shane Baz",
                        "Pete Fairbanks", "Jason Adam", "Colin Poche", "Shawn Armstrong", "Garrett Cleavinger"
                    ],
                    "Toronto Blue Jays": [
                        "Kevin Gausman", "José Berríos", "Chris Bassitt", "Yusei Kikuchi", "Alek Manoah",
                        "Jordan Romano", "Erik Swanson", "Tim Mayza", "Yimi García", "Génesis Cabrera"
                    ]
                },
                "AL Central": {
                    "Chicago White Sox": [
                        "Dylan Cease", "Michael Kopech", "Touki Toussaint", "Mike Clevinger", "Erick Fedde",
                        "Gregory Santos", "Aaron Bummer", "Bryan Shaw", "Jimmy Lambert", "Garrett Crochet"
                    ],
                    "Cleveland Guardians": [
                        "Shane Bieber", "Triston McKenzie", "Tanner Bibee", "Gavin Williams", "Cal Quantrill",
                        "Emmanuel Clase", "James Karinchak", "Trevor Stephan", "Sam Hentges", "Eli Morgan"
                    ],
                    "Detroit Tigers": [
                        "Tarik Skubal", "Matt Manning", "Casey Mize", "Reese Olson", "Alex Faedo",
                        "Alex Lange", "Jason Foley", "Will Vest", "Tyler Holton", "Beau Brieske"
                    ],
                    "Kansas City Royals": [
                        "Cole Ragans", "Brady Singer", "Jordan Lyles", "Daniel Lynch IV", "Kris Bubic",
                        "Will Smith", "James McArthur", "Taylor Clarke", "Carlos Hernández", "Austin Cox"
                    ],
                    "Minnesota Twins": [
                        "Pablo López", "Sonny Gray", "Joe Ryan", "Bailey Ober", "Kenta Maeda",
                        "Jhoan Duran", "Griffin Jax", "Caleb Thielbar", "Emilio Pagán", "Brock Stewart"
                    ]
                },
                "AL West": {
                    "Houston Astros": [
                        "Framber Valdez", "Justin Verlander", "Cristian Javier", "José Urquidy", "Hunter Brown",
                        "Ryan Pressly", "Hector Neris", "Bryan Abreu", "Ryne Stanek", "Phil Maton"
                    ],
                    "Los Angeles Angels": [
                        "Shohei Ohtani", "Patrick Sandoval", "Reid Detmers", "Griffin Canning", "Tyler Anderson",
                        "Carlos Estévez", "Matt Moore", "José Soriano", "Jaime Barría", "Jimmy Herget"
                    ],
                    "Oakland Athletics": [
                        "Paul Blackburn", "JP Sears", "Ken Waldichuk", "Luis Medina", "Mason Miller",
                        "Trevor May", "Dany Jiménez", "Zach Jackson", "Sam Moll", "Richard Lovelady"
                    ],
                    "Seattle Mariners": [
                        "Luis Castillo", "Logan Gilbert", "George Kirby", "Bryce Miller", "Bryan Woo",
                        "Andrés Muñoz", "Matt Brash", "Justin Topa", "Gabe Speier", "Tayler Saucedo"
                    ],
                    "Texas Rangers": [
                        "Jacob deGrom", "Max Scherzer", "Nathan Eovaldi", "Jon Gray", "Andrew Heaney",
                        "José Leclerc", "Josh Sborz", "Will Smith", "Aroldis Chapman", "Brock Burke"
                    ]
                }
            },
            "National League": {
                "NL East": {
                    "Atlanta Braves": [
                        "Spencer Strider", "Max Fried", "Charlie Morton", "Bryce Elder", "AJ Smith-Shawver",
                        "Raisel Iglesias", "A.J. Minter", "Joe Jiménez", "Kirby Yates", "Pierce Johnson"
                    ],
                    "Miami Marlins": [
                        "Sandy Alcantara", "Eury Pérez", "Jesús Luzardo", "Braxton Garrett", "Edward Cabrera",
                        "Tanner Scott", "Andrew Nardi", "JT Chargois", "Huascar Brazoban", "Steven Okert"
                    ],
                    "New York Mets": [
                        "Kodai Senga", "José Quintana", "Luis Severino", "Sean Manaea", "Tylor Megill",
                        "Edwin Díaz", "Adam Ottavino", "Brooks Raley", "Drew Smith", "David Robertson"
                    ],
                    "Philadelphia Phillies": [
                        "Zack Wheeler", "Aaron Nola", "Ranger Suárez", "Taijuan Walker", "Cristopher Sánchez",
                        "Craig Kimbrel", "José Alvarado", "Seranthony Domínguez", "Gregory Soto", "Matt Strahm"
                    ],
                    "Washington Nationals": [
                        "Josiah Gray", "MacKenzie Gore", "Patrick Corbin", "Jake Irvin", "Trevor Williams",
                        "Kyle Finnegan", "Hunter Harvey", "Carl Edwards Jr.", "Jordan Weems", "Mason Thompson"
                    ]
                },
                "NL Central": {
                    "Chicago Cubs": [
                        "Justin Steele", "Jameson Taillon", "Kyle Hendricks", "Jordan Wicks", "Javier Assad",
                        "Adbert Alzolay", "Mark Leiter Jr.", "Julian Merryweather", "Michael Fulmer", "Drew Smyly"
                    ],
                    "Cincinnati Reds": [
                        "Hunter Greene", "Graham Ashcraft", "Andrew Abbott", "Nick Lodolo", "Brandon Williamson",
                        "Alexis Díaz", "Lucas Sims", "Ian Gibaut", "Sam Moll", "Fernando Cruz"
                    ],
                    "Milwaukee Brewers": [
                        "Corbin Burnes", "Brandon Woodruff", "Freddy Peralta", "Wade Miley", "Adrian Houser",
                        "Devin Williams", "Joel Payamps", "Hoby Milner", "Abner Uribe", "Elvis Peguero"
                    ],
                    "Pittsburgh Pirates": [
                        "Mitch Keller", "Johan Oviedo", "Quinn Priester", "Roansy Contreras", "Luis L. Ortiz",
                        "David Bednar", "Colin Holderman", "Carmen Mlodzinski", "Ryan Borucki", "Dauri Moreta"
                    ],
                    "St. Louis Cardinals": [
                        "Sonny Gray", "Lance Lynn", "Kyle Gibson", "Miles Mikolas", "Steven Matz",
                        "Ryan Helsley", "Giovanny Gallegos", "JoJo Romero", "Andre Pallante", "Zack Thompson"
                    ]
                },
                "NL West": {
                    "Arizona Diamondbacks": [
                        "Zac Gallen", "Merrill Kelly", "Brandon Pfaadt", "Ryne Nelson", "Tommy Henry",
                        "Paul Sewald", "Kevin Ginkel", "Miguel Castro", "Joe Mantiply", "Scott McGough"
                    ],
                    "Colorado Rockies": [
                        "Kyle Freeland", "Germán Márquez", "Austin Gomber", "Ryan Feltner", "Peter Lambert",
                        "Justin Lawrence", "Tyler Kinley", "Jake Bird", "Brent Suter", "Daniel Bard"
                    ],
                    "Los Angeles Dodgers": [
                        "Walker Buehler", "Julio Urías", "Bobby Miller", "Emmet Sheehan", "Gavin Stone",
                        "Evan Phillips", "Brusdar Graterol", "Caleb Ferguson", "Alex Vesia", "Ryan Yarbrough"
                    ],
                    "San Diego Padres": [
                        "Yu Darvish", "Joe Musgrove", "Blake Snell", "Michael Wacha", "Seth Lugo",
                        "Josh Hader", "Robert Suarez", "Steven Wilson", "Tim Hill", "Nick Martinez"
                    ],
                    "San Francisco Giants": [
                        "Logan Webb", "Alex Cobb", "Ross Stripling", "Anthony DeSclafani", "Kyle Harrison",
                        "Camilo Doval", "Taylor Rogers", "Tyler Rogers", "Luke Jackson", "Scott Alexander"
                    ]
                }
            }
        }
        
        # Convert to DataFrame
        rows = []
        for league, divisions in data.items():
            for division, teams in divisions.items():
                for team, players in teams.items():
                    for player in players:
                        rows.append({
                            'League': league,
                            'Division': division,
                            'Team': team,
                            'Player': player
                        })
        
        df = pd.DataFrame(rows)
        print(f"   Loaded {len(df)} pitcher entries from {len(df['Team'].unique())} teams")
        return df



    # --- CACHING METHODS ---
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for a given key."""
        return self.settings.get_cache_path(cache_key)

    def _is_cache_valid(self, cache_path: Path, max_age_seconds: int) -> bool:
        """Check if cache file exists and is within the max age."""
        if not cache_path.exists(): 
            return False
        file_age = time.time() - cache_path.stat().st_mtime
        return file_age < max_age_seconds

    def _load_cache(self, cache_path: Path) -> Any:
        """Load data from cache file."""
        try:
            with open(cache_path, 'rb') as f: 
                return pickle.load(f)
        except Exception as e:
            print(f"⚠️ Warning: Failed to load cache {cache_path}: {e}")
            return None

    def _save_cache(self, data: Any, cache_path: Path) -> None:
        """Save data to cache file."""
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f: 
                pickle.dump(data, f)
        except Exception as e:
            print(f"⚠️ Warning: Failed to save cache {cache_path}: {e}")

    # --- API-SPORTS FETCHER ---
    def _fetch_schedule_from_api_sports(self, game_date: str) -> list:
        """
        Fetch raw games JSON from API-Sports for a date.
        Returns the list directly under 'response' or 'results'.
        """
        base = self.settings.API_SPORTS_BASE_URL.rstrip("/")
        season = game_date[:4]
        url = f"{base}/games"
        params = {
            "date": game_date,
            "league": "1",      # MLB
            "season": season,
        }
        r = self.session.get(url, params=params, timeout=15)
        r.raise_for_status()
        payload = r.json()
        return payload.get("response") or payload.get("results") or []

    # --- MLB API WITH PROBABLE PITCHERS ---
    def _fetch_schedule_from_mlb_api(self, game_date: str) -> pd.DataFrame:
        """Fetch schedule from MLB StatsAPI with probable pitchers."""
        try:
            url = "https://statsapi.mlb.com/api/v1/schedule"
            params = {
                "sportId": 1,
                "startDate": game_date,
                "endDate": game_date,
                # Request probables
                "expand": "schedule.teams,schedule.linescore",
                "hydrate": "probablePitcher",
            }

            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            games = []
            abbrev_fixes = {
                'CWS': 'CHW','KC': 'KCR','LA': 'LAD','SD': 'SDP','SF': 'SFG',
                'TB': 'TBR','WSH': 'WSN','WAS': 'WSN'
            }

            for day in data.get("dates", []):
                for game in day.get("games", []):
                    home_team = game["teams"]["home"]["team"]["name"]
                    away_team = game["teams"]["away"]["team"]["name"]

                    # Probable pitchers (if present)
                    home_pp = game["teams"]["home"].get("probablePitcher")
                    away_pp = game["teams"]["away"].get("probablePitcher")
                    home_pitcher = (home_pp.get("fullName") if home_pp else "TBD") or "TBD"
                    away_pitcher = (away_pp.get("fullName") if away_pp else "TBD") or "TBD"

                    # Abbreviation mapping
                    home_abbrev = self.full_name_to_abbrev.get(home_team, home_team)
                    away_abbrev = self.full_name_to_abbrev.get(away_team, away_team)
                    home_abbrev = abbrev_fixes.get(home_abbrev, home_abbrev)
                    away_abbrev = abbrev_fixes.get(away_abbrev, away_abbrev)

                    games.append({
                        'Date': game_date,
                        'Home': home_abbrev,
                        'Away': away_abbrev,
                        'home_pitcher': home_pitcher,
                        'away_pitcher': away_pitcher,
                        'home_team_full': home_team,
                        'away_team_full': away_team,
                        'game_pk': game.get('gamePk')  # Keep for optional hydrate fallback
                    })

            return pd.DataFrame(games) if games else pd.DataFrame()

        except Exception as e:
            print(f"❌ MLB StatsAPI error: {e}")
            return pd.DataFrame()

    # --- FALLBACK HYDRATE FOR TBD PITCHERS ---
    def _fill_probables_from_game_details(self, df: pd.DataFrame) -> pd.DataFrame:
        """For rows still TBD, hydrate from the game details endpoint."""
        if df.empty or 'game_pk' not in df.columns:
            return df

        tbd_mask = (df['home_pitcher'] == 'TBD') | (df['away_pitcher'] == 'TBD')
        to_fill = df[tbd_mask].copy()
        if to_fill.empty:
            return df

        filled = 0
        for idx, row in to_fill.iterrows():
            game_pk = row.get('game_pk')
            if not game_pk:
                continue
            try:
                url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
                params = {"hydrate": "probablePitcher"}
                r = requests.get(url, params=params, timeout=10)
                r.raise_for_status()
                data = r.json()

                # Check gameData > probablePitchers
                home_pp = data.get('gameData', {}).get('probablePitchers', {}).get('home')
                away_pp = data.get('gameData', {}).get('probablePitchers', {}).get('away')
                
                # Also check liveData > boxscore > teams for player info
                players = data.get('gameData', {}).get('players', {})
                box = data.get('liveData', {}).get('boxscore', {})
                
                # Try boxscore roster as fallback
                if not players:
                    for side in ('home', 'away'):
                        roster = box.get('teams', {}).get(side, {}).get('players', {})
                        if roster:
                            players.update(roster)

                def name_from_pp(pp):
                    if not pp: return None
                    # Try fullName first
                    n = pp.get('fullName')
                    if n: return n
                    # Or lookup via player id
                    pid = pp.get('id') or pp.get('link', '').split('/')[-1]
                    if pid:
                        # Try gameData.players
                        if f"ID{pid}" in players:
                            player_info = players[f"ID{pid}"]
                            if isinstance(player_info, dict):
                                # Check for person.fullName (boxscore format)
                                if 'person' in player_info:
                                    return player_info['person'].get('fullName')
                                # Or direct fullName (gameData format)
                                return player_info.get('fullName')
                    return None

                home_name = name_from_pp(home_pp) or row['home_pitcher']
                away_name = name_from_pp(away_pp) or row['away_pitcher']

                if home_name != 'TBD' or away_name != 'TBD':
                    df.at[idx, 'home_pitcher'] = home_name
                    df.at[idx, 'away_pitcher'] = away_name
                    filled += 1
            except Exception:
                continue

        if filled:
            print(f"  🔁 Hydrate fallback filled probables for {filled} games")
        return df

    # --- DAILY SCHEDULE (PRIMARY METHOD) ---
    def get_daily_schedule(self, game_date: str) -> pd.DataFrame:
        """
        Fetch MLB schedule and probable pitchers for a given date.
        Tries API-Sports first, falls back to MLB StatsAPI with probables.
        """
        # Use v2 cache to force refresh with new pitcher logic
        cache_key = f"schedule_api_sports_v2_{game_date}"
        cache_path = self._get_cache_path(cache_key)
        
        if self._is_cache_valid(cache_path, self.settings.CACHE_DURATION):
            cached_data = self._load_cache(cache_path)
            if cached_data is not None:
                print(f"✅ Loaded schedule for {game_date} from cache.")
                return cached_data
        
        print(f"📋 Fetching new schedule for {game_date}...")
        
        # Try API-Sports first (only if API key is configured)
        if hasattr(self.settings, 'API_SPORTS_KEY') and self.settings.API_SPORTS_KEY:
            try:
                print(f"🌐 Fetching schedule from API-Sports for {game_date}...")
                games_json = self._fetch_schedule_from_api_sports(game_date)
                if games_json:
                    games_df = self._parse_api_sports_schedule(games_json, game_date)
                    if not games_df.empty:
                        print(f"✅ Found {len(games_df)} games from API-Sports.")
                        self._save_cache(games_df, cache_path)
                        return games_df
            except Exception as e:
                print(f"⚠️ API-Sports failed: {e}, trying MLB StatsAPI fallback...")
        else:
            print("⚠️ API-Sports key not configured, using MLB StatsAPI...")
        
        # Fallback to MLB StatsAPI (now requests probables)
        try:
            games_df = self._fetch_schedule_from_mlb_api(game_date)
            if not games_df.empty:
                # Optional: fill remaining TBDs via game details
                if (games_df['home_pitcher'] == 'TBD').any() or (games_df['away_pitcher'] == 'TBD').any():
                    games_df = self._fill_probables_from_game_details(games_df)

                print(f"✅ Found {len(games_df)} games from MLB StatsAPI.")
                self._save_cache(games_df, cache_path)
                return games_df
        except Exception as e:
            print(f"❌ MLB StatsAPI fallback also failed: {e}")
        
        # If we're in 2025 or later, create sample data for testing
        year = int(game_date.split('-')[0])
        if year >= 2025:
            print(f"📅 Creating sample games for future date {game_date}...")
            sample_games = self._create_sample_schedule(game_date)
            if not sample_games.empty:
                print(f"✅ Created {len(sample_games)} sample games for testing.")
                self._save_cache(sample_games, cache_path)
                return sample_games
        
        print(f"🚫 No games found for {game_date}.")
        return pd.DataFrame()

    def _parse_api_sports_schedule(self, games_json: List[Dict], game_date: str) -> pd.DataFrame:
        """Parses the JSON response from API-Sports into a clean DataFrame."""
        parsed_games = []
        for game in games_json:
            try:
                home_pitcher = 'TBD'
                away_pitcher = 'TBD'
                
                if 'lineups' in game:
                    if 'home' in game['lineups'] and 'startingPitcher' in game['lineups']['home']:
                        home_pitcher = game['lineups']['home']['startingPitcher'].get('name', 'TBD')
                    if 'away' in game['lineups'] and 'startingPitcher' in game['lineups']['away']:
                        away_pitcher = game['lineups']['away']['startingPitcher'].get('name', 'TBD')
                
                if home_pitcher == 'TBD' and 'pitchers' in game:
                    home_pitcher = game['pitchers'].get('home', 'TBD') or 'TBD'
                    away_pitcher = game['pitchers'].get('away', 'TBD') or 'TBD'
                
                home_team = game['teams']['home']['name']
                away_team = game['teams']['away']['name']
                
                # Convert to abbreviations for consistency
                home_abbrev = self.full_name_to_abbrev.get(home_team, home_team)
                away_abbrev = self.full_name_to_abbrev.get(away_team, away_team)
                
                parsed_games.append({
                    'Date': game_date,
                    'Home': home_abbrev,
                    'Away': away_abbrev,
                    'home_pitcher': home_pitcher,
                    'away_pitcher': away_pitcher,
                    'home_team_full': home_team,  # Keep full names for Statcast
                    'away_team_full': away_team,
                    'game_id': game.get('id', ''),
                    'game_time': game.get('time', ''),
                    'status': game.get('status', {}).get('long', 'Scheduled')
                })
            except Exception as e:
                print(f"⚠️ Error parsing game: {e}")
                continue
        
        if not parsed_games: 
            return pd.DataFrame()
        
        df = pd.DataFrame(parsed_games)
        df = team_normalizer.normalize_dataframe_teams(df, ['Home', 'Away'])
        return df[['Date', 'Home', 'Away', 'home_pitcher', 'away_pitcher']]

    def _create_sample_schedule(self, game_date: str) -> pd.DataFrame:
        """Create sample schedule for testing purposes."""
        sample_games = pd.DataFrame({
            'Date': [game_date] * 5,
            'Home': ['NYY', 'BOS', 'LAD', 'HOU', 'ATL'],
            'Away': ['TBR', 'TOR', 'SDP', 'SEA', 'PHI'],
            'home_pitcher': ['Gerrit Cole', 'Brayan Bello', 'Walker Buehler', 'Framber Valdez', 'Spencer Strider'],
            'away_pitcher': ['Shane Baz', 'José Berríos', 'Joe Musgrove', 'Logan Gilbert', 'Zack Wheeler']
        })
        
        return sample_games

    def get_enhanced_daily_schedule(self, game_date: str) -> pd.DataFrame:
        """
        Get daily schedule enhanced with odds and pitcher information.
        Uses enhanced normalizer for better pitcher matching.
        
        This method should replace the existing get_enhanced_daily_schedule in DataManager.
        Location: In the DataManager class, after get_daily_schedule method.
        """
        print(f"\n🏟️ Building enhanced schedule for {game_date}...")
        
        # 1. Get base schedule from MLB API
        games_df = self.get_daily_schedule(game_date)
        
        if games_df.empty:
            print("⚠️ No games found from MLB API")
            return games_df
        
        # 2. Normalize names using enhanced normalizer if available
        if hasattr(self, 'normalizer') and self.normalizer:
            print("🔧 Normalizing team and pitcher names...")
            original_pitchers = games_df[['home_pitcher', 'away_pitcher']].copy()
            
            games_df = self.normalizer.normalize_game_data(
                games_df,
                team_columns=['Home', 'Away'],
                pitcher_columns=['home_pitcher', 'away_pitcher']
            )
            
            # Show normalization results
            home_changed = (original_pitchers['home_pitcher'] != games_df['home_pitcher']).sum()
            away_changed = (original_pitchers['away_pitcher'] != games_df['away_pitcher']).sum()
            if home_changed > 0 or away_changed > 0:
                print(f"   ✅ Normalized {home_changed} home and {away_changed} away pitcher names")
                
                # Show examples of normalizations
                if home_changed > 0:
                    mask = original_pitchers['home_pitcher'] != games_df['home_pitcher']
                    for idx in games_df[mask].head(2).index:
                        orig = original_pitchers.loc[idx, 'home_pitcher']
                        new = games_df.loc[idx, 'home_pitcher']
                        if orig != 'TBD' and new != 'TBD':
                            print(f"      '{orig}' → '{new}'")
        elif hasattr(self, 'team_normalizer') and self.team_normalizer:
            # Fall back to just team normalization
            print("🔧 Normalizing team names only (no pitcher normalizer available)...")
            games_df = self.team_normalizer.normalize_dataframe_teams(games_df, ['Home', 'Away'])
        
        # 3. Try to get odds data
        if self.settings.ODDS_API_KEY:
            odds_df = self.fetch_odds_and_pitchers(game_date)
            
            if not odds_df.empty:
                # Normalize odds data too if normalizer available
                if hasattr(self, 'normalizer') and self.normalizer:
                    odds_df = self.normalizer.normalize_game_data(
                        odds_df,
                        team_columns=['Home', 'Away'],
                        pitcher_columns=['home_pitcher', 'away_pitcher'] if 'home_pitcher' in odds_df.columns else []
                    )
                elif hasattr(self, 'team_normalizer') and self.team_normalizer:
                    odds_df = self.team_normalizer.normalize_dataframe_teams(odds_df, ['Home', 'Away'])
                
                # Ensure odds_df has Date aligned
                if 'Date' not in odds_df.columns:
                    odds_df['Date'] = game_date
                
                # Determine merge keys based on what's available
                merge_keys = ['Date', 'Home', 'Away']
                if 'TimeBucket' in odds_df.columns and 'TimeBucket' in games_df.columns:
                    merge_keys.append('TimeBucket')
                
                # Select columns to merge (avoid duplicates)
                odds_columns = merge_keys.copy()
                if 'home_odds' in odds_df.columns:
                    odds_columns.append('home_odds')
                if 'away_odds' in odds_df.columns:
                    odds_columns.append('away_odds')
                if 'total' in odds_df.columns:
                    odds_columns.append('total')
                if 'num_books' in odds_df.columns:
                    odds_columns.append('num_books')
                
                # Merge odds with games
                games_df = pd.merge(
                    games_df,
                    odds_df[odds_columns],
                    on=merge_keys,
                    how='left'
                )
                
                if 'num_books' in odds_df.columns and odds_df['num_books'].max() > 0:
                    print(f"  ✅ Added odds for up to {odds_df['num_books'].max()} bookmakers")
            
            # 4. Try to get pitcher information from Odds API
            pitchers = self.fetch_pitchers_from_odds_api(game_date)
            if pitchers:
                updated_count = 0
                for idx, row in games_df.iterrows():
                    key = f"{row['Away']}@{row['Home']}"
                    if key in pitchers:
                        away_p, home_p = pitchers[key]
                        if away_p != "TBD" and row['away_pitcher'] == "TBD":
                            games_df.at[idx, 'away_pitcher'] = away_p
                            updated_count += 1
                        if home_p != "TBD" and row['home_pitcher'] == "TBD":
                            games_df.at[idx, 'home_pitcher'] = home_p
                            updated_count += 1
                
                if updated_count > 0:
                    print(f"  ✅ Updated {updated_count} TBD pitchers from Odds API")
        
        # 5. Calculate implied probabilities from odds
        if 'home_odds' in games_df.columns:
            games_df['home_implied_prob'] = games_df.apply(
                lambda x: self._american_to_implied_prob(x['home_odds']) if pd.notna(x['home_odds']) else None,
                axis=1
            )
            games_df['away_implied_prob'] = games_df.apply(
                lambda x: self._american_to_implied_prob(x['away_odds']) if pd.notna(x['away_odds']) else None,
                axis=1
            )
        
        # 6. Diagnose any potential merge issues (optional - only if we have pitcher stats loaded)
        if hasattr(self, 'normalizer') and self.normalizer and hasattr(self, 'pitcher_stats_df'):
            print("\n🔍 Diagnosing potential pitcher merge issues...")
            diagnostics = self.normalizer.diagnose_merge_issues(
                games_df, 
                self.pitcher_stats_df,
                merge_column='Name'
            )
            
            print(f"   Expected match rate: {diagnostics['match_rate']:.1f}%")
            
            if diagnostics['match_rate'] < 80 and diagnostics['unmatched_from_games']:
                print(f"   ⚠️ Unmatched pitchers: {', '.join(diagnostics['unmatched_from_games'][:5])}")
                
                # Show potential matches for first 3 unmatched
                shown = 0
                for pitcher, matches in list(diagnostics['potential_matches'].items())[:3]:
                    if matches and shown < 3:
                        best_match = matches[0]
                        if best_match[1] > 0.7:  # Only show good matches
                            print(f"      '{pitcher}' → maybe '{best_match[0]}' (similarity: {best_match[1]:.2f})")
                            shown += 1
        
        # 7. Summary
        print(f"\n📊 Enhanced Schedule Summary:")
        print(f"  Total games: {len(games_df)}")
        
        if 'home_odds' in games_df.columns:
            with_odds = games_df[games_df['home_odds'].notna()]
            print(f"  Games with odds: {len(with_odds)}")
        
        # Count games with pitchers (not TBD)
        with_home_pitcher = games_df[games_df['home_pitcher'] != 'TBD']
        with_away_pitcher = games_df[games_df['away_pitcher'] != 'TBD']
        with_both_pitchers = games_df[(games_df['home_pitcher'] != 'TBD') & (games_df['away_pitcher'] != 'TBD')]
        
        print(f"  Games with home pitcher: {len(with_home_pitcher)}")
        print(f"  Games with away pitcher: {len(with_away_pitcher)}")
        print(f"  Games with both pitchers: {len(with_both_pitchers)}")
        
        # Show sample of games with their pitchers
        if len(games_df) > 0:
            print("\n  Sample games:")
            for _, game in games_df.head(3).iterrows():
                odds_str = ""
                if 'home_odds' in game and pd.notna(game['home_odds']):
                    home_line = f"{int(game['home_odds']):+d}" if game['home_odds'] else "N/A"
                    away_line = f"{int(game['away_odds']):+d}" if game['away_odds'] else "N/A"
                    odds_str = f" | Odds: {away_line}/{home_line}"
                
                print(f"    {game['Away']} @ {game['Home']}: {game['away_pitcher']} vs {game['home_pitcher']}{odds_str}")
        
        return games_df

    # ============= ODDS API INTEGRATION METHODS =============

    def fetch_odds_and_pitchers(self, game_date: str = None) -> pd.DataFrame:
        """
        Fetch MLB games and betting odds from The Odds API.
        
        Args:
            game_date: Date to fetch (None for upcoming games)
        
        Returns:
            DataFrame with games and odds
        """
        try:
            print("🎲 Fetching data from The Odds API...")
            
            # The Odds API endpoint for upcoming MLB games
            url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
            
            params = {
                'apiKey': self.settings.ODDS_API_KEY,
                'regions': 'us',
                'markets': 'h2h,totals',  # Moneyline and totals
                'bookmakers': 'draftkings,fanduel,betmgm',  # Multiple books
                'oddsFormat': 'american'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            # Check remaining quota
            requests_remaining = response.headers.get('x-requests-remaining', 'unknown')
            requests_used = response.headers.get('x-requests-used', 'unknown')
            print(f"  API Quota: {requests_used} used, {requests_remaining} remaining")
            
            response.raise_for_status()
            data = response.json()
            
            games_list = []
            for game in data:
                # Extract basic game info
                game_time = game.get('commence_time', '')
                home_team = game.get('home_team', '')
                away_team = game.get('away_team', '')
                game_id = game.get('id', '')
                
                # Initialize odds tracking
                home_odds = []
                away_odds = []
                totals = []
                books_with_ml = set()  # Track actual bookmakers
                
                # Get odds from all bookmakers
                for bookmaker in game.get('bookmakers', []):
                    book_name = bookmaker.get('title', '')
                    
                    for market in bookmaker.get('markets', []):
                        if market['key'] == 'h2h':
                            books_with_ml.add(book_name)
                            for outcome in market['outcomes']:
                                if outcome['name'] == home_team:
                                    home_odds.append(outcome['price'])
                                elif outcome['name'] == away_team:
                                    away_odds.append(outcome['price'])
                        
                        elif market['key'] == 'totals':
                            for outcome in market['outcomes']:
                                if outcome['name'] == 'Over':
                                    totals.append(outcome['point'])
                                    break
                
                # Calculate average odds
                avg_home_odds = sum(home_odds) / len(home_odds) if home_odds else None
                avg_away_odds = sum(away_odds) / len(away_odds) if away_odds else None
                avg_total = sum(totals) / len(totals) if totals else None
                
                # Convert team names to abbreviations
                home_abbrev = self._odds_api_team_to_abbrev(home_team)
                away_abbrev = self._odds_api_team_to_abbrev(away_team)
                
                # Parse time for doubleheader handling
                time_bucket = self._to_hour_bucket(game_time)
                
                games_list.append({
                    'Date': game_time[:10] if game_time else game_date,
                    'Time': game_time[11:16] if len(game_time) > 11 else '',
                    'TimeBucket': time_bucket,
                    'Home': home_abbrev,
                    'Away': away_abbrev,
                    'home_team_full': home_team,
                    'away_team_full': away_team,
                    'home_odds': avg_home_odds,
                    'away_odds': avg_away_odds,
                    'total': avg_total,
                    'num_books': len(books_with_ml),  # Fixed: actual bookmaker count
                    'game_id': game_id,
                    'home_pitcher': 'TBD',  # Will try to get from other source
                    'away_pitcher': 'TBD'
                })
            
            if games_list:
                df = pd.DataFrame(games_list)
                print(f"✅ Found {len(df)} games from Odds API")
                
                # Show sample odds
                print("\n📊 Sample odds (American format):")
                for _, game in df.head(3).iterrows():
                    home_line = f"{int(game['home_odds']):+d}" if game['home_odds'] else "N/A"
                    away_line = f"{int(game['away_odds']):+d}" if game['away_odds'] else "N/A"
                    total_line = f"{game['total']:.1f}" if game['total'] else "N/A"
                    print(f"  {game['Away']} @ {game['Home']}: {away_line} / {home_line} | O/U: {total_line}")
                
                return df
            else:
                print("⚠️ No games found from Odds API")
                return pd.DataFrame()
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                print("❌ Invalid API key for The Odds API")
            elif e.response.status_code == 429:
                print("❌ Rate limit exceeded for The Odds API")
            else:
                print(f"❌ HTTP Error from Odds API: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"❌ Error fetching from Odds API: {e}")
            return pd.DataFrame()

    def _to_hour_bucket(self, timestamp: str) -> str:
        """Convert ISO timestamp to hour bucket for doubleheader handling."""
        try:
            return datetime.fromisoformat(timestamp.replace('Z', '+00:00')).strftime('%H')
        except:
            return ''

    def fetch_pitchers_from_odds_api(self, game_date: str = None) -> Dict[str, tuple]:
        """
        Try to extract pitcher information from The Odds API.
        Some bookmakers include this in player props or game descriptions.
        
        Args:
            game_date: Optional date (defaults to today)
        
        Returns:
            Dict mapping "AWAY@HOME" to (away_pitcher, home_pitcher)
        """
        try:
            print("⚾ Checking for pitcher data in Odds API...")
            
            # Try player props endpoint which might have pitcher props
            url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/events"
            
            params = {
                'apiKey': self.settings.ODDS_API_KEY,
                'regions': 'us',
                'dateFormat': 'iso'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            events = response.json()
            pitchers = {}
            
            for event in events:
                # Sometimes pitcher info is in event descriptions
                home_team = event.get('home_team', '')
                away_team = event.get('away_team', '')
                
                # Check if there's additional info
                home_pitcher = "TBD"
                away_pitcher = "TBD"
                
                # Try to get pitcher props (this would be in premium APIs)
                event_id = event.get('id', '')
                if event_id:
                    # Try to fetch player props for this event
                    props_url = f"https://api.the-odds-api.com/v4/sports/baseball_mlb/events/{event_id}/odds"
                    props_params = {
                        'apiKey': self.settings.ODDS_API_KEY,
                        'regions': 'us',
                        'markets': 'pitcher_strikeouts,pitcher_earned_runs'  # Pitcher-specific props
                    }
                    
                    try:
                        props_response = requests.get(props_url, params=props_params, timeout=5)
                        if props_response.status_code == 200:
                            props_data = props_response.json()
                            # Parse pitcher names from props if available
                            # This would need to be implemented based on actual API response
                    except:
                        pass  # Props might not be available
                
                # Map to abbreviations
                home_abbrev = self._odds_api_team_to_abbrev(home_team)
                away_abbrev = self._odds_api_team_to_abbrev(away_team)
                
                key = f"{away_abbrev}@{home_abbrev}"
                pitchers[key] = (away_pitcher, home_pitcher)
            
            # Count how many we found
            found = sum(1 for (a, h) in pitchers.values() if a != "TBD" or h != "TBD")
            if found > 0:
                print(f"  ✅ Found pitcher info for {found} games")
            else:
                print(f"  ⚠️ No pitcher information available from Odds API")
            
            return pitchers
            
        except Exception as e:
            print(f"  ⚠️ Could not fetch pitcher data: {e}")
            return {}

    def _odds_api_team_to_abbrev(self, team_name: str) -> str:
        """
        Convert Odds API team names to our standard abbreviations.
        FIXED: Guards against None/weird strings
        """
        if not team_name:
            return ''  # or 'UNK'
        
        team_name = str(team_name).strip()
        
        odds_team_mapping = {
            # AL
            'Baltimore Orioles':'BAL','Boston Red Sox':'BOS','Chicago White Sox':'CHW',
            'Cleveland Guardians':'CLE','Detroit Tigers':'DET','Houston Astros':'HOU',
            'Kansas City Royals':'KCR','Los Angeles Angels':'LAA','Minnesota Twins':'MIN',
            'New York Yankees':'NYY','Oakland Athletics':'OAK','Seattle Mariners':'SEA',
            'Tampa Bay Rays':'TBR','Texas Rangers':'TEX','Toronto Blue Jays':'TOR',
            # NL
            'Arizona Diamondbacks':'ARI','Atlanta Braves':'ATL','Chicago Cubs':'CHC',
            'Cincinnati Reds':'CIN','Colorado Rockies':'COL','Los Angeles Dodgers':'LAD',
            'Miami Marlins':'MIA','Milwaukee Brewers':'MIL','New York Mets':'NYM',
            'Philadelphia Phillies':'PHI','Pittsburgh Pirates':'PIT','San Diego Padres':'SDP',
            'San Francisco Giants':'SFG','St. Louis Cardinals':'STL','St Louis Cardinals':'STL',
            'Washington Nationals':'WSN',
            # Common book/short names / nicknames
            'Athletics':'OAK','Guardians':'CLE','D-backs':'ARI','Arizona Dbacks':'ARI',
            'Chi White Sox':'CHW','Chi Cubs':'CHC','NY Yankees':'NYY','NY Mets':'NYM',
            'LA Dodgers':'LAD','SF Giants':'SFG','SD Padres':'SDP','KC Royals':'KCR',
            'TB Rays':'TBR','Wash Nationals':'WSN'
        }
        
        if team_name in odds_team_mapping:
            return odds_team_mapping[team_name]
        return team_name[:3].upper() if len(team_name) >= 3 else team_name.upper()

    def get_enhanced_daily_schedule(self, game_date: str) -> pd.DataFrame:
        """
        Get daily schedule enhanced with odds and pitcher information.
        Fixed to merge on Date + Home + Away (+ TimeBucket if available).
        """
        print(f"\n🏟️ Building enhanced schedule for {game_date}...")
        
        # 1. Get base schedule from MLB API
        games_df = self.get_daily_schedule(game_date)
        
        if games_df.empty:
            print("⚠️ No games found from MLB API")
            return games_df
        
        # 2. Try to get odds data
        if self.settings.ODDS_API_KEY:
            odds_df = self.fetch_odds_and_pitchers(game_date)
            
            if not odds_df.empty:
                # Ensure odds_df has Date aligned
                if 'Date' not in odds_df.columns:
                    odds_df['Date'] = game_date
                
                # Determine merge keys based on what's available
                merge_keys = ['Date', 'Home', 'Away']
                if 'TimeBucket' in odds_df.columns and 'TimeBucket' in games_df.columns:
                    merge_keys.append('TimeBucket')
                
                # Merge odds with games
                games_df = pd.merge(
                    games_df,
                    odds_df[merge_keys + ['home_odds', 'away_odds', 'total', 'num_books']],
                    on=merge_keys,
                    how='left'
                )
                
                if odds_df['num_books'].max():
                    print(f"  ✅ Added odds for up to {odds_df['num_books'].max()} bookmakers")
            
            # 3. Try to get pitcher information
            pitchers = self.fetch_pitchers_from_odds_api(game_date)
            if pitchers:
                for idx, row in games_df.iterrows():
                    key = f"{row['Away']}@{row['Home']}"
                    if key in pitchers:
                        away_p, home_p = pitchers[key]
                        if away_p != "TBD" and row['away_pitcher'] == "TBD":
                            games_df.at[idx, 'away_pitcher'] = away_p
                        if home_p != "TBD" and row['home_pitcher'] == "TBD":
                            games_df.at[idx, 'home_pitcher'] = home_p
        
        # 4. Calculate implied probabilities from odds
        if 'home_odds' in games_df.columns:
            games_df['home_implied_prob'] = games_df.apply(
                lambda x: self._american_to_implied_prob(x['home_odds']) if pd.notna(x['home_odds']) else None,
                axis=1
            )
            games_df['away_implied_prob'] = games_df.apply(
                lambda x: self._american_to_implied_prob(x['away_odds']) if pd.notna(x['away_odds']) else None,
                axis=1
            )
        
        # Summary
        print(f"\n📊 Enhanced Schedule Summary:")
        print(f"  Total games: {len(games_df)}")
        
        if 'home_odds' in games_df.columns:
            with_odds = games_df[games_df['home_odds'].notna()]
            print(f"  Games with odds: {len(with_odds)}")
        
        with_pitchers = games_df[(games_df['home_pitcher'] != 'TBD') | (games_df['away_pitcher'] != 'TBD')]
        print(f"  Games with pitchers: {len(with_pitchers)}")
        
        return games_df

    def _american_to_implied_prob(self, odds: float) -> float:
        """Convert American odds to implied probability."""
        if pd.isna(odds):
            return None
        
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)

    def test_odds_api(self) -> pd.DataFrame:
        """
        Test The Odds API connection and show available data.
        Good for checking your API key and quota.
        """
        try:
            print("\n🧪 Testing The Odds API Connection...")
            print("=" * 50)
            
            # First check what sports are available
            url = "https://api.the-odds-api.com/v4/sports"
            params = {'apiKey': self.settings.ODDS_API_KEY}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            sports = response.json()
            mlb = [s for s in sports if 'baseball_mlb' in s.get('key', '')]
            
            if mlb:
                mlb_info = mlb[0]
                print(f"✅ MLB Status:")
                print(f"   Key: {mlb_info.get('key')}")
                print(f"   Active: {mlb_info.get('active', False)}")
                print(f"   Group: {mlb_info.get('group')}")
                print(f"   Description: {mlb_info.get('description')}")
            else:
                print("⚠️ MLB not found in available sports")
            
            # Check API usage
            print(f"\n📊 API Quota Status:")
            print(f"   Requests Used: {response.headers.get('x-requests-used', 'unknown')}")
            print(f"   Requests Remaining: {response.headers.get('x-requests-remaining', 'unknown')}")
            
            # Now get actual games with odds
            print(f"\n🎲 Fetching current MLB odds...")
            games_df = self.fetch_odds_and_pitchers()
            
            if not games_df.empty:
                print(f"\n✅ Successfully fetched {len(games_df)} games")
                
                # Calculate value opportunities
                print("\n💰 Potential Value Bets (if your model disagrees):")
                for _, game in games_df.head(5).iterrows():
                    if pd.notna(game['home_odds']):
                        home_prob = self._american_to_implied_prob(game['home_odds'])
                        away_prob = self._american_to_implied_prob(game['away_odds'])
                        print(f"   {game['Away']} @ {game['Home']}")
                        print(f"      Market: Home {home_prob:.1%} | Away {away_prob:.1%}")
            
            print("=" * 50)
            return games_df
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                print("❌ Invalid API key for The Odds API")
                print("   Please check your ODDS_API_KEY in .env file")
            else:
                print(f"❌ HTTP Error: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"❌ Odds API test failed: {e}")
            return pd.DataFrame()

    # --- HISTORICAL DATA METHODS ---

    def get_historical_team_data(self, seasons: List[int]) -> pd.DataFrame:
        """
        Get aggregate team batting and pitching stats for past seasons.
        FIXED: Ensures team names are in abbreviation format.
        """
        cache_key = f"team_data_agg_{'_'.join(map(str, seasons))}"
        cache_path = self._get_cache_path(cache_key)
        
        if self._is_cache_valid(cache_path, self.settings.DAILY_CACHE_DURATION * 30):
            cached_data = self._load_cache(cache_path)
            if cached_data is not None:
                print(f"✅ Loaded historical team data for {seasons} from cache.")
                return cached_data
        
        all_seasons_data = []
        for season in seasons:
            print(f"📊 Fetching team data for {season} season...")
            try:
                batting_df = pyb.team_batting(season)
                pitching_df = pyb.team_pitching(season)
                
                season_df = pd.merge(
                    batting_df, 
                    pitching_df, 
                    on="Team", 
                    suffixes=('_batting', '_pitching')
                )
                season_df['Season'] = season
                
                # CRITICAL: Normalize team names to abbreviations
                # pybaseball might return different formats depending on version
                season_df = team_normalizer.normalize_dataframe_teams(season_df, ['Team'])
                
                # Debug: Show what team format we have
                if self.verbose:
                    print(f"  📊 Sample team names from pybaseball: {season_df['Team'].head(3).tolist()}")
                
                all_seasons_data.append(season_df)
            except Exception as e:
                print(f"❌ Error fetching team data for {season}: {e}")
                continue
        
        if not all_seasons_data:
            return pd.DataFrame()
        
        combined_df = pd.concat(all_seasons_data, ignore_index=True)
        print(f"✅ Combined historical team data for {seasons}.")
        
        # Final check: ensure all team names are abbreviations
        unique_teams = combined_df['Team'].unique()
        print(f"  📊 Unique team formats in final data: {unique_teams[:5].tolist()}...")
        
        self._save_cache(combined_df, cache_path)
        return combined_df

    def get_historical_pitcher_data(self, seasons: List[int]) -> pd.DataFrame:
        """
        Get pitcher stats using regular pitching_stats.
        FIXED: Now fetches ALL pitchers with qual=1, not just qualified pitchers.
        """
        cache_key = f"pitcher_data_all_v2_{'_'.join(map(str, seasons))}"  # v2 for new comprehensive data
        cache_path = self._get_cache_path(cache_key)
        
        if self._is_cache_valid(cache_path, self.settings.DAILY_CACHE_DURATION * 30):
            cached_data = self._load_cache(cache_path)
            if cached_data is not None:
                print(f"✅ Loaded historical pitcher data for {seasons} from cache.")
                return cached_data
        
        all_pitcher_data = []
        for season in seasons:
            print(f"⚾ Fetching ALL pitcher data for {season} season...")
            try:
                # CRITICAL FIX: Add qual=1 to get ALL pitchers, not just qualified ones
                pitcher_df = pyb.pitching_stats(season, qual=1)
                
                if not pitcher_df.empty:
                    pitcher_df['Season'] = season
                    
                    # Clean pitcher names for better matching
                    if 'Name' in pitcher_df.columns:
                        pitcher_df['Name'] = pitcher_df['Name'].str.strip()
                    
                    all_pitcher_data.append(pitcher_df)
                    
                    # Show how many unique pitchers we got
                    unique_pitchers = pitcher_df['Name'].nunique() if 'Name' in pitcher_df.columns else len(pitcher_df)
                    print(f"✅ Fetched {len(pitcher_df)} records for {unique_pitchers} unique pitchers in {season}")
                else:
                    print(f"⚠️  No pitcher data returned for {season}")
                    
            except Exception as e:
                print(f"❌ Error fetching pitcher data for {season}: {e}")
                continue
        
        if not all_pitcher_data:
            print("⚠️ No pitcher data could be fetched, creating sample data...")
            return self._create_sample_pitcher_data(seasons)
            
        combined_df = pd.concat(all_pitcher_data, ignore_index=True)
        
        # Report total unique pitchers across all seasons
        if 'Name' in combined_df.columns:
            total_unique = combined_df['Name'].nunique()
            print(f"✅ Combined historical pitcher data: {len(combined_df)} total records")
            print(f"   Covering {total_unique} unique pitchers across {seasons}")
        else:
            print(f"✅ Combined historical pitcher data for {seasons}.")
        
        self._save_cache(combined_df, cache_path)
        return combined_df

    def _create_sample_pitcher_data(self, seasons: List[int]) -> pd.DataFrame:
        """Create sample pitcher data for testing when real data fails."""
        import random
        
        sample_pitchers = []
        pitcher_profiles = [
            {'name': 'Gerrit Cole', 'era_range': (2.5, 3.5), 'whip_range': (0.95, 1.15)},
            {'name': 'Shane Bieber', 'era_range': (2.8, 3.8), 'whip_range': (1.0, 1.2)},
            {'name': 'Walker Buehler', 'era_range': (2.4, 3.4), 'whip_range': (0.95, 1.1)},
            {'name': 'Zack Wheeler', 'era_range': (2.7, 3.7), 'whip_range': (1.0, 1.2)},
            {'name': 'Spencer Strider', 'era_range': (2.6, 3.6), 'whip_range': (0.9, 1.1)},
        ]
        
        for season in seasons:
            for profile in pitcher_profiles:
                era = random.uniform(*profile['era_range'])
                whip = random.uniform(*profile['whip_range'])
                
                sample_pitchers.append({
                    'Name': profile['name'],
                    'Season': season,
                    'ERA': round(era, 2),
                    'WHIP': round(whip, 3),
                    'K/9': round(random.uniform(8.0, 12.0), 1),
                    'BB/9': round(random.uniform(2.0, 3.5), 1),
                    'HR/9': round(random.uniform(0.8, 1.5), 1),
                    'FIP': round(era + random.uniform(-0.5, 0.5), 2),
                    'IP': round(random.uniform(150, 200), 1),
                    'W': random.randint(10, 18),
                    'L': random.randint(6, 12),
                    'G': random.randint(28, 33),
                    'GS': random.randint(28, 33)
                })
        
        return pd.DataFrame(sample_pitchers)

    def get_historical_nrfi_training_data(self, seasons: List[int], 
                                        max_games_per_season: Optional[int] = None) -> pd.DataFrame:
        """
        Get historical game-by-game data with NRFI outcomes for model training.
        FIXED VERSION: Uses MLB StatsAPI for schedule and optimized Statcast calls.
        """
        cache_key = f"nrfi_training_data_v4_{'_'.join(map(str, seasons))}"  # v4 to force fresh data
        if max_games_per_season: 
            cache_key += f"_limit_{max_games_per_season}"
        cache_path = self._get_cache_path(cache_key)
        
        # Check cache first
        if self._is_cache_valid(cache_path, self.settings.DAILY_CACHE_DURATION * 90):
            cached_data = self._load_cache(cache_path)
            if cached_data is not None:
                print(f"✅ Loaded NRFI training data for {seasons} from cache.")
                return cached_data
        
        print(f"🏗️ Building REAL NRFI training dataset for seasons {seasons}...")
        print("⚠️ This will take time on first run, but will be cached for future use.")
        
        all_training_data = []
        
        for season in seasons:
            print(f"\n📅 Processing {season} season...")
            
            # CRITICAL FIX: Use MLB StatsAPI instead of Baseball Reference
            schedule_df = self._get_season_schedule_mlb_api(season)
            
            if schedule_df.empty:
                print(f"⚠️ Could not get schedule for {season}, skipping.")
                continue
            
            print(f"  Found {len(schedule_df)} games in {season} schedule")
            
            # Apply even sampling to avoid early-season bias
            if max_games_per_season and len(schedule_df) > max_games_per_season:
                schedule_df = self._limit_games_evenly(schedule_df, max_games_per_season)
                print(f"  Sampled {len(schedule_df)} games evenly across season")
            
            # CRITICAL FIX: Optimized NRFI outcome detection
            games_with_outcomes = self._add_nrfi_outcomes_optimized(schedule_df, season)
            
            if not games_with_outcomes.empty:
                all_training_data.append(games_with_outcomes)
                
                # Calculate season NRFI rate
                nrfi_rate = games_with_outcomes['nrfi_result'].mean()
                nrfi_games = games_with_outcomes['nrfi_result'].sum()
                total_games = len(games_with_outcomes)
                
                print(f"  ✅ {season}: {total_games} games with NRFI data")
                print(f"     NRFI Rate: {nrfi_rate:.1%} ({nrfi_games}/{total_games} games)")
            else:
                print(f"  ⚠️ No games with NRFI outcomes found for {season}")
        
        if not all_training_data:
            print("❌ No training data could be fetched from real games")
            print("💡 Troubleshooting tips:")
            print("   1. Check your internet connection")
            print("   2. Verify pybaseball is working: try pybaseball.statcast('2023-04-01', '2023-04-01')")
            print("   3. Consider using more recent seasons (2021-2023)")
            return pd.DataFrame()
        
        # Combine all seasons
        combined_df = pd.concat(all_training_data, ignore_index=True)
        
        # Display overall statistics
        if 'nrfi_result' in combined_df.columns:
            overall_nrfi_rate = combined_df['nrfi_result'].mean()
            total_games = len(combined_df)
            total_nrfi = combined_df['nrfi_result'].sum()
            
            print(f"\n🎯 Successfully built REAL NRFI training dataset!")
            print(f"📊 Total games with data: {total_games}")
            print(f"📈 Overall NRFI Rate: {overall_nrfi_rate:.1%} ({total_nrfi}/{total_games} games)")
            print(f"   (Historical MLB average is 55-60%)")
            
            # Show season breakdown
            print("\n📅 Season Breakdown:")
            for season in combined_df['Season'].unique():
                season_data = combined_df[combined_df['Season'] == season]
                season_rate = season_data['nrfi_result'].mean()
                print(f"   {season}: {len(season_data)} games, {season_rate:.1%} NRFI rate")
        
        # Cache the real data
        self._save_cache(combined_df, cache_path)
        print(f"\n💾 Cached training data for future use")
        
        return combined_df

    def get_historical_game_winner_training_data(self, seasons: List[int], 
                                                max_games_per_season: Optional[int] = None) -> pd.DataFrame:
        """
        Fetches historical game data with winners for training the game winner model.
        
        Args:
            seasons: List of seasons to fetch
            max_games_per_season: Optional limit on games per season
            
        Returns:
            DataFrame with game results and winner labels
        """
        cache_key = f"game_winner_training_data_v2_{'_'.join(map(str, seasons))}"
        if max_games_per_season:
            cache_key += f"_limit_{max_games_per_season}"
        cache_path = self._get_cache_path(cache_key)
        
        # Check cache
        if self._is_cache_valid(cache_path, self.settings.DAILY_CACHE_DURATION * 90):
            cached_data = self._load_cache(cache_path)
            if cached_data is not None:
                print(f"✅ Loaded Game Winner training data for {seasons} from cache.")
                return cached_data
        
        print(f"🏗️ Building Game Winner training dataset for seasons {seasons}...")
        
        all_games_data = []
        
        for season in seasons:
            print(f"\n📅 Processing {season} season...")
            
            # Get schedule using MLB API
            schedule_df = self._get_season_schedule_mlb_api(season)
            
            if schedule_df.empty:
                print(f"⚠️ Could not get schedule for {season}, skipping.")
                continue
            
            print(f"  Found {len(schedule_df)} games")
            
            # Apply limit if specified
            if max_games_per_season and len(schedule_df) > max_games_per_season:
                schedule_df = self._limit_games_evenly(schedule_df, max_games_per_season)
                print(f"  Limited to {len(schedule_df)} games")
            
            # Get game results from MLB API
            games_with_results = self._add_game_results(schedule_df, season)
            
            if not games_with_results.empty:
                all_games_data.append(games_with_results)
                
                # Calculate win rate
                home_wins = games_with_results['home_team_won'].sum()
                total_games = len(games_with_results)
                home_win_rate = home_wins / total_games if total_games > 0 else 0
                
                print(f"  ✅ {season}: {total_games} games with results")
                print(f"     Home Win Rate: {home_win_rate:.1%} ({home_wins}/{total_games})")
            else:
                print(f"  ⚠️ No game results found for {season}")
        
        if not all_games_data:
            print("❌ No Game Winner training data could be fetched")
            return pd.DataFrame()
        
        # Combine all seasons
        combined_df = pd.concat(all_games_data, ignore_index=True)
        
        # Display overall statistics
        overall_home_win_rate = combined_df['home_team_won'].mean()
        total_games = len(combined_df)
        total_home_wins = combined_df['home_team_won'].sum()
        
        print(f"\n🎯 Successfully built Game Winner training dataset!")
        print(f"📊 Total games: {total_games}")
        print(f"📈 Overall Home Win Rate: {overall_home_win_rate:.1%} ({total_home_wins}/{total_games})")
        
        # Show season breakdown
        print("\n📅 Season Breakdown:")
        for season in combined_df['Season'].unique():
            season_data = combined_df[combined_df['Season'] == season]
            season_rate = season_data['home_team_won'].mean()
            print(f"   {season}: {len(season_data)} games, {season_rate:.1%} home win rate")
        
        # Cache the data
        self._save_cache(combined_df, cache_path)
        print(f"\n💾 Cached training data for future use")
        
        return combined_df

    def get_historical_game_winner_training_data_with_pitchers(self, seasons: List[int],
                                                            max_games_per_season: Optional[int] = None) -> pd.DataFrame:
        """
        Enhanced version of the training data fetcher that INCLUDES probable pitcher names.
        This fetches historical games with results AND pitcher information for enhanced model training.
        
        Args:
            seasons: List of seasons to fetch (e.g., [2022, 2023])
            max_games_per_season: Optional limit on games per season
            
        Returns:
            DataFrame with columns including:
            - Date, Home, Away (team abbreviations)
            - home_score, away_score
            - home_team_won (0 or 1)
            - home_pitcher, away_pitcher (names)
            - Season
        """
        # Use a new cache key to store this enhanced data
        cache_key = f"game_winner_pitchers_training_data_v1_{'_'.join(map(str, seasons))}"
        if max_games_per_season:
            cache_key += f"_limit_{max_games_per_season}"
        cache_path = self._get_cache_path(cache_key)
        
        # Check cache
        if self._is_cache_valid(cache_path, self.settings.DAILY_CACHE_DURATION * 90):
            cached_data = self._load_cache(cache_path)
            if cached_data is not None:
                print(f"✅ Loaded Game Winner (w/ Pitchers) training data for {seasons} from cache.")
                return cached_data
        
        print(f"🏗️ Building Game Winner (w/ Pitchers) training dataset for seasons {seasons}...")
        print("   This includes fetching probable pitcher data for each game.")
        
        all_games_data = []
        
        for season in seasons:
            print(f"\n📅 Processing {season} season...")
            
            # KEY CHANGE: Call the enhanced schedule fetcher with hydration for pitchers
            schedule_df = self._get_season_schedule_mlb_api(season, hydrate=['probablePitcher'])
            
            if schedule_df.empty:
                print(f"⚠️ Could not get schedule for {season}, skipping.")
                continue
            
            # Show pitcher coverage
            has_pitchers = ((schedule_df['home_pitcher'] != 'TBD') | 
                        (schedule_df['away_pitcher'] != 'TBD')).sum()
            print(f"  Found {len(schedule_df)} games, {has_pitchers} with pitcher data")
            
            # Apply limit if specified
            if max_games_per_season and len(schedule_df) > max_games_per_season:
                schedule_df = self._limit_games_evenly(schedule_df, max_games_per_season)
                print(f"  Limited to {len(schedule_df)} games (evenly sampled across season)")
            
            # Get game results from MLB API (adds scores and winner)
            games_with_results = self._add_game_results(schedule_df, season)
            
            if not games_with_results.empty:
                # Merge pitcher data back in if _add_game_results doesn't carry it forward
                if 'home_pitcher' not in games_with_results.columns:
                    pitcher_cols = ['Date', 'Home', 'Away', 'home_pitcher', 'away_pitcher']
                    if 'home_pitcher_id' in schedule_df.columns:
                        pitcher_cols.extend(['home_pitcher_id', 'away_pitcher_id'])
                    
                    games_with_results = pd.merge(
                        games_with_results, 
                        schedule_df[pitcher_cols], 
                        on=['Date', 'Home', 'Away'], 
                        how='left'
                    )
                
                all_games_data.append(games_with_results)
                
                # Calculate statistics
                home_wins = games_with_results['home_team_won'].sum()
                total_games = len(games_with_results)
                home_win_rate = home_wins / total_games if total_games > 0 else 0
                
                # Pitcher data statistics
                has_home_pitcher = (games_with_results['home_pitcher'] != 'TBD').sum()
                has_away_pitcher = (games_with_results['away_pitcher'] != 'TBD').sum()
                has_both_pitchers = ((games_with_results['home_pitcher'] != 'TBD') & 
                                (games_with_results['away_pitcher'] != 'TBD')).sum()
                
                print(f"  ✅ {season}: {total_games} games with results")
                print(f"     Home Win Rate: {home_win_rate:.1%} ({home_wins}/{total_games})")
                print(f"     Pitcher Data: {has_both_pitchers} games with both pitchers")
                
                # Optional: Try to fill in more pitcher data for TBD games
                if has_both_pitchers < total_games * 0.5:  # Less than 50% coverage
                    print(f"  🔄 Attempting to hydrate more pitcher data...")
                    games_with_results = self._fill_probables_from_game_details(games_with_results)
                    
                    # Recount after hydration
                    has_both_after = ((games_with_results['home_pitcher'] != 'TBD') & 
                                    (games_with_results['away_pitcher'] != 'TBD')).sum()
                    if has_both_after > has_both_pitchers:
                        print(f"     ✅ Hydrated {has_both_after - has_both_pitchers} additional games")
            else:
                print(f"  ⚠️ No game results found for {season}")
        
        if not all_games_data:
            print("❌ No Game Winner training data could be fetched")
            return pd.DataFrame()
        
        # Combine all seasons
        combined_df = pd.concat(all_games_data, ignore_index=True)
        
        # Normalize pitcher names if enhanced normalizer is available
        if hasattr(self, 'normalizer') and self.normalizer:
            print("\n🔧 Normalizing pitcher names...")
            combined_df = self.normalizer.normalize_game_data(
                combined_df,
                team_columns=['Home', 'Away'],
                pitcher_columns=['home_pitcher', 'away_pitcher']
            )
        
        # Display overall statistics
        overall_home_win_rate = combined_df['home_team_won'].mean()
        total_games = len(combined_df)
        total_home_wins = combined_df['home_team_won'].sum()
        
        # Pitcher coverage statistics
        has_home_pitcher = (combined_df['home_pitcher'] != 'TBD').sum()
        has_away_pitcher = (combined_df['away_pitcher'] != 'TBD').sum()
        has_both_pitchers = ((combined_df['home_pitcher'] != 'TBD') & 
                            (combined_df['away_pitcher'] != 'TBD')).sum()
        
        print(f"\n🎯 Successfully built Game Winner w/ Pitchers training dataset!")
        print(f"📊 Total games: {total_games}")
        print(f"📈 Overall Home Win Rate: {overall_home_win_rate:.1%} ({total_home_wins}/{total_games})")
        print(f"⚾ Pitcher Coverage:")
        print(f"   Home pitchers: {has_home_pitcher}/{total_games} ({has_home_pitcher/total_games*100:.1f}%)")
        print(f"   Away pitchers: {has_away_pitcher}/{total_games} ({has_away_pitcher/total_games*100:.1f}%)")
        print(f"   Both pitchers: {has_both_pitchers}/{total_games} ({has_both_pitchers/total_games*100:.1f}%)")
        
        # Show season breakdown
        print("\n📅 Season Breakdown:")
        for season in combined_df['Season'].unique():
            season_data = combined_df[combined_df['Season'] == season]
            season_rate = season_data['home_team_won'].mean()
            season_both = ((season_data['home_pitcher'] != 'TBD') & 
                        (season_data['away_pitcher'] != 'TBD')).sum()
            print(f"   {season}: {len(season_data)} games, {season_rate:.1%} home win rate, "
                f"{season_both} with both pitchers")
        
        # Show sample of data
        print("\n📋 Sample of training data:")
        sample_cols = ['Date', 'Home', 'Away', 'home_pitcher', 'away_pitcher', 
                    'home_score', 'away_score', 'home_team_won']
        print(combined_df[sample_cols].head(10).to_string())
        
        # Cache the data
        self._save_cache(combined_df, cache_path)
        print(f"\n💾 Cached training data for future use")
        
        return combined_df
 # --- PITCHER FEATURE ENGINEERING ---

    def add_pitcher_features(self, games_df: pd.DataFrame, pitcher_stats_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add pitcher statistics to games DataFrame with improved name matching.
        Now handles enhanced normalizer for better pitcher name matching.
        """
        print("⚾ Adding pitcher statistics with enhanced matching...")
        
        # Store original counts for reporting
        original_home_pitchers = (games_df['home_pitcher'].notna() & (games_df['home_pitcher'] != 'TBD')).sum()
        original_away_pitchers = (games_df['away_pitcher'].notna() & (games_df['away_pitcher'] != 'TBD')).sum()
        
        # Create copies to avoid modifying originals
        games_enhanced = games_df.copy()
        pitcher_stats_enhanced = pitcher_stats_df.copy()
        
        # Ensure pitcher names are clean
        if 'Name' in pitcher_stats_enhanced.columns:
            pitcher_stats_enhanced['Name'] = pitcher_stats_enhanced['Name'].str.strip()
        
        # Check if we have the enhanced normalizer
        if hasattr(self, 'normalizer') and self.normalizer:
            print("   🔧 Using enhanced pitcher name normalization...")
            
            # Step 1: Normalize pitcher names in the stats DataFrame
            pitcher_stats_enhanced['Name_normalized'] = pitcher_stats_enhanced['Name'].apply(
                lambda x: self.normalizer.pitcher_normalizer.normalize_pitcher_name(x) if pd.notna(x) else x
            )
            
            # Also create lowercase version for fallback matching
            pitcher_stats_enhanced['Name_lower'] = pitcher_stats_enhanced['Name'].str.lower().str.strip()
            
            # Step 2: Normalize pitcher names in games DataFrame
            games_enhanced['home_pitcher_normalized'] = games_enhanced['home_pitcher'].apply(
                lambda x: self.normalizer.pitcher_normalizer.normalize_pitcher_name(x) if pd.notna(x) and x != 'TBD' else x
            )
            games_enhanced['away_pitcher_normalized'] = games_enhanced['away_pitcher'].apply(
                lambda x: self.normalizer.pitcher_normalizer.normalize_pitcher_name(x) if pd.notna(x) and x != 'TBD' else x
            )
            
            # Also create lowercase versions
            games_enhanced['home_pitcher_lower'] = games_enhanced['home_pitcher'].str.lower().str.strip()
            games_enhanced['away_pitcher_lower'] = games_enhanced['away_pitcher'].str.lower().str.strip()
            
            # Step 3: Try primary merge on normalized names
            print("   📊 Attempting primary merge on normalized names...")
            
            # For better merging, get the most recent season stats for each pitcher
            if 'Season' in pitcher_stats_enhanced.columns:
                # Group by pitcher and take the most recent season
                pitcher_stats_latest = pitcher_stats_enhanced.sort_values('Season').groupby('Name_normalized').last().reset_index()
            else:
                pitcher_stats_latest = pitcher_stats_enhanced
            
            # Merge for home pitchers
            games_enhanced = pd.merge(
                games_enhanced,
                pitcher_stats_latest.add_suffix('_home'),
                left_on='home_pitcher_normalized',
                right_on='Name_normalized_home',
                how='left'
            )
            
            # Merge for away pitchers
            games_enhanced = pd.merge(
                games_enhanced,
                pitcher_stats_latest.add_suffix('_away'),
                left_on='away_pitcher_normalized',
                right_on='Name_normalized_away',
                how='left'
            )
            
            # Step 4: Fallback merge for unmatched using exact match on original names
            home_unmatched_mask = games_enhanced['Name_home'].isna() & games_enhanced['home_pitcher'].notna() & (games_enhanced['home_pitcher'] != 'TBD')
            away_unmatched_mask = games_enhanced['Name_away'].isna() & games_enhanced['away_pitcher'].notna() & (games_enhanced['away_pitcher'] != 'TBD')
            
            if home_unmatched_mask.any() or away_unmatched_mask.any():
                print(f"   🔄 Attempting exact name fallback for {home_unmatched_mask.sum()} home and {away_unmatched_mask.sum()} away pitchers...")
                
                # Try direct name matching for remaining unmatched
                for idx in games_enhanced[home_unmatched_mask].index:
                    pitcher_name = games_enhanced.loc[idx, 'home_pitcher']
                    matches = pitcher_stats_enhanced[pitcher_stats_enhanced['Name'] == pitcher_name]
                    if not matches.empty:
                        # Use most recent season if multiple
                        best_match = matches.sort_values('Season').iloc[-1] if 'Season' in matches.columns else matches.iloc[0]
                        for col in pitcher_stats_enhanced.columns:
                            if col not in ['Name_normalized', 'Name_lower']:
                                games_enhanced.loc[idx, f'{col}_home'] = best_match[col]
                
                for idx in games_enhanced[away_unmatched_mask].index:
                    pitcher_name = games_enhanced.loc[idx, 'away_pitcher']
                    matches = pitcher_stats_enhanced[pitcher_stats_enhanced['Name'] == pitcher_name]
                    if not matches.empty:
                        # Use most recent season if multiple
                        best_match = matches.sort_values('Season').iloc[-1] if 'Season' in matches.columns else matches.iloc[0]
                        for col in pitcher_stats_enhanced.columns:
                            if col not in ['Name_normalized', 'Name_lower']:
                                games_enhanced.loc[idx, f'{col}_away'] = best_match[col]
            
            # Clean up temporary columns
            cols_to_drop = [
                'home_pitcher_normalized', 'away_pitcher_normalized',
                'home_pitcher_lower', 'away_pitcher_lower',
                'Name_normalized_home', 'Name_normalized_away',
                'Name_lower_home', 'Name_lower_away'
            ]
            games_enhanced = games_enhanced.drop(columns=[col for col in cols_to_drop if col in games_enhanced.columns])
            
        else:
            # Fallback to simple merge without enhanced normalizer
            print("   ⚠️ Enhanced normalizer not available, using basic merge...")
            
            # Get most recent season for each pitcher if Season column exists
            if 'Season' in pitcher_stats_enhanced.columns:
                pitcher_stats_latest = pitcher_stats_enhanced.sort_values('Season').groupby('Name').last().reset_index()
            else:
                pitcher_stats_latest = pitcher_stats_enhanced
            
            # Basic merge on exact name match
            games_enhanced = pd.merge(
                games_enhanced,
                pitcher_stats_latest.add_suffix('_home'),
                left_on='home_pitcher',
                right_on='Name_home',
                how='left'
            )
            
            games_enhanced = pd.merge(
                games_enhanced,
                pitcher_stats_latest.add_suffix('_away'),
                left_on='away_pitcher',
                right_on='Name_away',
                how='left'
            )
        
        # Step 6: Report merge success statistics
        home_matched = games_enhanced['Name_home'].notna().sum()
        away_matched = games_enhanced['Name_away'].notna().sum()
        
        home_match_rate = (home_matched / original_home_pitchers * 100) if original_home_pitchers > 0 else 0
        away_match_rate = (away_matched / original_away_pitchers * 100) if original_away_pitchers > 0 else 0
        
        print(f"\n   📊 Pitcher Merge Statistics:")
        print(f"      Home pitchers: {home_matched}/{original_home_pitchers} matched ({home_match_rate:.1f}%)")
        print(f"      Away pitchers: {away_matched}/{original_away_pitchers} matched ({away_match_rate:.1f}%)")
        
        # Show which pitchers didn't match (for debugging)
        if home_match_rate < 100 or away_match_rate < 100:
            unmatched_home = games_enhanced[
                games_enhanced['Name_home'].isna() & 
                games_enhanced['home_pitcher'].notna() & 
                (games_enhanced['home_pitcher'] != 'TBD')
            ]['home_pitcher'].unique()
            
            unmatched_away = games_enhanced[
                games_enhanced['Name_away'].isna() & 
                games_enhanced['away_pitcher'].notna() & 
                (games_enhanced['away_pitcher'] != 'TBD')
            ]['away_pitcher'].unique()
            
            if len(unmatched_home) > 0:
                print(f"      ❌ Unmatched home pitchers: {', '.join(unmatched_home[:5])}")
                if len(unmatched_home) > 5:
                    print(f"         ... and {len(unmatched_home) - 5} more")
            
            if len(unmatched_away) > 0:
                print(f"      ❌ Unmatched away pitchers: {', '.join(unmatched_away[:5])}")
                if len(unmatched_away) > 5:
                    print(f"         ... and {len(unmatched_away) - 5} more")
        
        # Step 7: Add key pitcher metrics if successful
        if 'ERA_home' in games_enhanced.columns and 'ERA_away' in games_enhanced.columns:
            print("   ✅ Adding derived pitcher metrics...")
            
            # ERA differential
            games_enhanced['pitcher_ERA_diff'] = games_enhanced['ERA_away'] - games_enhanced['ERA_home']
            
            # WHIP differential (if available)
            if 'WHIP_home' in games_enhanced.columns and 'WHIP_away' in games_enhanced.columns:
                games_enhanced['pitcher_WHIP_diff'] = games_enhanced['WHIP_away'] - games_enhanced['WHIP_home']
            
            # K/9 differential (if available)
            if 'K/9_home' in games_enhanced.columns and 'K/9_away' in games_enhanced.columns:
                games_enhanced['pitcher_K9_diff'] = games_enhanced['K/9_home'] - games_enhanced['K/9_away']
            
            # FIP differential (if available)
            if 'FIP_home' in games_enhanced.columns and 'FIP_away' in games_enhanced.columns:
                games_enhanced['pitcher_FIP_diff'] = games_enhanced['FIP_away'] - games_enhanced['FIP_home']
        
        return games_enhanced

    def get_pitcher_stats_for_analysis(self, seasons: List[int] = None) -> pd.DataFrame:
        """
        Helper method to get pitcher stats in the right format for merging.
        Can be called before add_pitcher_features to prepare the stats DataFrame.
        
        Args:
            seasons: List of seasons to get stats for (defaults to current and previous)
            
        Returns:
            DataFrame with pitcher statistics ready for merging
        """
        if seasons is None:
            current_year = datetime.now().year
            seasons = [current_year - 1, current_year]
        
        # Get historical pitcher data - now with ALL pitchers thanks to qual=1
        pitcher_stats = self.get_historical_pitcher_data(seasons)
        
        # Store for diagnostic purposes
        self.pitcher_stats_df = pitcher_stats
        
        # If we have the normalizer, add normalized names to stats
        if hasattr(self, 'normalizer') and self.normalizer and not pitcher_stats.empty:
            print("   🔧 Pre-normalizing pitcher names in stats database...")
            pitcher_stats['Name_normalized'] = pitcher_stats['Name'].apply(
                lambda x: self.normalizer.pitcher_normalizer.normalize_pitcher_name(x) if pd.notna(x) else x
            )
            
            # Show some examples of normalization
            changed = pitcher_stats[pitcher_stats['Name'] != pitcher_stats['Name_normalized']]
            if not changed.empty:
                print(f"      Normalized {len(changed)} pitcher names in stats")
                for _, row in changed.head(3).iterrows():
                    print(f"         '{row['Name']}' → '{row['Name_normalized']}'")
        
        # Report coverage statistics
        unique_pitchers = pitcher_stats['Name'].nunique() if 'Name' in pitcher_stats.columns else 0
        print(f"   📊 Pitcher stats database: {unique_pitchers} unique pitchers available")
        
        return pitcher_stats

    def _add_game_results(self, schedule_df: pd.DataFrame, season: int) -> pd.DataFrame:
        """
        Add game results (scores and winner) to schedule.
        FIXED: Ensures team names use correct pybaseball abbreviations.
        """
        games_with_results = []
        unique_dates = schedule_df['Date'].unique()
        
        print(f"  🔍 Fetching game results for {len(unique_dates)} dates...")
        
        # Define the abbreviation fixes once
        abbrev_fixes = {
            'CWS': 'CHW',  # Chicago White Sox
            'KC': 'KCR',   # Kansas City Royals
            'LA': 'LAD',   # Los Angeles Dodgers
            'SD': 'SDP',   # San Diego Padres
            'SF': 'SFG',   # San Francisco Giants
            'TB': 'TBR',   # Tampa Bay Rays
            'WSH': 'WSN',  # Washington Nationals
            'WAS': 'WSN',  # Washington Nationals (alternative)
        }
        
        successful_games = 0
        
        for i, game_date in enumerate(unique_dates):
            try:
                daily_games = schedule_df[schedule_df['Date'] == game_date]
                
                url = "https://statsapi.mlb.com/api/v1/schedule"
                params = {
                    "sportId": 1,
                    "startDate": game_date,
                    "endDate": game_date,
                    "gameType": "R"
                }
                
                response = requests.get(url, params=params, timeout=15)
                response.raise_for_status()
                data = response.json()
                
                for day in data.get("dates", []):
                    for game in day.get("games", []):
                        if game.get("status", {}).get("abstractGameState") != "Final":
                            continue
                        
                        # Get team names and scores
                        home_team_full = game["teams"]["home"]["team"]["name"]
                        away_team_full = game["teams"]["away"]["team"]["name"]
                        
                        # Convert to abbreviations
                        home_abbrev = self.full_name_to_abbrev.get(home_team_full, home_team_full)
                        away_abbrev = self.full_name_to_abbrev.get(away_team_full, away_team_full)
                        
                        # Apply fixes for problematic abbreviations
                        home_abbrev = abbrev_fixes.get(home_abbrev, home_abbrev)
                        away_abbrev = abbrev_fixes.get(away_abbrev, away_abbrev)
                        
                        home_score = game["teams"]["home"].get("score", 0)
                        away_score = game["teams"]["away"].get("score", 0)
                        
                        game_result = {
                            'Date': game_date,
                            'Home': home_abbrev,
                            'Away': away_abbrev,
                            'home_score': home_score,
                            'away_score': away_score,
                            'home_team_won': 1 if home_score > away_score else 0,
                            'Season': season
                        }
                        
                        games_with_results.append(game_result)
                        successful_games += 1
                
                if (i + 1) % 20 == 0 or (i + 1) == len(unique_dates):
                    print(f"    📊 Processed {i+1}/{len(unique_dates)} dates, {successful_games} games with results")
                    
            except Exception as e:
                if self.verbose:
                    print(f"    ⚠️ Error processing {game_date}: {e}")
                continue
        
        result_df = pd.DataFrame(games_with_results)
        if not result_df.empty:
            print(f"  ✅ Successfully fetched results for {len(result_df)} games")
            
            # Debug: Verify team formats
            if self.verbose:
                unique_teams = set(result_df['Home'].unique()) | set(result_df['Away'].unique())
                print(f"  📊 Teams in results (should match pybaseball): {sorted(list(unique_teams))[:10]}...")
        
        return result_df

    def _get_season_schedule_mlb_api(self, season: int, hydrate: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get complete schedule using MLB StatsAPI.
        ENHANCED: Can now hydrate data like probable pitchers.
        
        Args:
            season: The season year to fetch
            hydrate: Optional list of fields to hydrate (e.g., ['probablePitcher'])
        
        Returns:
            DataFrame with game schedule, optionally including hydrated data
        """
        try:
            print(f"  📋 Fetching MLB schedule for {season} using StatsAPI...")
            
            start_date = f"{season}-03-01"  # Start a bit earlier to catch March games
            end_date = f"{season}-10-31"    # End after regular season
            
            url = "https://statsapi.mlb.com/api/v1/schedule"
            params = {
                "sportId": 1,
                "startDate": start_date,
                "endDate": end_date,
                "gameType": "R"  # Regular season games only
            }

            # NEW: Add hydration if requested
            if hydrate:
                params["hydrate"] = ",".join(hydrate)
                print(f"     💧 Hydrating with: {hydrate}")

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            all_games = []
            for day in data.get("dates", []):
                game_date = day.get("date")
                for game in day.get("games", []):
                    if game.get("gameType") != "R":
                        continue
                    
                    # Get full team names from API
                    home_team_full = game["teams"]["home"]["team"]["name"]
                    away_team_full = game["teams"]["away"]["team"]["name"]
                    
                    # Convert to abbreviations using mapping
                    home_abbrev = self.full_name_to_abbrev.get(home_team_full, home_team_full)
                    away_abbrev = self.full_name_to_abbrev.get(away_team_full, away_team_full)
                    
                    # CRITICAL FIX: Normalize specific problematic abbreviations
                    abbrev_fixes = {
                        'CWS': 'CHW',  # Chicago White Sox
                        'KC': 'KCR',   # Kansas City Royals
                        'LA': 'LAD',   # Los Angeles Dodgers
                        'SD': 'SDP',   # San Diego Padres
                        'SF': 'SFG',   # San Francisco Giants
                        'TB': 'TBR',   # Tampa Bay Rays
                        'WSH': 'WSN',  # Washington Nationals
                        'WAS': 'WSN',  # Washington Nationals (alternative)
                    }
                    
                    home_abbrev = abbrev_fixes.get(home_abbrev, home_abbrev)
                    away_abbrev = abbrev_fixes.get(away_abbrev, away_abbrev)
                    
                    game_data = {
                        'Date': game_date,
                        'Home': home_abbrev,
                        'Away': away_abbrev,
                        'game_pk': game.get('gamePk')  # Keep for future use
                    }

                    # NEW: Extract pitcher names if hydrated
                    if hydrate and 'probablePitcher' in hydrate:
                        home_pp = game["teams"]["home"].get("probablePitcher")
                        away_pp = game["teams"]["away"].get("probablePitcher")
                        game_data['home_pitcher'] = (home_pp.get("fullName") if home_pp else "TBD") or "TBD"
                        game_data['away_pitcher'] = (away_pp.get("fullName") if away_pp else "TBD") or "TBD"
                        
                        # Also try to get pitcher IDs if available
                        if home_pp:
                            game_data['home_pitcher_id'] = home_pp.get('id')
                        if away_pp:
                            game_data['away_pitcher_id'] = away_pp.get('id')
                    
                    all_games.append(game_data)
            
            if all_games:
                schedule_df = pd.DataFrame(all_games)
                schedule_df = schedule_df.drop_duplicates().sort_values('Date').reset_index(drop=True)
                print(f"  ✅ Got {len(schedule_df)} regular season games for {season}")
                
                # Show pitcher coverage if hydrated
                if hydrate and 'probablePitcher' in hydrate:
                    has_home = (schedule_df['home_pitcher'] != 'TBD').sum()
                    has_away = (schedule_df['away_pitcher'] != 'TBD').sum()
                    has_both = ((schedule_df['home_pitcher'] != 'TBD') & 
                            (schedule_df['away_pitcher'] != 'TBD')).sum()
                    print(f"     ⚾ Pitcher data: {has_home} home, {has_away} away, {has_both} both")
                
                return schedule_df
            else:
                print(f"  ❌ No regular season games found for {season}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"  ❌ Error fetching {season} schedule from MLB API: {e}")
            return pd.DataFrame()

    def _limit_games_evenly(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Sample games evenly across the season to avoid April bias."""
        if len(df) <= n:
            return df
        
        df_copy = df.copy()
        df_copy['month'] = pd.to_datetime(df_copy['Date']).dt.month
        
        # Sample proportionally from each month, but ensure we don't exceed n total
        sampled_dfs = []
        remaining_samples = n
        months = sorted(df_copy['month'].unique())
        
        for i, month in enumerate(months):
            month_data = df_copy[df_copy['month'] == month]
            
            if i == len(months) - 1:  
                # Last month gets all remaining samples
                month_samples = min(remaining_samples, len(month_data))
            else:
                # Calculate proportional samples for this month
                month_proportion = len(month_data) / len(df_copy)
                month_samples = max(1, min(int(n * month_proportion), remaining_samples, len(month_data)))
            
            if month_samples > 0 and remaining_samples > 0:
                if month_samples >= len(month_data):
                    # Take all games from this month
                    sampled_month = month_data
                else:
                    # Sample from this month
                    sampled_month = month_data.sample(n=month_samples, random_state=42)
                
                sampled_dfs.append(sampled_month)
                remaining_samples -= len(sampled_month)
                
                if remaining_samples <= 0:
                    break
        
        if sampled_dfs:
            result = pd.concat(sampled_dfs, ignore_index=True)
            # Final shuffle and ensure we don't exceed n
            if len(result) > n:
                result = result.sample(n=n, random_state=42)
            return result.drop('month', axis=1).reset_index(drop=True)
        else:
            # Fallback to simple random sampling - ensure we don't sample more than available
            return df.sample(n=min(n, len(df)), random_state=42).reset_index(drop=True)

    def _add_nrfi_outcomes_optimized(self, schedule_df: pd.DataFrame, season: int) -> pd.DataFrame:
        """
        OPTIMIZED VERSION: Batch Statcast calls by date to avoid redundant API calls.
        """
        games_with_nrfi = []
        total_games = len(schedule_df)
        
        print(f"  🔍 Analyzing NRFI outcomes for {total_games} games (optimized approach)...")
        
        # Group games by date to minimize Statcast API calls
        unique_dates = schedule_df['Date'].unique()
        print(f"     Processing {len(unique_dates)} unique game dates...")
        
        successful_outcomes = 0
        
        for i, game_date in enumerate(unique_dates):
            try:
                # Get all games for this date
                daily_games = schedule_df[schedule_df['Date'] == game_date]
                
                # PERFORMANCE FIX: Get Statcast data once per date
                statcast_df = self._get_statcast_for_date(game_date)
                
                if statcast_df is not None and not statcast_df.empty:
                    # Process all games for this date
                    for _, game in daily_games.iterrows():
                        nrfi_outcome = self._get_nrfi_from_statcast_data(
                            statcast_df, 
                            game['Home'],  # These are now full team names
                            game['Away']
                        )
                        
                        if nrfi_outcome is not None:
                            game_data = game.to_dict()
                            game_data['nrfi_result'] = nrfi_outcome
                            game_data['Season'] = season
                            games_with_nrfi.append(game_data)
                            successful_outcomes += 1
                
                # Progress update every 10 dates (more frequent)
                if (i + 1) % 10 == 0 or (i + 1) == len(unique_dates):
                    processed_games = sum(len(schedule_df[schedule_df['Date'] == d]) for d in unique_dates[:i+1])
                    success_rate = (successful_outcomes / processed_games * 100) if processed_games > 0 else 0
                    print(f"    📊 Processed {i+1}/{len(unique_dates)} dates ({(i+1)/len(unique_dates)*100:.1f}%), {processed_games}/{total_games} games, {successful_outcomes} with NRFI data ({success_rate:.1f}% success rate)")
                    
                    # Show current date being processed
                    print(f"       🗓️  Currently processing: {game_date}")
            except Exception as e:
                if self.verbose:
                    print(f"    ⚠️ Error processing {game_date}: {e}")
                continue
        
        result_df = pd.DataFrame(games_with_nrfi)
        if not result_df.empty:
            print(f"  ✅ Successfully determined NRFI outcomes for {len(result_df)} games")
        
        return result_df

    def _get_statcast_for_date(self, date: str) -> Optional[pd.DataFrame]:
        """
        PERFORMANCE FIX: Cache Statcast data by date to avoid repeated API calls.
        Added timeout and retry logic for large queries.
        """
        cache_key = f"statcast_day_{date}"
        cache_path = self._get_cache_path(cache_key)
        
        # Check cache first (7 day cache)
        if self._is_cache_valid(cache_path, 86400 * 7):
            cached_data = self._load_cache(cache_path)
            if cached_data is not None:
                return cached_data
        
        # Try fetching with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"       📥 Fetching Statcast data for {date} (attempt {attempt + 1}/{max_retries})...")
                
                # Fetch fresh data from Statcast with timeout
                statcast_df = pyb.statcast(start_dt=date, end_dt=date)
                
                if not statcast_df.empty:
                    print(f"       ✅ Got {len(statcast_df)} plays for {date}")
                    # Cache the data
                    self._save_cache(statcast_df, cache_path)
                    return statcast_df
                else:
                    print(f"       ⚠️  No Statcast data found for {date}")
                    return None
                    
            except Exception as e:
                print(f"       ❌ Attempt {attempt + 1} failed for {date}: {str(e)[:100]}...")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5  # Exponential backoff
                    print(f"       ⏳ Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"       🚫 All attempts failed for {date}")
        
        return None

    def _get_nrfi_from_statcast_data(self, statcast_df: pd.DataFrame, 
                                    home_team: str, away_team: str) -> Optional[int]:
        """
        CRITICAL FIX: Much more aggressive team name matching and debugging.
        """
        try:
            if statcast_df.empty:
                return None
            
            if self.verbose:
                unique_games = statcast_df.groupby(['home_team', 'away_team']).size()
                print(f"        Available games: {list(unique_games.index)}")
                print(f"        Looking for: {home_team} vs {away_team}")
            
            # CRITICAL FIX: More aggressive name normalization
            def normalize_name(name):
                if pd.isna(name):
                    return ""
                name = str(name).lower()
                # Remove common variations
                name = name.replace(".", "").replace("-", " ").replace("'", "")
                name = name.replace("los angeles", "la").replace("new york", "ny")
                name = name.replace("san francisco", "sf").replace("san diego", "sd")
                name = name.replace("kansas city", "kc").replace("tampa bay", "tb")
                name = name.strip()
                return name
            
            home_norm = normalize_name(home_team)
            away_norm = normalize_name(away_team)
            
            # Add normalized columns to statcast data
            statcast_df = statcast_df.copy()
            statcast_df['home_norm'] = statcast_df['home_team'].apply(normalize_name)
            statcast_df['away_norm'] = statcast_df['away_team'].apply(normalize_name)
            
            # Try exact match first
            game_data = statcast_df[
                (statcast_df['home_norm'] == home_norm) & 
                (statcast_df['away_norm'] == away_norm)
            ]
            
            # If no exact match, try partial matches
            if game_data.empty:
                # Try matching key words
                home_words = set(home_norm.split())
                away_words = set(away_norm.split())
                
                for (h_norm, a_norm), _ in statcast_df.groupby(['home_norm', 'away_norm']).size().items():
                    sc_home_words = set(h_norm.split())
                    sc_away_words = set(a_norm.split())
                    
                    # If we find overlapping words, it might be a match
                    if (len(home_words & sc_home_words) >= 1 and 
                        len(away_words & sc_away_words) >= 1):
                        
                        game_data = statcast_df[
                            (statcast_df['home_norm'] == h_norm) & 
                            (statcast_df['away_norm'] == a_norm)
                        ]
                        if self.verbose:
                            print(f"        🎯 Partial match found: {h_norm} vs {a_norm}")
                        break
            
            # Last resort: if only one game on this date, use it
            if game_data.empty:
                unique_games = statcast_df.groupby(['home_norm', 'away_norm']).size()
                if len(unique_games) == 1:
                    home_fallback, away_fallback = unique_games.index[0]
                    game_data = statcast_df[
                        (statcast_df['home_norm'] == home_fallback) & 
                        (statcast_df['away_norm'] == away_fallback)
                    ]
                    if self.verbose:
                        print(f"        🎯 Single game fallback: {home_fallback} vs {away_fallback}")
            
            if game_data.empty:
                if self.verbose:
                    print(f"        ❌ No match found for: {home_team} vs {away_team}")
                return None
            
            # Get first inning data
            first_inning = game_data[game_data['inning'] == 1].copy()
            
            if first_inning.empty:
                return 1  # No recorded plays in 1st inning = NRFI
            
            # CRITICAL FIX: Sort by play sequence and use score diffs
            first_inning = first_inning.sort_values(['inning_topbot', 'at_bat_number', 'pitch_number'])
            
            # Check if any runs were scored using home_score/away_score progression
            if 'home_score' in first_inning.columns and 'away_score' in first_inning.columns:
                # Look for any increase in either team's score during the first inning
                home_scored = (first_inning['home_score'].diff().fillna(0) > 0).any()
                away_scored = (first_inning['away_score'].diff().fillna(0) > 0).any()
                
                result = 0 if home_scored or away_scored else 1
                if self.verbose:
                    print(f"        ✅ NRFI result: {result} ({'YRFI' if result == 0 else 'NRFI'})")
                
                return result
            
            return 1  # Default to NRFI if no clear scoring detected
            
        except Exception as e:
            if self.verbose:
                print(f"        ⚠️ Error analyzing NRFI for {home_team} vs {away_team}: {e}")
            return None

    def get_pitcher_id(self, pitcher_name: str, season: int) -> Optional[int]:
        """Get pitcher ID from name with season validation."""
        try:
            name_parts = pitcher_name.split()
            if len(name_parts) >= 2:
                player_ids = pyb.playerid_lookup(
                    name_parts[-1],  # Last name
                    name_parts[0]    # First name
                )
                
                if player_ids.empty:
                    return None
                
                if len(player_ids) > 1:
                    active_players = player_ids[
                        (player_ids['mlb_played_first'] <= season) & 
                        (player_ids['mlb_played_last'] >= season)
                    ]
                    
                    if not active_players.empty:
                        return active_players.iloc[0]['key_mlbam']
                else:
                    player = player_ids.iloc[0]
                    if player['mlb_played_first'] <= season <= player['mlb_played_last']:
                        return player['key_mlbam']
                        
        except Exception as e:
            print(f"Error looking up pitcher ID for {pitcher_name}: {e}")
            
        return None

    def get_statcast_data_for_nrfi(self, pitcher_name: str, season: int) -> pd.DataFrame:
        """Fetch granular Statcast data for NRFI model."""
        cache_key = f"statcast_{pitcher_name.replace(' ', '_')}_{season}"
        cache_path = self._get_cache_path(cache_key)
        
        if self._is_cache_valid(cache_path, self.settings.CACHE_DURATION):
            cached_data = self._load_cache(cache_path)
            if cached_data is not None:
                return cached_data
        
        try:
            pitcher_id = self.get_pitcher_id(pitcher_name, season)
            
            if pitcher_id:
                start_date = f"{season}-03-01"
                end_date = f"{season}-11-01"
                
                statcast_data = pyb.statcast_pitcher(
                    start_dt=start_date,
                    end_dt=end_date,
                    player_id=pitcher_id
                )
                
                if not statcast_data.empty:
                    self._save_cache(statcast_data, cache_path)
                    return statcast_data
                    
        except Exception as e:
            print(f"Error fetching Statcast data for {pitcher_name}: {e}")
            
        return pd.DataFrame()

    def get_current_weather(self, city: str) -> Dict[str, Any]:
        """Get current weather conditions for a city."""
        return {
            'temperature': None,
            'wind_speed': None,
            'humidity': None,
            'conditions': None
        }



