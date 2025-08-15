# src/models/game_winner_model/feature_engineer.py

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')

class GameWinnerFeatureEngineer:
    """
    Creates features for the game winner prediction model.
    Focuses on team performance differentials, pitcher quality, and recent form.
    ENHANCED VERSION: Uses DataManager's enhanced normalizer for better pitcher matching.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        
        # Key batting metrics (higher is better)
        self.batting_metrics = ['R', 'H', 'HR', 'RBI', 'SB', 'BA', 'OBP', 'SLG', 'OPS']
        
        # Key pitching metrics (lower is better for ERA, WHIP; higher for SO)
        self.pitching_metrics = ['ERA', 'WHIP', 'SO', 'SV', 'IP']
        
        # Features we'll create
        self.feature_columns = []
        
        # Track merge success rates
        self.merge_stats = {}
    
    def engineer_features(self, 
                         games_df: pd.DataFrame, 
                         team_stats: pd.DataFrame, 
                         pitcher_stats: pd.DataFrame,
                         data_manager=None) -> pd.DataFrame:
        """
        Creates features for model training or prediction.
        
        Args:
            games_df: DataFrame with Date, Home, Away columns (and optionally home_team_won for training)
            team_stats: Historical team statistics from pybaseball
            pitcher_stats: Historical pitcher statistics from pybaseball
            data_manager: Optional DataManager instance with enhanced normalizer
            
        Returns:
            DataFrame with engineered features
        """
        if self.verbose:
            print("\n🔧 Starting Feature Engineering...")
            print(f"   Input games: {len(games_df)}")
            print(f"   Team stats available: {not team_stats.empty}")
            print(f"   Pitcher stats available: {not pitcher_stats.empty}")
            
            # Debug: Show what columns we have
            print(f"\n   Games columns: {list(games_df.columns)}")
            if not team_stats.empty:
                print(f"   Team stats columns (sample): {list(team_stats.columns[:10])}")
            if not pitcher_stats.empty:
                print(f"   Pitcher stats columns (sample): {list(pitcher_stats.columns[:10])}")
        
        features_df = games_df.copy()
        
        # 1. Add team statistics with robust matching
        features_df = self._add_team_stats_robust(features_df, team_stats)
        
        # 2. Add pitcher statistics if available (using enhanced normalizer if provided)
        if not pitcher_stats.empty and ('home_pitcher' in games_df.columns or 'Home_SP' in games_df.columns):
            features_df = self._add_pitcher_stats_robust(features_df, pitcher_stats, data_manager)
        
        # 3. Create differential features (Home advantage)
        features_df = self._create_differential_features(features_df)
        
        # 4. Add derived features
        features_df = self._add_derived_features(features_df)
        
        # 5. Handle missing values
        features_df = self._handle_missing_values(features_df)
        
        # Store feature columns for later use
        self.feature_columns = [col for col in features_df.columns 
                               if any(keyword in col for keyword in ['_diff', '_ratio', 'home_advantage', '_pct'])]
        
        if self.verbose:
            print(f"\n✅ Feature Engineering Complete!")
            print(f"   Created {len(self.feature_columns)} features")
            self._print_merge_stats()
            self._print_feature_quality(features_df)
        
        return features_df
    
    def _add_team_stats_robust(self, games_df: pd.DataFrame, team_stats: pd.DataFrame) -> pd.DataFrame:
        """Add team batting and pitching statistics with robust team name matching."""
        if team_stats.empty:
            print("⚠️ No team stats available")
            return games_df
        
        # Get the most recent season's stats for each team
        latest_stats = team_stats.sort_values('Season').groupby('Team').last().reset_index()
        
        if self.verbose:
            print(f"\n📊 Adding team statistics...")
            print(f"   Teams in games: {sorted(games_df['Home'].unique())[:5]}...")
            print(f"   Teams in stats: {sorted(latest_stats['Team'].unique())[:5]}...")
        
        # Create a mapping for team names (handle variations)
        team_mapping = self._create_team_mapping(games_df, latest_stats)
        
        # Apply mapping to games
        games_df['Home_mapped'] = games_df['Home'].map(team_mapping).fillna(games_df['Home'])
        games_df['Away_mapped'] = games_df['Away'].map(team_mapping).fillna(games_df['Away'])
        
        # Merge home team stats
        pre_merge_len = len(games_df)
        games_df = pd.merge(
            games_df,
            latest_stats.add_suffix('_home'),
            left_on='Home_mapped',
            right_on='Team_home',
            how='left'
        )
        
        # Track merge success
        home_merge_success = games_df['Team_home'].notna().sum()
        
        # Merge away team stats
        games_df = pd.merge(
            games_df,
            latest_stats.add_suffix('_away'),
            left_on='Away_mapped',
            right_on='Team_away',
            how='left'
        )
        
        # Track merge success
        away_merge_success = games_df['Team_away'].notna().sum()
        
        self.merge_stats['team_home'] = home_merge_success / pre_merge_len * 100
        self.merge_stats['team_away'] = away_merge_success / pre_merge_len * 100
        
        if self.verbose:
            print(f"   ✅ Home team merge success: {home_merge_success}/{pre_merge_len} ({self.merge_stats['team_home']:.1f}%)")
            print(f"   ✅ Away team merge success: {away_merge_success}/{pre_merge_len} ({self.merge_stats['team_away']:.1f}%)")
            
            # Debug: Find teams that aren't matching
            missing_home_teams = games_df[games_df['Team_home'].isna()][['Home']].drop_duplicates()
            if not missing_home_teams.empty:
                print(f"   ❌ Home teams not matching: {missing_home_teams['Home'].tolist()}")
            
            missing_away_teams = games_df[games_df['Team_away'].isna()][['Away']].drop_duplicates()
            if not missing_away_teams.empty:
                print(f"   ❌ Away teams not matching: {missing_away_teams['Away'].tolist()}")
        
        # Clean up temporary columns
        games_df = games_df.drop(['Home_mapped', 'Away_mapped'], axis=1, errors='ignore')
        
        return games_df
    
    def _add_pitcher_stats_robust(self, games_df: pd.DataFrame, pitcher_stats: pd.DataFrame, 
                                  data_manager=None) -> pd.DataFrame:
        """
        Add pitcher statistics with enhanced name matching using DataManager's normalizer.
        
        Args:
            games_df: DataFrame with pitcher columns
            pitcher_stats: DataFrame with pitcher statistics
            data_manager: Optional DataManager instance with enhanced normalizer
        
        Returns:
            DataFrame with pitcher stats merged
        """
        if pitcher_stats.empty:
            return games_df
        
        if self.verbose:
            print(f"\n⚾ Adding pitcher statistics...")
        
        # Check which pitcher columns exist
        home_pitcher_col = 'home_pitcher' if 'home_pitcher' in games_df.columns else 'Home_SP' if 'Home_SP' in games_df.columns else None
        away_pitcher_col = 'away_pitcher' if 'away_pitcher' in games_df.columns else 'Away_SP' if 'Away_SP' in games_df.columns else None
        
        if not home_pitcher_col:
            print("   ⚠️ No pitcher columns found in games data")
            return games_df
        
        # Get most recent stats for each pitcher
        latest_pitcher_stats = pitcher_stats.sort_values('Season').groupby('Name').last().reset_index()
        
        # ENHANCED: Use DataManager's normalizer if available
        if data_manager and hasattr(data_manager, 'normalizer') and data_manager.normalizer:
            print("   🔧 Using enhanced pitcher name normalization from DataManager...")
            
            # Normalize pitcher names in games using enhanced normalizer
            games_df['home_pitcher_normalized'] = games_df[home_pitcher_col].apply(
                lambda x: data_manager.normalizer.pitcher_normalizer.normalize_pitcher_name(x) 
                if pd.notna(x) and x != 'TBD' else x
            )
            
            if away_pitcher_col:
                games_df['away_pitcher_normalized'] = games_df[away_pitcher_col].apply(
                    lambda x: data_manager.normalizer.pitcher_normalizer.normalize_pitcher_name(x)
                    if pd.notna(x) and x != 'TBD' else x
                )
            
            # Normalize pitcher names in stats
            latest_pitcher_stats['Name_normalized'] = latest_pitcher_stats['Name'].apply(
                lambda x: data_manager.normalizer.pitcher_normalizer.normalize_pitcher_name(x)
                if pd.notna(x) else x
            )
            
            # Merge on normalized names
            pre_merge_len = len(games_df)
            
            # Home pitcher merge
            games_df = pd.merge(
                games_df,
                latest_pitcher_stats.add_suffix('_home_pitcher'),
                left_on='home_pitcher_normalized',
                right_on='Name_normalized_home_pitcher',
                how='left'
            )
            
            home_pitcher_success = games_df['Name_home_pitcher'].notna().sum()
            
            # Away pitcher merge
            if away_pitcher_col:
                games_df = pd.merge(
                    games_df,
                    latest_pitcher_stats.add_suffix('_away_pitcher'),
                    left_on='away_pitcher_normalized',
                    right_on='Name_normalized_away_pitcher',
                    how='left'
                )
                away_pitcher_success = games_df['Name_away_pitcher'].notna().sum()
            else:
                away_pitcher_success = 0
            
            # If still have unmatched, try fuzzy matching
            home_unmatched = games_df['Name_home_pitcher'].isna() & games_df['home_pitcher_normalized'].notna() & (games_df['home_pitcher_normalized'] != 'TBD')
            away_unmatched = games_df['Name_away_pitcher'].isna() & games_df.get('away_pitcher_normalized', pd.Series()).notna() if away_pitcher_col else pd.Series([False]*len(games_df))
            
            if home_unmatched.any() or away_unmatched.any():
                print(f"   🎯 Attempting fuzzy matching for remaining {home_unmatched.sum() + away_unmatched.sum()} pitchers...")
                
                # Use the enhanced normalizer's fuzzy matching
                for idx in games_df[home_unmatched].index:
                    pitcher_name = games_df.loc[idx, home_pitcher_col]
                    similar = data_manager.normalizer.pitcher_normalizer.find_similar_pitchers(pitcher_name, n=1)
                    if similar and similar[0][1] > 0.85:
                        best_match = similar[0][0]
                        matches = latest_pitcher_stats[latest_pitcher_stats['Name'] == best_match]
                        if not matches.empty:
                            for col in latest_pitcher_stats.columns:
                                if col not in ['Name_normalized']:
                                    games_df.loc[idx, f'{col}_home_pitcher'] = matches.iloc[0][col]
                            home_pitcher_success += 1
                            if self.verbose:
                                print(f"      Matched: '{pitcher_name}' → '{best_match}'")
                
                if away_pitcher_col:
                    for idx in games_df[away_unmatched].index:
                        pitcher_name = games_df.loc[idx, away_pitcher_col]
                        similar = data_manager.normalizer.pitcher_normalizer.find_similar_pitchers(pitcher_name, n=1)
                        if similar and similar[0][1] > 0.85:
                            best_match = similar[0][0]
                            matches = latest_pitcher_stats[latest_pitcher_stats['Name'] == best_match]
                            if not matches.empty:
                                for col in latest_pitcher_stats.columns:
                                    if col not in ['Name_normalized']:
                                        games_df.loc[idx, f'{col}_away_pitcher'] = matches.iloc[0][col]
                                away_pitcher_success += 1
                                if self.verbose:
                                    print(f"      Matched: '{pitcher_name}' → '{best_match}'")
            
            # Clean up normalized columns
            cols_to_drop = ['home_pitcher_normalized', 'away_pitcher_normalized', 
                           'Name_normalized_home_pitcher', 'Name_normalized_away_pitcher']
            games_df = games_df.drop([c for c in cols_to_drop if c in games_df.columns], axis=1)
            
        else:
            # FALLBACK: Use original basic normalization if DataManager not available
            print("   ⚠️ Enhanced normalizer not available, using basic normalization...")
            
            # Use the original normalization logic
            games_df['home_pitcher_norm'] = games_df[home_pitcher_col].apply(self._normalize_pitcher_name)
            if away_pitcher_col:
                games_df['away_pitcher_norm'] = games_df[away_pitcher_col].apply(self._normalize_pitcher_name)
            
            latest_pitcher_stats['Name_norm'] = latest_pitcher_stats['Name'].apply(self._normalize_pitcher_name)
            
            # Create pitcher name mapping
            pitcher_mapping = self._create_pitcher_mapping(games_df, latest_pitcher_stats, home_pitcher_col, away_pitcher_col)
            
            # Apply mapping
            games_df['home_pitcher_mapped'] = games_df['home_pitcher_norm'].map(pitcher_mapping).fillna(games_df['home_pitcher_norm'])
            if away_pitcher_col:
                games_df['away_pitcher_mapped'] = games_df['away_pitcher_norm'].map(pitcher_mapping).fillna(games_df['away_pitcher_norm'])
            
            # Merge home pitcher stats
            pre_merge_len = len(games_df)
            games_df = pd.merge(
                games_df,
                latest_pitcher_stats.add_suffix('_home_pitcher'),
                left_on='home_pitcher_mapped',
                right_on='Name_norm_home_pitcher',
                how='left'
            )
            
            home_pitcher_success = games_df['Name_home_pitcher'].notna().sum()
            
            # Merge away pitcher stats
            if away_pitcher_col:
                games_df = pd.merge(
                    games_df,
                    latest_pitcher_stats.add_suffix('_away_pitcher'),
                    left_on='away_pitcher_mapped',
                    right_on='Name_norm_away_pitcher',
                    how='left'
                )
                away_pitcher_success = games_df['Name_away_pitcher'].notna().sum()
            else:
                away_pitcher_success = 0
            
            # Clean up temporary columns
            cols_to_drop = ['home_pitcher_norm', 'away_pitcher_norm', 'home_pitcher_mapped', 'away_pitcher_mapped']
            games_df = games_df.drop([c for c in cols_to_drop if c in games_df.columns], axis=1)
        
        # Update merge stats
        pre_merge_len = len(games_df)
        self.merge_stats['pitcher_home'] = home_pitcher_success / pre_merge_len * 100
        self.merge_stats['pitcher_away'] = away_pitcher_success / pre_merge_len * 100 if away_pitcher_col else 0
        
        if self.verbose:
            print(f"   ✅ Home pitcher merge success: {home_pitcher_success}/{pre_merge_len} ({self.merge_stats['pitcher_home']:.1f}%)")
            if away_pitcher_col:
                print(f"   ✅ Away pitcher merge success: {away_pitcher_success}/{pre_merge_len} ({self.merge_stats['pitcher_away']:.1f}%)")
        
        return games_df
    
    def _normalize_pitcher_name(self, name: str) -> str:
        """Normalize pitcher name for better matching."""
        if pd.isna(name) or name == 'TBD':
            return 'TBD'
        
        # Convert to lowercase and remove extra spaces
        name = str(name).lower().strip()
        
        # Remove common suffixes
        name = name.replace(' jr.', '').replace(' jr', '')
        name = name.replace(' iii', '').replace(' ii', '')
        
        # Remove periods from initials
        name = name.replace('.', '')
        
        # Normalize accented characters (basic)
        replacements = {
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
            'ñ': 'n', 'ü': 'u'
        }
        for old, new in replacements.items():
            name = name.replace(old, new)
        
        return name
    
    def _create_team_mapping(self, games_df: pd.DataFrame, team_stats: pd.DataFrame) -> Dict[str, str]:
        """Create mapping between game team names and stats team names."""
        mapping = {}
        
        # Get unique teams from games
        game_teams = set(games_df['Home'].unique()) | set(games_df['Away'].unique())
        stat_teams = set(team_stats['Team'].unique())
        
        for game_team in game_teams:
            # Try exact match first
            if game_team in stat_teams:
                mapping[game_team] = game_team
            else:
                # Try case-insensitive match
                for stat_team in stat_teams:
                    if game_team.upper() == stat_team.upper():
                        mapping[game_team] = stat_team
                        break
        
        return mapping
    
    def _create_pitcher_mapping(self, games_df: pd.DataFrame, pitcher_stats: pd.DataFrame,
                               home_col: str, away_col: str) -> Dict[str, str]:
        """Create mapping between game pitcher names and stats pitcher names."""
        mapping = {}
        
        # Get unique pitchers from games
        game_pitchers = set(games_df['home_pitcher_norm'].unique())
        if away_col:
            game_pitchers |= set(games_df['away_pitcher_norm'].unique())
        
        stat_pitchers = pitcher_stats['Name_norm'].unique()
        
        for game_pitcher in game_pitchers:
            if game_pitcher == 'TBD' or pd.isna(game_pitcher):
                continue
            
            # Try exact match first
            if game_pitcher in stat_pitchers:
                mapping[game_pitcher] = game_pitcher
            else:
                # Try fuzzy matching (last name match)
                game_last = game_pitcher.split()[-1] if game_pitcher else ''
                for stat_pitcher in stat_pitchers:
                    stat_last = stat_pitcher.split()[-1] if stat_pitcher else ''
                    if game_last and stat_last and game_last == stat_last:
                        # Verify first initial matches
                        if game_pitcher[0] == stat_pitcher[0]:
                            mapping[game_pitcher] = stat_pitcher
                            break
        
        return mapping
    
    def _create_differential_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create differential features (Home - Away)."""
        
        # Team batting differentials (higher is better for home team)
        for metric in self.batting_metrics:
            home_col = f'{metric}_batting_home'
            away_col = f'{metric}_batting_away'
            if home_col in df.columns and away_col in df.columns:
                df[f'{metric}_batting_diff'] = df[home_col] - df[away_col]
        
        # Team pitching differentials (lower ERA/WHIP is better)
        for metric in self.pitching_metrics:
            home_col = f'{metric}_pitching_home'
            away_col = f'{metric}_pitching_away'
            if home_col in df.columns and away_col in df.columns:
                if metric in ['ERA', 'WHIP']:
                    # For these, lower is better, so flip the diff
                    df[f'{metric}_pitching_diff'] = df[away_col] - df[home_col]
                else:
                    df[f'{metric}_pitching_diff'] = df[home_col] - df[away_col]
        
        # Starting pitcher differentials (if available)
        if 'ERA_home_pitcher' in df.columns and 'ERA_away_pitcher' in df.columns:
            df['starter_ERA_diff'] = df['ERA_away_pitcher'] - df['ERA_home_pitcher']
            
        if 'WHIP_home_pitcher' in df.columns and 'WHIP_away_pitcher' in df.columns:
            df['starter_WHIP_diff'] = df['WHIP_away_pitcher'] - df['WHIP_home_pitcher']
            
        if 'SO_home_pitcher' in df.columns and 'SO_away_pitcher' in df.columns:
            df['starter_SO_diff'] = df['SO_home_pitcher'] - df['SO_away_pitcher']
        
        if 'FIP_home_pitcher' in df.columns and 'FIP_away_pitcher' in df.columns:
            df['starter_FIP_diff'] = df['FIP_away_pitcher'] - df['FIP_home_pitcher']
        
        return df
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features like win percentage differential."""
        
        # Win percentage differential
        if 'W_batting_home' in df.columns and 'L_batting_home' in df.columns:
            df['win_pct_home'] = df['W_batting_home'] / (df['W_batting_home'] + df['L_batting_home'] + 0.001)
            
        if 'W_batting_away' in df.columns and 'L_batting_away' in df.columns:
            df['win_pct_away'] = df['W_batting_away'] / (df['W_batting_away'] + df['L_batting_away'] + 0.001)
            
        if 'win_pct_home' in df.columns and 'win_pct_away' in df.columns:
            df['win_pct_diff'] = df['win_pct_home'] - df['win_pct_away']
        
        # Home field advantage (constant feature)
        df['home_advantage'] = 1  # Home teams historically win ~54% of games
        
        # Team quality composite (if we have OPS and ERA)
        if 'OPS_batting_home' in df.columns and 'ERA_pitching_home' in df.columns:
            # Normalize OPS (higher is better) and ERA (lower is better) to 0-1 scale
            # Using typical ranges: OPS 0.600-0.850, ERA 3.0-5.5
            df['team_quality_home'] = (
                (df['OPS_batting_home'] - 0.600) / 0.250 +  # Normalize OPS
                (5.5 - df['ERA_pitching_home']) / 2.5        # Invert and normalize ERA
            ) / 2
            
        if 'OPS_batting_away' in df.columns and 'ERA_pitching_away' in df.columns:
            df['team_quality_away'] = (
                (df['OPS_batting_away'] - 0.600) / 0.250 +
                (5.5 - df['ERA_pitching_away']) / 2.5
            ) / 2
            
        if 'team_quality_home' in df.columns and 'team_quality_away' in df.columns:
            df['team_quality_diff'] = df['team_quality_home'] - df['team_quality_away']
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features intelligently."""
        # For numerical columns, fill with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        # Track which columns have missing values
        missing_counts = {}
        
        for col in numerical_cols:
            if df[col].isnull().any():
                missing_count = df[col].isnull().sum()
                missing_pct = missing_count / len(df) * 100
                missing_counts[col] = missing_pct
                
                # Use different strategies based on column type
                if 'pitcher' in col.lower():
                    # For pitcher stats, use league average approximation
                    fill_value = self._get_pitcher_default(col)
                elif 'diff' in col:
                    # For differential features, 0 means no advantage
                    fill_value = 0
                else:
                    # For other features, use median
                    median_val = df[col].median()
                    fill_value = median_val if not pd.isna(median_val) else 0
                
                df[col] = df[col].fillna(fill_value)
        
        if self.verbose and missing_counts:
            print(f"\n⚠️ Missing value imputation:")
            top_missing = sorted(missing_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            for col, pct in top_missing:
                print(f"   {col}: {pct:.1f}% missing")
        
        return df
    
    def _get_pitcher_default(self, col_name: str) -> float:
        """Get sensible default values for pitcher statistics."""
        defaults = {
            'ERA': 4.50,    # League average ERA
            'WHIP': 1.30,   # League average WHIP
            'FIP': 4.50,    # League average FIP
            'SO': 150,      # Typical strikeouts for starter
            'BB': 50,       # Typical walks
            'IP': 150,      # Typical innings pitched
            'W': 8,         # Average wins
            'L': 8,         # Average losses
        }
        
        for stat, value in defaults.items():
            if stat in col_name:
                return value
        
        return 0
    
    def _print_merge_stats(self):
        """Print statistics about merge success rates."""
        if not self.merge_stats:
            return
        
        print(f"\n📊 Merge Success Summary:")
        for merge_type, success_rate in self.merge_stats.items():
            emoji = "✅" if success_rate > 80 else "⚠️" if success_rate > 50 else "❌"
            print(f"   {emoji} {merge_type}: {success_rate:.1f}%")
    
    def _print_feature_quality(self, df: pd.DataFrame):
        """Print statistics about feature quality."""
        if not self.feature_columns:
            return
        
        print(f"\n🎯 Feature Quality Check:")
        
        # Check for features with no variance
        low_variance_features = []
        for col in self.feature_columns:
            if col in df.columns:
                if df[col].std() < 0.01:
                    low_variance_features.append(col)
        
        if low_variance_features:
            print(f"   ⚠️ Low variance features: {len(low_variance_features)}")
            for feat in low_variance_features[:3]:
                print(f"      - {feat}")
        
        # Check for highly correlated features
        feature_subset = df[self.feature_columns].select_dtypes(include=[np.number])
        if not feature_subset.empty:
            corr_matrix = feature_subset.corr().abs()
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.95:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
            
            if high_corr_pairs:
                print(f"   ⚠️ Highly correlated feature pairs: {len(high_corr_pairs)}")
                for pair in high_corr_pairs[:3]:
                    print(f"      - {pair[0]} <-> {pair[1]}")
        
        # Show top features by variance (as a proxy for importance)
        feature_vars = []
        for col in self.feature_columns:
            if col in df.columns:
                feature_vars.append((col, df[col].std()))
        
        feature_vars.sort(key=lambda x: x[1], reverse=True)
        print(f"\n   📈 Top features by variance:")
        for feat, var in feature_vars[:5]:
            print(f"      - {feat}: {var:.3f}")
    
    def get_feature_columns(self) -> List[str]:
        """Return the list of feature columns used."""
        return self.feature_columns
