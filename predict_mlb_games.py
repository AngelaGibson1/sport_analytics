#!/usr/bin/env python3
"""
MLB Game Prediction Script - Complete Working Version
Predicts MLB games using the trained model
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime, timedelta
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.data_manager import DataManager
from src.data.config import Settings
from src.models.game_winner_model.feature_engineer import GameWinnerFeatureEngineer


class MLBGamePredictor:
    """MLB Game Predictor with feature alignment."""
    
    def __init__(self):
        """Initialize the predictor."""
        print("🚀 Initializing MLB Game Predictor...")
        
        self.settings = Settings()
        self.data_manager = DataManager(self.settings)
        self.feature_engineer = GameWinnerFeatureEngineer(verbose=False)
        
        # Load the model
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model, scaler, and feature names."""
        model_dir = Path('models/game_winner')
        
        # Try enhanced model first
        enhanced_model_path = model_dir / 'best_model.pkl'
        if enhanced_model_path.exists():
            print("📦 Loading enhanced model...")
            try:
                with open(enhanced_model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data['model']
                    self.scaler = model_data.get('scaler')
                    self.feature_names = model_data.get('feature_names', [])
                print(f"✅ Model loaded: {type(self.model).__name__}")
                print(f"   Features: {len(self.feature_names)}")
                return
            except Exception as e:
                print(f"⚠️ Could not load enhanced model: {e}")
        
        # Fall back to legacy model
        print("📦 Loading legacy model...")
        model_path = model_dir / 'random_forest_model.pkl'
        scaler_path = model_dir / 'scaler.pkl'
        features_path = model_dir / 'feature_names.pkl'
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"No model found! Please run:\n"
                f"  python scripts/fix_models_and_predict.py --retrain"
            )
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        
        if features_path.exists():
            with open(features_path, 'rb') as f:
                self.feature_names = pickle.load(f)
        
        print(f"✅ Model loaded: {type(self.model).__name__}")
        print(f"   Features: {len(self.feature_names)}")
    
    def predict_games(self, game_date: str = None) -> pd.DataFrame:
        """Predict games for a given date."""
        if game_date is None:
            game_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"\n🗓️  Getting games for {game_date}...")
        
        # Use the correct method from DataManager - get_enhanced_daily_schedule
        games_df = self.data_manager.get_enhanced_daily_schedule(game_date)
        
        if games_df is None or (hasattr(games_df, 'empty') and games_df.empty):
            print("❌ No games found for this date.")
            return pd.DataFrame()
        
        print(f"✅ Found {len(games_df)} games")
        
        # CRITICAL FIX: Clean the schedule data
        games_df = self.clean_schedule_data(games_df)
        print(f"   After cleaning: {len(games_df)} unique games")
        
        # Ensure we have the required columns for predictions
        if 'Home' not in games_df.columns or 'Away' not in games_df.columns:
            print("❌ Games data missing required columns (Home/Away)")
            return pd.DataFrame()
        
        # Add placeholder columns if they don't exist
        if 'home_pitcher' not in games_df.columns:
            games_df['home_pitcher'] = 'Unknown'
        if 'away_pitcher' not in games_df.columns:
            games_df['away_pitcher'] = 'Unknown'
        if 'home_pitcher_id' not in games_df.columns:
            games_df['home_pitcher_id'] = None
        if 'away_pitcher_id' not in games_df.columns:
            games_df['away_pitcher_id'] = None
        
        # Add dummy scores for feature engineering (will be ignored in predictions)
        games_df['home_score'] = 0
        games_df['away_score'] = 0
        games_df['home_team_won'] = 0
        
        # Get current season
        current_season = datetime.strptime(game_date, '%Y-%m-%d').year
        games_df['Season'] = current_season
        
        # Load team and pitcher stats
        print("📊 Loading team statistics...")
        team_stats = self.data_manager.get_historical_team_data([current_season-1, current_season])
        
        print("⚾ Loading pitcher statistics...")
        pitcher_stats = self.data_manager.get_pitcher_stats_for_analysis([current_season-1, current_season])
        
        # Engineer features
        print("🔧 Engineering features...")
        features_df = self.feature_engineer.engineer_features(
            games_df=games_df,
            team_stats=team_stats,
            pitcher_stats=pitcher_stats,
            data_manager=self.data_manager
        )
        
        # CRITICAL: Remove any duplicates that appeared during feature engineering
        if len(features_df) > len(games_df):
            print(f"   ⚠️ Feature engineering created duplicates: {len(features_df)} rows from {len(games_df)} games")
            # Keep only unique games based on Home/Away
            features_df = features_df.drop_duplicates(subset=['Home', 'Away'])
            print(f"   ✅ Removed duplicates: {len(features_df)} unique games")
        
        # Align features
        print("🎯 Aligning features with model...")
        X = self.align_features_for_prediction(features_df)
        
        # Scale if needed
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # Make predictions
        print("🎲 Making predictions...")
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Create results
        results = self.create_results_dataframe(
            games_df, features_df, predictions, probabilities
        )
        
        return results
    
    def align_features_for_prediction(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Align features with what the model expects."""
        # Remove non-feature columns
        exclude_cols = ['home_team_won', 'Date', 'Home', 'Away', 'Season', 
                       'home_score', 'away_score', 'home_pitcher', 'away_pitcher',
                       'home_pitcher_id', 'away_pitcher_id']
        
        # Get only numeric columns
        numeric_cols = features_df.select_dtypes(exclude=['object']).columns.tolist()
        available_features = [col for col in numeric_cols if col not in exclude_cols]
        
        # Create aligned dataframe with expected features
        aligned_df = pd.DataFrame(index=features_df.index)
        
        missing_count = 0
        for feature in self.feature_names:
            if feature in available_features:
                aligned_df[feature] = features_df[feature]
            else:
                # Fill missing features with 0
                aligned_df[feature] = 0
                missing_count += 1
        
        if missing_count > 0:
            print(f"   ⚠️ {missing_count}/{len(self.feature_names)} features missing (filled with 0)")
        
        # Fill any NaN values
        aligned_df = aligned_df.fillna(0)
        
        return aligned_df
    
    def create_results_dataframe(self, games_df, features_df, predictions, probabilities):
        """Create a nicely formatted results dataframe with odds validation."""
        results = pd.DataFrame()
        
        # Basic game info
        if 'Date' in features_df.columns:
            results['Date'] = features_df['Date']
        elif 'Date' in games_df.columns:
            results['Date'] = games_df['Date']
        
        # Teams
        if 'Home' in features_df.columns:
            results['Home'] = features_df['Home']
            results['Away'] = features_df['Away']
        elif 'Home' in games_df.columns:
            results['Home'] = games_df['Home'].iloc[:len(predictions)]
            results['Away'] = games_df['Away'].iloc[:len(predictions)]
        
        # Pitchers
        if 'home_pitcher' in games_df.columns:
            results['Home_Pitcher'] = games_df['home_pitcher'].iloc[:len(predictions)]
            results['Away_Pitcher'] = games_df['away_pitcher'].iloc[:len(predictions)]
        
        # Predictions
        results['Predicted_Winner'] = ['Home' if pred == 1 else 'Away' for pred in predictions]
        results['Home_Win_Prob'] = probabilities[:, 1]
        results['Away_Win_Prob'] = probabilities[:, 0]
        results['Confidence'] = np.maximum(probabilities[:, 0], probabilities[:, 1])
        
        # Add odds if available
        if 'home_odds' in games_df.columns:
            results['Home_Odds'] = games_df['home_odds'].iloc[:len(predictions)]
            results['Away_Odds'] = games_df['away_odds'].iloc[:len(predictions)]
            
            # Validate odds and calculate EV only for valid odds
            results['Odds_Valid'] = results.apply(
                lambda row: self.validate_odds(row['Home_Odds'], row['Away_Odds']),
                axis=1
            )
            
            # Calculate expected value only for valid odds
            results['Home_EV'] = results.apply(
                lambda row: self.calculate_ev(row['Home_Win_Prob'], row['Home_Odds']) 
                           if row['Odds_Valid'] else np.nan,
                axis=1
            )
            results['Away_EV'] = results.apply(
                lambda row: self.calculate_ev(row['Away_Win_Prob'], row['Away_Odds'])
                           if row['Odds_Valid'] else np.nan,
                axis=1
            )
            
            # Best bet (only for valid odds with positive EV)
            results['Best_Bet'] = results.apply(
                lambda row: self.determine_best_bet(row),
                axis=1
            )
        
        return results
    
    def clean_schedule_data(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Clean schedule data to fix common issues."""
        df = games_df.copy()
        
        # Fix team names
        team_fixes = {
            'Athletics': 'OAK',
            'Oakland Athletics': 'OAK',
            'Arizona Diamondbacks': 'ARI',
            'Diamondbacks': 'ARI'
        }
        
        if 'Home' in df.columns:
            df['Home'] = df['Home'].replace(team_fixes)
        if 'Away' in df.columns:
            df['Away'] = df['Away'].replace(team_fixes)
        
        # Remove invalid odds
        if 'home_odds' in df.columns and 'away_odds' in df.columns:
            # Mark invalid odds as NaN
            invalid_mask = (
                (df['home_odds'].abs() > 500) | 
                (df['away_odds'].abs() > 500) |
                ((df['home_odds'] < 0) & (df['away_odds'] < 0))
            )
            df.loc[invalid_mask, ['home_odds', 'away_odds']] = [None, None]
        
        # Fix known pitcher-team mismatches
        pitcher_team_map = {
            'Shohei Ohtani': 'LAD',
            'Walker Buehler': 'LAD',
            'Gerrit Cole': 'NYY',
            'Spencer Strider': 'ATL'
        }
        
        if 'home_pitcher' in df.columns and 'away_pitcher' in df.columns:
            for pitcher, correct_team in pitcher_team_map.items():
                # Fix home pitcher
                wrong_home = (df['home_pitcher'] == pitcher) & (df['Home'] != correct_team)
                if wrong_home.any():
                    df.loc[wrong_home, 'home_pitcher'] = 'TBD'
                
                # Fix away pitcher
                wrong_away = (df['away_pitcher'] == pitcher) & (df['Away'] != correct_team)
                if wrong_away.any():
                    df.loc[wrong_away, 'away_pitcher'] = 'TBD'
        
        # Remove exact duplicates
        df = df.drop_duplicates(subset=['Home', 'Away'])
        
        return df
    
    def validate_odds(self, home_odds, away_odds):
        """
        Validate that odds are realistic for MLB games.
        
        Returns:
            bool: True if odds are valid, False otherwise
        """
        # Check for NaN
        if pd.isna(home_odds) or pd.isna(away_odds):
            return False
        
        # MLB odds should never exceed these bounds
        if abs(home_odds) > 500 or abs(away_odds) > 500:
            print(f"   ⚠️ Invalid odds detected: Home {home_odds:+.0f} / Away {away_odds:+.0f} (exceeds ±500)")
            return False
        
        # Both can't be negative (book would lose money)
        if home_odds < 0 and away_odds < 0:
            print(f"   ⚠️ Invalid odds detected: Both negative - Home {home_odds:+.0f} / Away {away_odds:+.0f}")
            return False
        
        # At least one should be positive (underdog)
        if home_odds < -400 and away_odds < 100:
            print(f"   ⚠️ Suspicious odds: Heavy favorite without proper underdog odds")
            return False
            
        return True
    
    def calculate_ev(self, probability, odds):
        """
        Calculate expected value from probability and American odds.
        Fixed calculation with validation.
        """
        if pd.isna(odds) or pd.isna(probability):
            return np.nan
        
        # Validate probability is between 0 and 1
        if probability < 0 or probability > 1:
            return np.nan
        
        # Convert American odds to decimal
        if odds > 0:
            # Underdog: +150 means bet $100 to win $150
            decimal_odds = 1 + (odds / 100)
        else:
            # Favorite: -150 means bet $150 to win $100
            decimal_odds = 1 + (100 / abs(odds))
        
        # Calculate EV as a percentage of stake
        # EV = (probability * profit) - (1 - probability) * loss
        # Where profit = (decimal_odds - 1) and loss = 1
        ev = (probability * (decimal_odds - 1)) - ((1 - probability) * 1)
        
        # Cap EV at reasonable bounds (-100% to +50%)
        # In reality, finding even 10% EV is extremely rare
        ev_capped = max(-1.0, min(0.5, ev))
        
        # Warn if EV seems unrealistic
        if ev > 0.5:
            print(f"   ⚠️ EV capped from {ev:.1%} to {ev_capped:.1%} (unrealistic)")
        
        return ev_capped
    
    def determine_best_bet(self, row):
        """Determine the best bet based on EV with realistic thresholds."""
        # Check if odds are valid first
        if 'Odds_Valid' in row and not row['Odds_Valid']:
            return 'Invalid Odds'
        
        home_ev = row.get('Home_EV', np.nan)
        away_ev = row.get('Away_EV', np.nan)
        
        if pd.isna(home_ev) or pd.isna(away_ev):
            return 'No Bet'
        
        # Only recommend bets with at least 5% EV (realistic threshold)
        # In real betting, even 5% EV is excellent
        MIN_EV_THRESHOLD = 0.05
        
        if home_ev > MIN_EV_THRESHOLD and home_ev > away_ev:
            return f"Home ({home_ev:.1%} EV)"
        elif away_ev > MIN_EV_THRESHOLD and away_ev > home_ev:
            return f"Away ({away_ev:.1%} EV)"
        else:
            return 'No Edge'
    
    def display_predictions(self, results):
        """Display predictions in a formatted way."""
        print("\n" + "="*80)
        print("⚾ MLB GAME PREDICTIONS")
        print("="*80)
        
        if results.empty:
            print("No predictions to display.")
            return
        
        # Check for invalid odds
        if 'Odds_Valid' in results.columns:
            invalid_odds_count = (~results['Odds_Valid']).sum()
            if invalid_odds_count > 0:
                print(f"\n⚠️ WARNING: {invalid_odds_count} games have invalid/suspicious odds!")
                print("   These games will be shown but betting recommendations disabled.\n")
        
        # Sort by confidence or EV
        if 'Home_EV' in results.columns:
            # Sort by max EV, but only for valid odds
            results['Max_EV'] = results.apply(
                lambda row: max(row['Home_EV'], row['Away_EV']) 
                           if row.get('Odds_Valid', True) and pd.notna(row['Home_EV']) 
                           else -1,
                axis=1
            )
            results = results.sort_values('Max_EV', ascending=False)
        else:
            results = results.sort_values('Confidence', ascending=False)
        
        # Display each game
        for idx, row in results.iterrows():
            print(f"\n🏟️  {row['Away']} @ {row['Home']}")
            
            # Pitchers if available
            if 'Home_Pitcher' in row and pd.notna(row['Home_Pitcher']):
                print(f"   ⚾ {row['Away_Pitcher']} vs {row['Home_Pitcher']}")
            
            # Prediction
            winner = row['Predicted_Winner']
            prob = row['Home_Win_Prob'] if winner == 'Home' else row['Away_Win_Prob']
            print(f"   🎯 Prediction: {winner} wins ({prob:.1%} confidence)")
            
            # Probabilities
            print(f"   📊 Win Probability: Home {row['Home_Win_Prob']:.1%} | Away {row['Away_Win_Prob']:.1%}")
            
            # Odds and betting info if available
            if 'Home_Odds' in row and pd.notna(row['Home_Odds']):
                # Check if odds are valid
                if 'Odds_Valid' in row and not row['Odds_Valid']:
                    print(f"   ⚠️ INVALID ODDS: Home {row['Home_Odds']:+.0f} | Away {row['Away_Odds']:+.0f}")
                    print(f"      (Odds exceed realistic bounds or have errors)")
                else:
                    print(f"   💰 Betting Odds: Home {row['Home_Odds']:+.0f} | Away {row['Away_Odds']:+.0f}")
                    
                    if 'Home_EV' in row and pd.notna(row['Home_EV']):
                        print(f"   📈 Expected Value: Home {row['Home_EV']:.1%} | Away {row['Away_EV']:.1%}")
                        
                        if row['Best_Bet'] not in ['No Bet', 'No Edge', 'Invalid Odds']:
                            # Only show recommendation for reasonable EVs
                            ev_value = float(row['Best_Bet'].split('(')[1].split('%')[0]) / 100
                            if ev_value < 0.3:  # Only show if EV is less than 30%
                                print(f"   ⭐ RECOMMENDED: {row['Best_Bet']}")
                            else:
                                print(f"   ⚠️ EV too high to be realistic: {row['Best_Bet']}")
        
        # Summary
        print("\n" + "-"*80)
        print("📊 SUMMARY")
        print("-"*80)
        
        print(f"Total games: {len(results)}")
        print(f"Home favorites: {(results['Predicted_Winner'] == 'Home').sum()}")
        print(f"Away favorites: {(results['Predicted_Winner'] == 'Away').sum()}")
        
        if 'Odds_Valid' in results.columns:
            valid_odds = results[results['Odds_Valid']]
            print(f"Games with valid odds: {len(valid_odds)}/{len(results)}")
        
        if 'Best_Bet' in results.columns:
            # Only count reasonable value bets
            value_bets = results[
                (results['Best_Bet'] != 'No Bet') & 
                (results['Best_Bet'] != 'No Edge') & 
                (results['Best_Bet'] != 'Invalid Odds')
            ]
            
            # Filter to reasonable EVs only
            reasonable_bets = []
            for idx, row in value_bets.iterrows():
                try:
                    ev_str = row['Best_Bet'].split('(')[1].split('%')[0]
                    ev_value = float(ev_str) / 100
                    if ev_value <= 0.3:  # 30% EV max
                        reasonable_bets.append(row)
                except:
                    continue
            
            if reasonable_bets:
                print(f"\n💎 VALUE BETS: {len(reasonable_bets)} games with positive EV (5%+ edge)")
                for row in reasonable_bets[:10]:  # Show max 10
                    print(f"   • {row['Away']} @ {row['Home']}: {row['Best_Bet']}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Predict MLB games')
    parser.add_argument('--date', type=str, default=None,
                       help='Date to predict (YYYY-MM-DD format)')
    parser.add_argument('--save', action='store_true',
                       help='Save predictions to CSV file')
    parser.add_argument('--tomorrow', action='store_true',
                       help='Predict tomorrow\'s games')
    args = parser.parse_args()
    
    # Determine date
    if args.tomorrow:
        game_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        game_date = args.date
    
    print("\n" + "⚾"*40)
    print("   MLB GAME PREDICTION SYSTEM")
    print("⚾"*40)
    
    try:
        # Initialize predictor
        predictor = MLBGamePredictor()
        
        # Make predictions
        results = predictor.predict_games(game_date=game_date)
        
        if not results.empty:
            # Display predictions
            predictor.display_predictions(results)
            
            # Save if requested
            if args.save:
                output_dir = Path('predictions')
                output_dir.mkdir(exist_ok=True)
                
                date_str = game_date or datetime.now().strftime('%Y-%m-%d')
                filename = f"predictions_{date_str}.csv"
                output_path = output_dir / filename
                
                results.to_csv(output_path, index=False)
                print(f"\n💾 Predictions saved to: {output_path}")
                
                # Also save a summary
                summary_path = output_dir / f"summary_{date_str}.txt"
                with open(summary_path, 'w') as f:
                    f.write(f"MLB Predictions for {date_str}\n")
                    f.write("="*50 + "\n\n")
                    
                    for idx, row in results.iterrows():
                        f.write(f"{row['Away']} @ {row['Home']}\n")
                        f.write(f"  Prediction: {row['Predicted_Winner']} ")
                        f.write(f"({row['Home_Win_Prob' if row['Predicted_Winner'] == 'Home' else 'Away_Win_Prob']:.1%})\n")
                        
                        if 'Best_Bet' in row and row['Best_Bet'] not in ['No Bet', 'No Edge']:
                            f.write(f"  Betting: {row['Best_Bet']}\n")
                        f.write("\n")
                
                print(f"📝 Summary saved to: {summary_path}")
        else:
            print("\n❌ No games found for the specified date")
            print(f"   Date checked: {game_date or 'today'}")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Check if model exists: ls models/game_winner/")
        print("   2. Retrain if needed: python scripts/fix_models_and_predict.py --retrain")
        print("   3. Check date format: YYYY-MM-DD")
        
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
