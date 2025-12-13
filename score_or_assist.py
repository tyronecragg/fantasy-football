#!/usr/bin/env python3
"""
HTML Betting Odds Parser
Extracts player betting data from score_or_assist.html file and saves to CSV.
Handles multiple market groups and various betting markets.
"""

import re
import csv
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BettingOddsParser:
    def __init__(self, html_file_path: str):
        self.html_file_path = html_file_path
        self.soup = None
        self.all_matches_data = []

    def load_html(self) -> None:
        """Load and parse the HTML file."""
        try:
            with open(self.html_file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                self.soup = BeautifulSoup(content, 'html.parser')
                logger.info(f"Successfully loaded HTML file: {self.html_file_path}")
        except FileNotFoundError:
            logger.error(f"File not found: {self.html_file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading HTML file: {e}")
            raise

    def extract_player_info(self, player_element) -> Dict[str, str]:
        """Extract player name and team from player element."""
        player_info = {
            'player_name': 'Unknown Player',
            'team': 'Unknown Team'
        }

        # Extract player name
        name_element = player_element.find('div', class_='srb-ParticipantLabelWithTeam_Name')
        if name_element:
            player_info['player_name'] = name_element.get_text(strip=True)

        # Determine team from jersey image source
        img_element = player_element.find('img', class_='tk-TeamKitBackImage_SVG')
        if img_element and img_element.get('src'):
            src = img_element.get('src').lower()
            
            if 'liverpool' in src:
                player_info['team'] = 'Liverpool'
            elif 'bournemouth' in src:
                player_info['team'] = 'Bournemouth'
            elif 'villa' in src:
                player_info['team'] = 'Aston Villa'
            elif 'westham' in src:
                player_info['team'] = 'West Ham'
            elif 'man city' in src:
                player_info['team'] = 'Manchester City'
            elif 'manchester_united' in src:
                player_info['team'] = 'Manchester United'
            elif 'palace' in src:
                player_info['team'] = 'Crystal Palace'
            elif 'nottingham' in src:
                player_info['team'] = 'Nottingham Forest'
            elif 'everton' in src:
                player_info['team'] = 'Everton'
            elif 'leeds' in src:
                player_info['team'] = 'Leeds United'
            else:
                # Try to extract team name from the filename
                team_match = re.search(r'/([^/_]+)_', src)
                if team_match:
                    player_info['team'] = team_match.group(1).title()

        return player_info

    def extract_market_odds(self, market_container, market_name: str) -> List[float]:
        """Extract odds from a specific market container."""
        odds = []
        odds_elements = market_container.find_all('span', class_='gl-ParticipantOddsOnly_Odds')

        for odds_element in odds_elements:
            try:
                odds_value = float(odds_element.get_text(strip=True))
                odds.append(odds_value)
            except ValueError:
                # Handle non-numeric odds (e.g., "Evens", "N/A")
                odds_text = odds_element.get_text(strip=True)
                if odds_text.lower() == 'evens':
                    odds.append(2.0)
                else:
                    odds.append(None)  # Will be handled later

        return odds

    def extract_market_headers(self, market_group) -> List[str]:
        """Extract market column headers."""
        headers = []
        header_elements = market_group.find_all('div', class_='gl-MarketColumnHeader')

        for header in header_elements:
            header_text = header.get_text(strip=True)
            if header_text:
                headers.append(header_text)

        return headers

    def parse_market_group(self, market_group) -> List[Dict]:
        """Parse a single market group and return player data."""
        group_data = []

        # Find the market container
        market_container = market_group.find('div', class_='gl-MarketGroupContainer')
        if not market_container:
            logger.warning("No market container found in this group")
            return group_data

        # Extract market headers
        market_headers = self.extract_market_headers(market_group)
        logger.info(f"Found markets: {market_headers}")

        # Find all player elements
        player_elements = market_container.find_all('div', class_='srb-ParticipantLabelWithTeam')

        # Find all market columns
        market_columns = market_container.find_all('div', class_='gl-Market')

        # Filter out header columns and get actual odds columns
        odds_columns = [col for col in market_columns if col.find('div', class_='gl-MarketColumnHeader')]

        logger.info(f"Found {len(player_elements)} players and {len(odds_columns)} market columns")

        # Process each player
        for i, player_element in enumerate(player_elements):
            player_info = self.extract_player_info(player_element)

            # Create base player record
            player_record = {
                'player_name': player_info['player_name'],
                'team': player_info['team']
            }

            # Extract odds for each market
            for j, market_column in enumerate(odds_columns):
                if j < len(market_headers):
                    market_name = market_headers[j]
                    odds_list = self.extract_market_odds(market_column, market_name)

                    # Add odds for this player (if available)
                    if i < len(odds_list):
                        player_record[f'{market_name}_odds'] = odds_list[i]
                    else:
                        player_record[f'{market_name}_odds'] = None

            group_data.append(player_record)

        return group_data

    def parse_all_market_groups(self) -> None:
        """Parse all market groups in the HTML file."""
        if not self.soup:
            raise ValueError("HTML not loaded. Call load_html() first.")

        # Find all market group containers
        market_groups = self.soup.find_all('div', class_='gl-MarketGroupPod src-FixtureSubGroupWithShowMore')

        logger.info(f"Found {len(market_groups)} market groups to process")

        for i, market_group in enumerate(market_groups):
            logger.info(f"Processing market group {i + 1}/{len(market_groups)}")
            group_data = self.parse_market_group(market_group)
            self.all_matches_data.extend(group_data)

        logger.info(f"Total players processed: {len(self.all_matches_data)}")

    def save_to_csv(self, output_file: str = 'betting_odds.csv') -> None:
        """Save extracted data to CSV file."""
        if not self.all_matches_data:
            logger.warning("No data to save")
            return

        # Get all unique column names
        all_columns = set()
        for record in self.all_matches_data:
            all_columns.update(record.keys())

        # Sort columns for consistent output
        base_columns = ['player_name', 'team']
        odds_columns = sorted([col for col in all_columns if col.endswith('_odds')])
        other_columns = sorted([col for col in all_columns if col not in base_columns and not col.endswith('_odds')])

        column_order = base_columns + odds_columns + other_columns

        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=column_order)
                writer.writeheader()

                for record in self.all_matches_data:
                    writer.writerow(record)

            logger.info(f"Data successfully saved to {output_file}")
            logger.info(f"Total records: {len(self.all_matches_data)}")
            logger.info(f"Columns: {len(column_order)}")

        except Exception as e:
            logger.error(f"Error saving to CSV: {e}")
            raise

    def get_summary_stats(self) -> Dict:
        """Get summary statistics of the parsed data."""
        if not self.all_matches_data:
            return {}

        stats = {
            'total_players': len(self.all_matches_data),
            'teams': list(set(record['team'] for record in self.all_matches_data)),
            'markets': [col.replace('_odds', '') for col in self.all_matches_data[0].keys() if col.endswith('_odds')]
        }

        return stats


def main():
    """Main function to run the parser."""
    input_file = 'score_or_assist.html'
    output_file = 'score_or_assist.csv'

    try:
        # Initialize parser
        parser = BettingOddsParser(input_file)

        # Load and parse HTML
        parser.load_html()
        parser.parse_all_market_groups()

        # Save to CSV
        parser.save_to_csv(output_file)

        # Display summary statistics
        stats = parser.get_summary_stats()
        print("\n=== PARSING SUMMARY ===")
        print(f"Total players processed: {stats.get('total_players', 0)}")
        print(f"Teams found: {', '.join(stats.get('teams', []))}")
        print(f"Markets found: {', '.join(stats.get('markets', []))}")
        print(f"Output saved to: {output_file}")

    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
