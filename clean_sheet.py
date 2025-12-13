#!/usr/bin/env python3
"""
Clean Sheet Odds Parser
Extracts clean sheet betting data from clean_sheet.html file and saves to CSV in long format.
Output format: team_name, yes, no
"""

import csv
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CleanSheetParser:
    def __init__(self, html_file_path: str):
        self.html_file_path = html_file_path
        self.soup = None
        self.clean_sheet_data = []

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

    def extract_match_info(self, market_group) -> Dict[str, str]:
        """Extract match information from market group."""
        match_info = {
            'match_name': 'Unknown Match',
            'match_time': 'Unknown Time',
            'team1': 'Unknown Team1',
            'team2': 'Unknown Team2'
        }

        # Extract match name and time
        match_header = market_group.find('div', class_='src-FixtureSubGroupButton_Text')
        if match_header:
            match_text = match_header.get_text(strip=True)
            match_info['match_name'] = match_text

            # Parse team names from match text (e.g., "Liverpool v Bournemouth")
            if ' v ' in match_text:
                teams = match_text.split(' v ')
                if len(teams) == 2:
                    match_info['team1'] = teams[0].strip()
                    match_info['team2'] = teams[1].strip()

        time_element = market_group.find('div', class_='src-FixtureSubGroupButton_BookCloses')
        if time_element:
            match_info['match_time'] = time_element.get_text(strip=True)

        return match_info

    def extract_team_odds(self, market_container, team_name: str) -> Dict[str, Optional[float]]:
        """Extract yes/no odds for a specific team."""
        odds_data = {
            'yes': None,
            'no': None
        }

        # Find all odds elements for this team's market
        odds_elements = market_container.find_all('span', class_='srb-ParticipantCenteredStackedWithMarketBorders_Odds')
        handicap_elements = market_container.find_all('span',
                                                      class_='srb-ParticipantCenteredStackedWithMarketBorders_Handicap')

        # Match handicaps with odds
        for i, handicap_element in enumerate(handicap_elements):
            handicap_text = handicap_element.get_text(strip=True).lower()

            if i < len(odds_elements):
                try:
                    odds_value = float(odds_elements[i].get_text(strip=True))

                    if handicap_text == 'yes':
                        odds_data['yes'] = odds_value
                    elif handicap_text == 'no':
                        odds_data['no'] = odds_value

                except ValueError:
                    logger.warning(f"Could not parse odds value: {odds_elements[i].get_text(strip=True)}")
                    continue

        return odds_data

    def parse_market_group(self, market_group) -> List[Dict]:
        """Parse a single market group and return clean sheet data."""
        group_data = []

        # Skip closed market groups (they don't have odds displayed)
        if 'src-FixtureSubGroup_Closed' in market_group.get('class', []):
            logger.debug("Skipping closed market group")
            return group_data

        # Extract match information
        match_info = self.extract_match_info(market_group)
        logger.info(f"Processing match: {match_info['match_name']} at {match_info['match_time']}")

        # Find the market container
        market_container = market_group.find('div', class_='gl-MarketGroupContainer')
        if not market_container:
            logger.warning("No market container found in this group")
            return group_data

        # Find all market columns (one for each team)
        market_columns = market_container.find_all('div', class_='gl-Market')

        for market_column in market_columns:
            # Get team name from market header
            team_header = market_column.find('div', class_='gl-MarketColumnHeader')
            if not team_header:
                continue

            team_name = team_header.get_text(strip=True)

            # Skip if this is not a team column (might be other market types)
            if team_name.lower() in ['matches', 'all', '']:
                continue

            if team_name == 'Tottenham':
                team_name = 'Spurs'
            elif team_name == 'Wolverhampton':
                team_name = 'Wolves'
            elif team_name == 'Nottm Forest':
                team_name = "Nott'm Forest"

            logger.debug(f"Processing team: {team_name}")

            # Extract odds for this team
            team_odds = self.extract_team_odds(market_column, team_name)

            # Create record for this team
            team_record = {
                'match_name': match_info['match_name'],
                'match_time': match_info['match_time'],
                'team_name': team_name,
                'yes': team_odds['yes'],
                'no': team_odds['no']
            }

            group_data.append(team_record)
            logger.debug(f"Added record for {team_name}: Yes={team_odds['yes']}, No={team_odds['no']}")

        return group_data

    def parse_all_market_groups(self) -> None:
        """Parse all market groups in the HTML file."""
        if not self.soup:
            raise ValueError("HTML not loaded. Call load_html() first.")

        # Find all market group containers
        market_groups = self.soup.find_all('div', class_='gl-MarketGroupPod src-FixtureSubGroup')

        logger.info(f"Found {len(market_groups)} market groups to process")

        for i, market_group in enumerate(market_groups):
            logger.info(f"Processing market group {i + 1}/{len(market_groups)}")
            group_data = self.parse_market_group(market_group)
            self.clean_sheet_data.extend(group_data)

        logger.info(f"Total team records processed: {len(self.clean_sheet_data)}")

    def save_to_csv(self, output_file: str = 'clean_sheet.csv') -> None:
        """Save extracted data to CSV file in long format."""
        if not self.clean_sheet_data:
            logger.warning("No data to save")
            return

        # Define column order for long format
        columns = ['match_name', 'match_time', 'team_name', 'yes', 'no']

        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=columns)
                writer.writeheader()

                for record in self.clean_sheet_data:
                    # Ensure all required columns are present
                    csv_record = {col: record.get(col, '') for col in columns}
                    writer.writerow(csv_record)

            logger.info(f"Clean sheet data successfully saved to {output_file}")
            logger.info(f"Total records: {len(self.clean_sheet_data)}")

        except Exception as e:
            logger.error(f"Error saving to CSV: {e}")
            raise

    def get_summary_stats(self) -> Dict:
        """Get summary statistics of the parsed data."""
        if not self.clean_sheet_data:
            return {}

        # Get unique matches and teams
        unique_matches = list(set(record['match_name'] for record in self.clean_sheet_data))
        unique_teams = list(set(record['team_name'] for record in self.clean_sheet_data))

        # Count records with valid odds
        valid_yes_odds = sum(1 for record in self.clean_sheet_data if record['yes'] is not None)
        valid_no_odds = sum(1 for record in self.clean_sheet_data if record['no'] is not None)

        stats = {
            'total_records': len(self.clean_sheet_data),
            'unique_matches': len(unique_matches),
            'unique_teams': len(unique_teams),
            'matches': unique_matches,
            'teams': unique_teams,
            'valid_yes_odds': valid_yes_odds,
            'valid_no_odds': valid_no_odds
        }

        return stats

    def validate_data(self) -> List[str]:
        """Validate the extracted data and return any issues found."""
        issues = []

        if not self.clean_sheet_data:
            issues.append("No data was extracted")
            return issues

        for i, record in enumerate(self.clean_sheet_data):
            record_id = f"Record {i + 1} ({record.get('team_name', 'Unknown')})"

            # Check for missing team name
            if not record.get('team_name') or record['team_name'] == 'Unknown Team':
                issues.append(f"{record_id}: Missing team name")

            # Check for missing odds
            if record['yes'] is None:
                issues.append(f"{record_id}: Missing 'Yes' odds")

            if record['no'] is None:
                issues.append(f"{record_id}: Missing 'No' odds")

            # Check for unrealistic odds values
            if record['yes'] is not None and (record['yes'] < 1.01 or record['yes'] > 1000):
                issues.append(f"{record_id}: Unusual 'Yes' odds value: {record['yes']}")

            if record['no'] is not None and (record['no'] < 1.01 or record['no'] > 1000):
                issues.append(f"{record_id}: Unusual 'No' odds value: {record['no']}")

        return issues


def main():
    """Main function to run the parser."""
    input_file = 'clean_sheet.html'
    output_file = 'clean_sheet.csv'

    try:
        # Initialize parser
        parser = CleanSheetParser(input_file)

        # Load and parse HTML
        parser.load_html()
        parser.parse_all_market_groups()

        # Validate data
        issues = parser.validate_data()
        if issues:
            logger.warning("Data validation issues found:")
            for issue in issues[:10]:  # Show first 10 issues
                logger.warning(f"  - {issue}")
            if len(issues) > 10:
                logger.warning(f"  ... and {len(issues) - 10} more issues")

        # Save to CSV
        parser.save_to_csv(output_file)

        # Display summary statistics
        stats = parser.get_summary_stats()
        print("\n=== CLEAN SHEET PARSING SUMMARY ===")
        print(f"Total records processed: {stats.get('total_records', 0)}")
        print(f"Number of matches: {stats.get('unique_matches', 0)}")
        print(f"Number of teams: {stats.get('unique_teams', 0)}")
        print(f"Records with valid 'Yes' odds: {stats.get('valid_yes_odds', 0)}")
        print(f"Records with valid 'No' odds: {stats.get('valid_no_odds', 0)}")

        if stats.get('matches'):
            print(f"\nMatches processed:")
            for match in sorted(stats['matches'])[:5]:  # Show first 5 matches
                print(f"  - {match}")
            if len(stats['matches']) > 5:
                print(f"  ... and {len(stats['matches']) - 5} more matches")

    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
