import csv
from bs4 import BeautifulSoup
import re


def extract_team_goals_odds(html_file):
    """
    Extract team total goals odds from HTML file and save to CSV
    Focus on over/under 1.5 and 3.5 goals for each team
    """
    # Read the HTML file
    with open(html_file, 'r', encoding='utf-8') as file:
        html_content = file.read()

    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all match sections that are open (have odds displayed)
    match_sections = soup.find_all('div', class_='gl-MarketGroupPod')

    data = []

    for section in match_sections:
        # Skip closed sections (they don't have odds displayed)
        if 'src-FixtureSubGroup_Closed' in section.get('class', []):
            continue

        # Get match name and date
        match_button = section.find('div', class_='src-FixtureSubGroupButton_Text')
        date_element = section.find('div', class_='src-FixtureSubGroupButton_BookCloses')

        if not match_button:
            continue

        match_name = match_button.get_text(strip=True)
        match_date = date_element.get_text(strip=True) if date_element else 'N/A'

        # Extract team names from match name (e.g., "Liverpool v Bournemouth")
        teams = match_name.split(' v ')
        if len(teams) != 2:
            continue

        for i in range(0, 2):
            if teams[i] == 'Tottenham':
                teams[i] = 'Spurs'
            elif teams[i] == 'Wolverhampton':
                teams[i] = 'Wolves'
            elif teams[i] == 'Nottm Forest':
                teams[i] = "Nott'm Forest"
            i+=1

        home_team, away_team = teams[0].strip(), teams[1].strip()

        # Find both team columns
        team_columns = section.find_all('div', class_='gl-Market_General-columnheader')

        for i, column in enumerate(team_columns):
            team_header = column.find('div', class_='gl-MarketColumnHeader')
            if not team_header:
                continue

            team_name = team_header.get_text(strip=True)

            if team_name == 'Tottenham':
                team_name = 'Spurs'
            elif team_name == 'Wolverhampton':
                team_name = 'Wolves'
            elif team_name == 'Nottm Forest':
                team_name = "Nott'm Forest"

            # Determine opponent
            if team_name == home_team:
                opponent = away_team
            elif team_name == away_team:
                opponent = home_team
            else:
                continue

            # Extract odds for this team
            odds_elements = column.find_all('div', class_='srb-ParticipantCenteredStackedWithMarketBorders')

            # Initialize row data
            row_data = {
                'Match': match_name,
                'Date': match_date,
                'Team': team_name,
                'Opponent': opponent
            }

            # Extract specific odds we're interested in
            for odds_element in odds_elements:
                handicap_span = odds_element.find('span',
                                                  class_='srb-ParticipantCenteredStackedWithMarketBorders_Handicap')
                odds_span = odds_element.find('span', class_='srb-ParticipantCenteredStackedWithMarketBorders_Odds')

                if handicap_span and odds_span:
                    handicap = handicap_span.get_text(strip=True)
                    odds = odds_span.get_text(strip=True)

                    # Map the odds to our desired columns
                    if handicap == 'Over 1.5':
                        row_data['Team_Over_1.5'] = odds
                        row_data['Opponent_Concedes_Over_1.5'] = odds
                    elif handicap == 'Under 1.5':
                        row_data['Team_Under_1.5'] = odds
                        row_data['Opponent_Concedes_Under_1.5'] = odds
                    elif handicap == 'Over 3.5':
                        row_data['Team_Over_3.5'] = odds
                        row_data['Opponent_Concedes_Over_3.5'] = odds
                    elif handicap == 'Under 3.5':
                        row_data['Team_Under_3.5'] = odds
                        row_data['Opponent_Concedes_Under_3.5'] = odds

            # Only add row if we have the key odds
            if any(key in row_data for key in ['Team_Over_1.5', 'Team_Over_3.5']):
                data.append(row_data)

    return data


def save_to_csv(data, filename):
    """
    Save extracted data to CSV file
    """
    if not data:
        print("No data to save")
        return

    # Define the column order
    headers = [
        'Match', 'Date', 'Team', 'Opponent',
        'Team_Over_1.5', 'Team_Under_1.5',
        'Team_Over_3.5', 'Team_Under_3.5',
        'Opponent_Concedes_Over_1.5', 'Opponent_Concedes_Under_1.5',
        'Opponent_Concedes_Over_3.5', 'Opponent_Concedes_Under_3.5'
    ]

    # Write to CSV
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)

    print(f"Data saved to {filename}")
    print(f"Extracted {len(data)} team records")


def main():
    """
    Main function to extract team goals odds and save to CSV
    """
    html_file = 'team_total_goals.html'
    csv_file = 'team_total_goals.csv'

    try:
        data = extract_team_goals_odds(html_file)
        save_to_csv(data, csv_file)
    except FileNotFoundError:
        print(f"Error: Could not find {html_file}")
        print("Please save the HTML content to 'team_total_goals.html'")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Install required package if not already installed
    try:
        import bs4
    except ImportError:
        print("Installing required package...")
        import subprocess
        import sys

        subprocess.check_call([sys.executable, "-m", "pip", "install", "beautifulsoup4"])
        from bs4 import BeautifulSoup

    main()
