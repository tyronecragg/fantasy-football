import csv
from bs4 import BeautifulSoup
import re


def extract_goalkeeper_odds(html_file):
    """
    Extract goalkeeper saves odds from HTML file and save to CSV
    """
    # Read the HTML file
    with open(html_file, 'r', encoding='utf-8') as file:
        html_content = file.read()

    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all match sections
    match_sections = soup.find_all('div', class_='gl-MarketGroupPod')

    data = []

    for section in match_sections:
        # Get match name and date
        match_button = section.find('div', class_='src-FixtureSubGroupButton_Text')
        date_element = section.find('div', class_='src-FixtureSubGroupButton_BookCloses')

        if not match_button:
            continue

        match_name = match_button.get_text(strip=True)
        match_date = date_element.get_text(strip=True) if date_element else 'N/A'

        # Find all goalkeeper sections within this match
        goalkeeper_sections = section.find_all('div', class_='srb-HScrollParticipantMarket')

        for gk_section in goalkeeper_sections:
            # Get team name
            team_header = gk_section.find('div', class_='srb-HScrollParticipantHeader')
            if not team_header:
                continue
            team_name = team_header.get_text(strip=True)

            # Get goalkeeper name
            gk_name_element = gk_section.find('div', class_='srb-ParticipantLabelWithTeam_Name')
            if not gk_name_element:
                continue
            goalkeeper_name = gk_name_element.get_text(strip=True)

            # Get odds for different save thresholds
            odds_section = gk_section.find_next_sibling('div', class_='srb-HScrollOddsMarket')
            if not odds_section:
                continue

            # Extract save thresholds and odds
            save_columns = odds_section.find_all('div', class_='srb-HScrollPlaceColumnMarket')

            row_data = {
                'Match': match_name,
                'Date': match_date,
                'Team': team_name,
                'Goalkeeper': goalkeeper_name
            }

            for column in save_columns:
                # Get save threshold (e.g., "1.5", "2.5", etc.)
                threshold_element = column.find('div', class_='srb-HScrollPlaceHeader')
                odds_element = column.find('span', class_='gl-ParticipantOddsOnly_Odds')

                if threshold_element and odds_element:
                    threshold = threshold_element.get_text(strip=True)
                    odds = odds_element.get_text(strip=True)
                    row_data[f'{threshold} Saves'] = odds

            data.append(row_data)

    return data


def save_to_csv(data, filename):
    """
    Save extracted data to CSV file
    """
    if not data:
        print("No data to save")
        return

    # Get all unique column headers
    all_headers = set()
    for row in data:
        all_headers.update(row.keys())

    # Sort headers to have consistent order
    base_headers = ['Match', 'Date', 'Team', 'Goalkeeper']
    save_headers = sorted([h for h in all_headers if 'Saves' in h],
                          key=lambda x: float(x.split()[0]))
    headers = base_headers + save_headers

    # Write to CSV
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)

    print(f"Data saved to {filename}")
    print(f"Extracted {len(data)} goalkeeper records")


def main():
    """
    Main function to extract odds and save to CSV
    """
    html_file = 'goalkeeper_saves.html'
    csv_file = 'goalkeeper_saves.csv'

    try:
        # Extract data from HTML
        data = extract_goalkeeper_odds(html_file)

        # Save to CSV
        save_to_csv(data, csv_file)

        # Print sample of extracted data
        if data:
            print("\nSample of extracted data:")
            for i, row in enumerate(data[:3]):  # Show first 3 rows
                print(f"\nRow {i + 1}:")
                for key, value in row.items():
                    print(f"  {key}: {value}")

    except FileNotFoundError:
        print(f"Error: Could not find {html_file}")
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
