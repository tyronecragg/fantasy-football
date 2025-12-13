import csv
from bs4 import BeautifulSoup
import re


def extract_yellow_card_odds(html_file):
    """
    Extract player yellow card booking odds from HTML file and save to CSV
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
        # Skip closed sections (they don't have odds displayed)
        if 'src-FixtureSubGroupWithShowMore_Closed' in section.get('class', []):
            continue

        # Get match name and date
        match_button = section.find('div', class_='src-FixtureSubGroupButton_Text')
        date_element = section.find('div', class_='src-FixtureSubGroupButton_BookCloses')

        if not match_button:
            continue

        match_name = match_button.get_text(strip=True)
        match_date = date_element.get_text(strip=True) if date_element else 'N/A'

        # Find all player booking odds within this match
        player_elements = section.find_all('div', class_='gl-ParticipantBorderless')

        for player_element in player_elements:
            # Get player name
            name_span = player_element.find('span', class_='gl-ParticipantBorderless_Name')
            odds_span = player_element.find('span', class_='gl-ParticipantBorderless_Odds')

            if name_span and odds_span:
                player_name = name_span.get_text(strip=True)
                booking_odds = odds_span.get_text(strip=True)

                # Clean up player name (remove extra spaces)
                player_name = ' '.join(player_name.split())

                # Create row data
                row_data = {
                    'Match': match_name,
                    'Date': match_date,
                    'Player_Name': player_name,
                    'Booking_Odds': booking_odds
                }

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
    headers = ['Match', 'Date', 'Player_Name', 'Booking_Odds']

    # Write to CSV
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)

    print(f"Data saved to {filename}")
    print(f"Extracted {len(data)} player booking odds")


def main():
    """
    Main function to extract yellow card booking odds and save to CSV
    """
    html_file = 'yellow_card.html'
    csv_file = 'yellow_card.csv'

    try:
        data = extract_yellow_card_odds(html_file)
        save_to_csv(data, csv_file)

        print("\n" + "=" * 60)
        print("FILES CREATED:")
        print("=" * 60)
        print(f"• {csv_file} - Complete player booking odds")
        print(f"• {csv_file.replace('.csv', '_match_summary.csv')} - Match-level summary")
        print("\nThe data shows odds for players to receive a yellow card.")
        print("Lower odds = higher probability of booking")

    except FileNotFoundError:
        print(f"Error: Could not find {html_file}")
        print("Please save the HTML content to 'yellow_card.html'")
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
