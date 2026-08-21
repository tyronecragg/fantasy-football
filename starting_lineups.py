import re
import requests
import pandas as pd
from bs4 import BeautifulSoup

from fpl_pipeline.names import player_map

NAME_MAPPINGS = player_map()
STAGING_CSV = "inputs/ffs_predicted_lineups.csv"
CURATED_CSV = "inputs/starting_lineups.csv"
NEWS_MD = "inputs/ffs_team_news.md"
PROB_COLUMNS = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]


def convert_special_characters(text):
    """
    Convert special characters to their English alphabet equivalents
    """
    char_map = {
        'Đ': 'Dj', 'đ': 'dj',
        'á': 'a', 'à': 'a', 'ä': 'a', 'â': 'a', 'ā': 'a', 'ã': 'a', 'å': 'a',
        'Á': 'A', 'À': 'A', 'Ä': 'A', 'Â': 'A', 'Ā': 'A', 'Ã': 'A', 'Å': 'A',
        'é': 'e', 'è': 'e', 'ë': 'e', 'ê': 'e', 'ē': 'e',
        'É': 'E', 'È': 'E', 'Ë': 'E', 'Ê': 'E', 'Ē': 'E',
        'í': 'i', 'ì': 'i', 'ï': 'i', 'î': 'i', 'ī': 'i',
        'Í': 'I', 'Ì': 'I', 'Ï': 'I', 'Î': 'I', 'Ī': 'I',
        'ó': 'o', 'ò': 'o', 'ö': 'o', 'ô': 'o', 'ō': 'o', 'õ': 'o', 'ø': 'o',
        'Ó': 'O', 'Ò': 'O', 'Ö': 'O', 'Ô': 'O', 'Ō': 'O', 'Õ': 'O', 'Ø': 'O',
        'ú': 'u', 'ù': 'u', 'ü': 'u', 'û': 'u', 'ū': 'u',
        'Ú': 'U', 'Ù': 'U', 'Ü': 'U', 'Û': 'U', 'Ū': 'U',
        'ñ': 'n', 'Ñ': 'N',
        'ç': 'c', 'Ç': 'C',
        'ý': 'y', 'ÿ': 'y', 'Ý': 'Y', 'Ÿ': 'Y',
        'ž': 'z', 'Ž': 'Z',
        'š': 's', 'Š': 'S',
        'č': 'c', 'Č': 'C',
        'ř': 'r', 'Ř': 'R',
        'ď': 'd', 'Ď': 'D',
        'ť': 't', 'Ť': 'T',
        'ň': 'n', 'Ň': 'N',
        'ľ': 'l', 'Ľ': 'L',
        'ĺ': 'l', 'Ĺ': 'L',
        'ŕ': 'r', 'Ŕ': 'R',
        # 'đ': 'd', 'Đ': 'D',
        'ș': 's', 'Ș': 'S',
        'ț': 't', 'Ț': 'T',
        'ă': 'a', 'Ă': 'A',
        'ğ': 'g', 'Ğ': 'G',
        'ı': 'i', 'İ': 'I',
        'ş': 's', 'Ş': 'S',
        'ć': 'c', 'Ć': 'C',
        'ł': 'l', 'Ł': 'L',
        'ń': 'n', 'Ń': 'N',
        'ś': 's', 'Ś': 'S',
        'ź': 'z', 'Ź': 'Z',
        'ż': 'z', 'Ż': 'Z',
        'ą': 'a', 'Ą': 'A',
        'ę': 'e', 'Ę': 'E',
        'ő': 'o', 'Ő': 'O',
        'ű': 'u', 'Ű': 'U',
        'æ': 'ae', 'Æ': 'AE',
        'œ': 'oe', 'Œ': 'OE',
        'ß': 'ss',
        'Ã¡': 'a', 'Ã©': 'e', 'Ã­': 'i', 'Ã³': 'o', 'Ãº': 'u',
        'Ã±': 'n', 'Ã«': 'e', 'Ã¶': 'o', 'Ã¼': 'u',
        "’": "'",
    }

    # Replace special characters
    result = text
    for special_char, replacement in char_map.items():
        result = result.replace(special_char, replacement)

    return result


def normalise_player_name(name):

    name = convert_special_characters(name)

    # All renames live in inputs/name_mappings.csv (see fpl_pipeline.names)
    # Return mapped name if exists, otherwise return original
    return NAME_MAPPINGS.get(name, name)


def extract_full_name(title_text):
    if not title_text:
        return ""

    # Check if there's a pattern like "Last Name (First Name)"
    bracket_match = re.search(r'^(.+?)\s*\((.+?)\)$', title_text)
    if bracket_match:
        last_part = bracket_match.group(1).strip()
        first_name = bracket_match.group(2).strip()
        name = f"{first_name} {last_part}"
        name_normalised = normalise_player_name(name)
    else:
        name = title_text.strip()
        name_normalised = normalise_player_name(name)
    return name_normalised


def extract_team_news(team_item):
    """Pull the write-up parts from one FFS team block: next match, Out / Doubts /
    Banned lists (doubts carry FFS's own percentage), the Latest News paragraph
    (empty pre-season, populated in-season) and the last-updated stamp."""
    news = {}

    next_match = team_item.find('div', class_='next-match')
    if next_match:
        news['next_match'] = next_match.get_text(" ", strip=True).replace("Next Match:", "").strip()

    parts = team_item.find('ul', class_='story-parts')
    if not parts:
        return news

    for li in parts.find_all('li', recursive=False):
        if 'grey' in li.get('class', []):
            news['updated'] = li.get_text(" ", strip=True)
            continue
        strong = li.find('strong')
        label = strong.get_text(strip=True).rstrip(':') if strong else None
        if label in ('Out', 'Doubts', 'Banned'):
            players = []
            for p in li.find_all('li'):
                pct = p.find('span', class_='doubt-percent')
                name = p.get_text(" ", strip=True)
                if pct:
                    pct_text = pct.get_text(strip=True)
                    name = f"{name.replace(pct_text, '').strip()} ({pct_text})"
                players.append(name)
            news[label.lower()] = players
        elif label == 'Latest News':
            text = li.get_text(" ", strip=True)
            text = re.sub(r'^Latest News:\s*', '', text).strip()
            if text:
                news['latest'] = text
    return news


def get_team_lineups():
    """
    Scrape FFS team news page for predicted lineups and per-team write-ups
    """
    url = "https://www.fantasyfootballscout.co.uk/team-news"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        teams_data = []
        news_data = {}

        # Find all team news items
        team_items = soup.find_all('li', class_='team-news-item')

        for team_item in team_items:
            # Get team name
            team_header = team_item.find('header')
            if not team_header:
                continue

            team_name_elem = team_header.find('h2')
            if not team_name_elem:
                continue

            team_name = team_name_elem.get_text(strip=True)
            news_data[team_name] = extract_team_news(team_item)

            # Find the formation/lineup section
            formation_div = team_item.find('div', class_='scout-picks-pitch')
            if not formation_div:
                continue

            # Get all player elements from all rows (excluding bench)
            player_elements = []

            # Look for rows 1-5 (starting XI, excluding bench row-5 in some formations)
            for row_num in range(1, 6):
                row = formation_div.find('ul', class_=f'row-{row_num}')
                if row:
                    players_in_row = row.find_all('li')
                    for player in players_in_row:
                        # Skip reserve players (bench players)
                        if 'reserve' not in player.get('class', []):
                            player_elements.append(player)

            # Extract player names from starting XI
            for player_elem in player_elements:
                title = player_elem.get('title', '')
                if title:
                    full_name = extract_full_name(title)
                    if full_name:
                        teams_data.append({
                            'Player': full_name,
                            'Team': team_name,
                        })

        return teams_data, news_data

    except requests.RequestException as e:
        print(f"Error fetching the webpage: {e}")
        return [], {}
    except Exception as e:
        print(f"Error parsing the webpage: {e}")
        return [], {}


def stage_predictions(dataframe, staging_csv=STAGING_CSV, curated_csv=CURATED_CSV):
    """Write the FFS predicted XIs to a STAGING file and report the differences vs the
    curated lineups. FFS is one signal for the weekly curation (done in conversation,
    with news and judgement layered on top) — it never overwrites the curated
    starting_lineups.csv, whose graded probabilities are the pipeline's actual input."""
    dataframe.to_csv(staging_csv, index=False)
    print(f"FFS predictions staged: {len(dataframe)} players -> {staging_csv}")

    try:
        curated = pd.read_csv(curated_csv)
    except FileNotFoundError:
        return

    # Compare on accent/case-folded names so "Antonín Kinský" matches FFS's
    # "Antonin Kinsky" — the diff should show real disagreements, not encoding noise
    def key(name):
        return convert_special_characters(str(name)).casefold()

    ffs = {key(n): n for n in dataframe["Player"]}
    cur = {key(n): n for n in curated["Player"]}
    only_ffs = sorted(ffs[k] for k in ffs.keys() - cur.keys())
    only_cur = sorted(cur[k] for k in cur.keys() - ffs.keys())
    if only_ffs:
        print(f"  FFS predicts but NOT in curated lineups ({len(only_ffs)}): {', '.join(only_ffs)}")
    if only_cur:
        print(f"  In curated lineups but NOT FFS-predicted ({len(only_cur)}): {', '.join(only_cur)}")
    if not only_ffs and not only_cur:
        print("  FFS predictions match the curated player set exactly")


def stage_team_news(news_data, teams_data, news_md=NEWS_MD):
    """Write the per-team write-ups (plus each predicted XI) to a markdown brief for
    the weekly curation read-through. Nothing in the pipeline consumes this file."""
    from datetime import date

    xi_by_team = {}
    for row in teams_data:
        xi_by_team.setdefault(row['Team'], []).append(row['Player'])

    lines = [f"# FFS Team News — staged {date.today().isoformat()}",
             "",
             "*Curation context only — nothing in the pipeline reads this file.*"]
    flagged = []
    for team in sorted(news_data):
        news = news_data[team]
        lines += ["", f"## {team}"]
        if news.get('next_match'):
            lines.append(f"- **Next match:** {news['next_match']}")
        for key, label in (('out', 'Out'), ('doubts', 'Doubts'), ('banned', 'Banned')):
            players = news.get(key)
            if players:
                lines.append(f"- **{label}:** {', '.join(players)}")
        if news.get('out') or news.get('doubts'):
            flagged.append(team)
        if xi_by_team.get(team):
            lines.append(f"- **Predicted XI:** {', '.join(xi_by_team[team])}")
        if news.get('updated'):
            lines.append(f"- *{news['updated']}*")
        if news.get('latest'):
            lines += ["", news['latest']]

    with open(news_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Team news staged: {len(news_data)} teams -> {news_md}")
    if flagged:
        print(f"  Teams with outs/doubts to review: {', '.join(flagged)}")


def save(teams_data, news_data=None):
    """Stage the scraped FFS predictions for curation (never overwrites the curated lineups)."""
    if not teams_data:
        print("No data to save")
        return

    df = pd.DataFrame(teams_data, columns=['Player', 'Team'])
    stage_predictions(df)
    if news_data:
        stage_team_news(news_data, teams_data)
    print(f"Total players extracted: {len(teams_data)}")


def main():
    """
    Main function to run the scraper
    """
    print("Scraping FFS Team News for predicted lineups...")

    teams_data, news_data = get_team_lineups()

    if teams_data:
        save(teams_data, news_data)

        # Summary by team
        teams_count = {}
        for player in teams_data:
            team = player['Team']
            teams_count[team] = teams_count.get(team, 0) + 1

        print(f"\nPlayers per team:")
        for team, count in sorted(teams_count.items()):
            print(f"{team}: {count} players")

    else:
        print("No data was extracted.")


if __name__ == "__main__":
    main()
