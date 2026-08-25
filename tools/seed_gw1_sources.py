# -*- coding: utf-8 -*-
"""One-off GW1 back-fill of the external cross-check sources into the source-history ledger.

Transcribed verbatim from each source's PRE-DEADLINE GW1 predicted XIs (the predictions we're
scoring), then resolved to the roster by tools/source_history.Resolver. Run once:

    env/Scripts/python tools/seed_gw1_sources.py

Going forward the weekly deadline sweep writes these straight to the ledger via
source_history.capture(); this script only exists to seed GW1, whose sources predate the tooling.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fpl_pipeline import config  # noqa: E402
from tools.source_history import Resolver, capture  # noqa: E402

# AAF labels -> roster team names
TEAM = {
    "AFC Bournemouth": "Bournemouth", "Brighton and Hove Albion": "Brighton",
    "Leeds United": "Leeds", "Manchester City": "Man City", "Manchester United": "Man Utd",
    "Newcastle United": "Newcastle", "Nottingham Forest": "Nott'm Forest",
    "Tottenham Hotspur": "Spurs",
}

# allaboutfpl.com "Predicted GW1 Lineups of All 20 PL Teams" — raw article text, 2026-08-21 update.
AAF = {
    "AFC Bournemouth": ["Petrovic", "Smith", "Silva", "Hill", "Truffert", "Adams", "Scott", "Rayan", "Kluivert", "Tavernier", "Evanilson"],
    "Arsenal": ["Raya", "White", "Mosquera", "Gabriel", "Calafiori", "Rice", "Lewis-Skelly", "Madueke", "Odegaard", "Tzolis", "Havertz"],
    "Aston Villa": ["Bizot", "Cash", "Lindelof", "Pau", "Maatsen", "Bogarde", "Kamara", "McGinn", "Buendia", "Hemmings", "Abraham"],
    "Brentford": ["Kelleher", "Kayode", "Ajer", "Collins", "Lewis-Potter", "Janelt", "Sangare", "Ouattara", "Jensen", "Schade", "Thiago"],
    "Brighton and Hove Albion": ["Verbruggen", "Wieffer", "Vuskovic", "Boscagli", "Kadioglu", "Gross", "Ayari", "Gomez", "Hinshelwood", "De Cuyper", "Rutter"],
    "Chelsea": ["Sanchez", "James", "Lacroix", "Colwill", "Neto", "Caicedo", "Enzo", "Palestra", "Palmer", "Joao Pedro", "Rogers"],
    "Coventry City": ["Rushworth", "Van Ewijk", "Amenda", "Thomas", "Da Silva", "Grimes", "Onyeka", "Tchaouna", "Yirenkyi", "Thomas-Asante", "Simms"],
    "Crystal Palace": ["Henderson", "Richards", "Canvot", "Riad", "Mingueza", "Wharton", "Kamada", "Mitchell", "Sarr", "Strand Larsen", "McNeil"],
    "Everton": ["Pickford", "O'Brien", "Tarkowski", "Branthwaite", "Mykolenko", "Hackney", "Armstrong", "Ndiaye", "Dewsbury-Hall", "George", "Barry"],
    "Fulham": ["Leno", "Castagne", "Cuenca", "Bassey", "Robinson", "Berge", "Iwobi", "Bobb", "King", "Palacios", "Garcia"],
    "Hull City": ["Tzolakis", "Coyle", "Ajayi", "Egan", "Giles", "Stroud", "Slater", "Hjerto-Dahl", "Belloumi", "McBurnie", "Millar"],
    "Ipswich Town": ["Scherpen", "O'Shea", "Diop", "Greaves", "Davis", "Florentino", "Lukic", "Fatawu", "Egeli", "Maeda", "Emersonn"],
    "Leeds United": ["Trafford", "Rodon", "Bijol", "Muharemovic", "Bogle", "Stach", "Ampadu", "Justin", "Wilson", "DCL", "Aaronson"],
    "Liverpool": ["Alisson", "Frimpong", "Jacquet", "Van Dijk", "Kerkez", "Gravenberch", "Szoboszlai", "Ngumoha", "Wirtz", "Gakpo", "Isak"],
    "Manchester City": ["Donnarumma", "Nunes", "Dias", "Gvardiol", "O'Reilly", "Kovacic", "Anderson", "Cherki", "Foden", "Semenyo", "Haaland"],
    "Manchester United": ["Lammens", "Dalot", "Maguire", "Heaven", "Shaw", "Santos", "Tielemens", "Amad", "Bruno", "Cunha", "Mbeumo"],
    "Newcastle United": ["Hornicek", "Dedic", "Thiaw", "Burn", "Hall", "Steur", "Bamba", "Elanga", "Woltemade", "Toure", "Wissa"],
    "Nottingham Forest": ["Sels", "Murillo", "Milenkovic", "Diomande", "Aina", "Sangare", "McAtee", "Williams", "Ndoye", "Jesus", "Gibbs-White"],
    "Sunderland": ["Roefs", "Mukiele", "Ballard", "O'Nein", "Reinildo", "Xhaka", "Sadiki", "Hume", "Le Fee", "Angulo", "Brobbey"],
    "Tottenham Hotspur": ["Kinsky", "Gray", "Van Hecke", "Senesi", "Robertson", "Tonali", "Fernandes", "Moore", "Gallagher", "Tel", "Richarlison"],
}


def main():
    resolver = Resolver()
    pairs = [(TEAM.get(t, t), p) for t, xi in AAF.items() for p in xi]
    capture(config.SEASON, 1, "AllAboutFPL", pairs, resolver=resolver)


if __name__ == "__main__":
    main()
