from bs4 import BeautifulSoup
import requests


def get_all_locations():
    letters = (chr(code) for code in range(65, 91))
    locations = []
    for letter in letters:
        locations.extend(get_locations_for_letter(letter))
    return locations


def get_locations_for_letter(letter):
    print(f"getting locations for letter {letter}")
    soup = BeautifulSoup(requests.get("http://www.deutsche-staedte.de/staedte.php?city=" + letter).content)
    ul = soup.find("ul", style="list-style-type: square; list-style-position: outside; padding-left: 25px;")
    return (location_link.text.strip() for location_link in ul.find_all("a") if location_link.text.strip() != "")


if __name__ == "__main__":
    locations = get_all_locations()
    with open("data/locations", mode="w", encoding="UTF-8") as fp:
        for loc in locations:
            if loc.strip() != "":
                fp.write(f"{loc}\n")
