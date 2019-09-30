import urllib.request
import re

from inscriptis import get_text


def wiki(theme):
    url = "https://en.wikipedia.org/wiki/" + theme
    html = urllib.request.urlopen(url).read().decode('utf-8')

    text = get_text(html)

    with open('../datasets/wiki_' + theme + '.txt', 'w') as out:
        for row in text.split('\n'):
            if len(row) >= 80 and not row[0].isdigit() and not row[1].isdigit() and not row[2] == '*':
                row = re.sub(r'\[\d+]', '', row)
                row = row.rstrip('\n')
                out.write(row)
                out.write('\n')


if __name__ == '__main__':

    # Video games topic
    wiki('Video_game-related_health_problems')
    wiki('Video_game_addiction_in_China')
    wiki('Video_game_addiction')
    wiki('Gaming_disorder')
    wiki('2017_in_video_gaming')
    wiki('2018_in_video_gaming')
    wiki('History_of_video_games')
    wiki('PC_game')
    wiki('Video_game_culture')
    wiki('Gaming_computer')
    wiki('Video_game_console')
    wiki('Video_gaming_in_the_United_States')
    wiki('Video_game_music')
    wiki('Video_game_industry')
    wiki('Video_game_development')
    wiki('Game_design')
    wiki('Video_game_programmer')
    wiki('Early_history_of_video_games')
    wiki('1990s_in_video_gaming')
    wiki('Video_gaming_in_Japan')
    wiki('Video_game_crash_of_1983')
    wiki('Sixth_generation_of_video_game_consoles')
    wiki('Video_gaming_in_China')
    wiki('1980s_in_video_gaming')
    wiki('Home_computer')
    wiki('Nintendo')
    wiki('Game_Boy')
    wiki('Fourth_generation_of_video_game_consoles')
    wiki('Game')
    wiki('The_Game_Awards')

    # Democracy topic
    wiki('Democracy')
    wiki('History_of_democracy')
    wiki('Athenian_democracy')
    wiki('Representative democracy')
    wiki('Direct_democracy')
    wiki('Types_of_democracy')
    wiki('Democracy_Index')
    wiki('Criticism_of_democracy')

    # Multiculturalism
    wiki('Multiculturalism')
    wiki('Criticism_of_multiculturalism')
    wiki('Cultural_pluralism')
    wiki('Multiculturalism_in_Canada')
    wiki('Multicultural_education')
    wiki('Multicultural_and_diversity_management')
    wiki('Multiculturalism_in_Australia')
    wiki('Multicultural_transruption')

    # wiki('Video_game')
