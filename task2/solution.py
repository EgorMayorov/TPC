start_parenthesis = r'(\(.\)|{.}|\[.\])*'
def build_parenthesis_regexp(result, depth):
    if depth == 1:
        return result.replace('.', '')
    result = result.replace('.', start_parenthesis)
    return build_parenthesis_regexp(result, depth - 1)

PARENTHESIS_REGEXP = build_parenthesis_regexp(start_parenthesis, 9)

SENTENCES_REGEXP = r'(?P<sentence>(?:[А-Я\d][^.?!]*:\s+(?:\d+\.\s+[А-Я][^;]*;\s*)+(?:\d+\.\s+[А-Я][^\.\n]*\.\s*))|(?:[А-Я\d]+(?:\.\s)?[^.?!:]*(?:\:\n|\?|\!|\.))|(?:[А-Я\d]+(?:\.\s+)?[^.?!]*[\.\n]))'

PERSONS_REGEXP = r'(?P<person>(?:(?:[А-Я][а-я]+) )?[А-Я][а-я]*(?:ов|ев|ан|ин)(?:а|у|е|ым|ой)?(?![а-я])(?: [А-Я][а-я]+)?)'

serial_name = r'<h1.*><a.*href=\"\/series\/\d*\/\">(?P<name>.*)<\/a><\/h1>'
episodes_count = r'<td.*><b>Эпизоды(?:.|\n)*<td.*>(?P<episodes_count>\d+)<\/td>'
season_info = r'<td.*><h1.*>Сезон\s(?P<season>\d+).*<\/h1>\s*(?P<season_year>\d{4}),\sэпизодов:\s(?P<season_episodes>\d+)\s*<\/td>'
episode_info = r'<span.*>Эпизод\s(?P<episode_number>\d+)<\/span><br\/>\s*<h1.*><b>(?P<episode_name>.*)<\/b><\/h1>\s*(?:<span.*>(?P<original_name>.*)<\/span>.*)?<\/td>\s*<td.*>(?P<episode_date>.*)<\/td>'

SERIES_REGEXP = '|'.join([serial_name, episodes_count, season_info, episode_info])

