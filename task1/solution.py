rgb_color = r'^rgb\((?P<rgb_variant>(?P<numbers>(?P<num1>\d|[1-9]\d|1\d\d|2[0-4]\d|25[0-5]),\s*(?P<num2>P\d|[1-9]\d|1\d\d|2[0-4]\d|25[0-5]),\s*(?P<num3>P\d|[1-9]\d|1\d\d|2[0-4]\d|25[0-5]))|(?P<persents>(?P<per1>(\d|[1-9]\d|100)%),\s*(?P<per2>(\d|[1-9]\d|100)%),\s*(?P<per3>(\d|[1-9]\d|100)%)))\)$'
hex_color = r'^#(?P<color>(?P<hex_variant>(?P<two_color1>[\da-fA-F][\da-fA-F])(?P<two_color2>[\da-fA-F][\da-fA-F])(?P<two_color3>[\da-fA-F][\da-fA-F])|(?P<one_color1>[\da-fA-F])(?P<one_color2>[\da-fA-F])(?P<one_color3>[\da-fA-F])))$'
hsl_color = r'^hsl\((?P<ton>\d|[1-9]\d|[12]\d\d|3[0-5]\d|360),\s*(?P<persent1>(?P<value1>\d|[1-9]\d|100)%),\s*(?P<persent2>(?P<value2>\d|[1-9]\d|100)%)\)$'

var = r'(?P<variable>[a-zA-Z_][0-9a-zA-Z_]*)'
num = r'(?P<number>(0\.\d+)|([1-9]\d*(\.\d*)?))'
const = r'(?P<constant>(pi|e|sqrt2|ln2|ln10)(?![a-zA-Z_0-9]))'
func = r'(?P<function>(sin|cos|tg|ctg|tan|cot|sinh|cosh|th|cth|tanh|coth|ln|lg|log|exp|sqrt|cbrt|abs|sign)(?![a-zA-Z_0-9]))'
operator = r'(?P<operator>[\^\-\+\/\*])'
left_par = r'(?P<left_parenthesis>\()'
right_par = r'(?P<right_parenthesis>\))'

date_ddmmyy = r'^(((0?[1-9]|[12]\d|3[01])(?P<sep1>[\.\-\/])(0?[13578]|10|12)(?P=sep1)(\d+))|((0?[1-9]|[12]\d|30)(?P<sep2>[\.\-\/])(0?[469]|11)(?P=sep2)(\d+))|((0?[1-9]|1\d|2[0-8])(?P<sep3>[\.\-\/])(0?2)(?P=sep3)(\d+)))$'
date_yymmdd = r'^(([1-9]\d*)(?P<sep4>[\.\-\/])(0?[13578]|1[02])(?P=sep4)(0?[1-9]|[12]\d|3[01])|(([1-9]\d*)(?P<sep5>[\.\-\/])(0?[469]|11)(?P=sep5)(0?[1-9]|[12]\d|30))|(([1-9]\d*)(?P<sep6>[\.\-\/])(0?2)(?P=sep6)(0?[1-9]|1\d|2[0-8])))$'
date_ddmmrusyy = r'^(((0?[1-9]|[12]\d|3[01])\s+(января|марта|мая|июля|августа|октября|декабря)\s+(\d+))|((0?[1-9]|[12]\d|30])\s+(апреля|июня|сентября|ноября)\s+(\d+))|((0?[1-9]|1\d|2[0-8])\s+(февраля)\s+(\d+)))$'
date_mmengddyy = r'^(([Jj]an(uary?)|[Mm]ar(ch)?|[Mm]ay|[Jj]ul(y)?|[Aa]ug(ust)?|[Oo]ct(ober)?|[Dd]ec(ember)?)\s+(0?[1-9]|[12]\d|3[01])\,\s*(\d+))|(([Aa]pr(il)?|[Jj]un(e)?|[Ss]ep(tember)?|[Nn]ov(ember)?)\s+(0?[1-9]|[12]\d|30)\,\s*(\d+))|([Ff]eb(ruary)?\s+(0?[1-9]|1\d|2[0-8])\,\s*(\d+))$'
date_yymmengdd = r'^((\d+)\,\s*([Jj]an(uary?)|[Mm]ar(ch)?|[Mm]ay|[Jj]ul(y)?|[Aa]ug(ust)?|[Oo]ct(ober)?|[Dd]ec(ember)?)\s+(0?[1-9]|[12]\d|3[01]))|((\d+)\,\s*([Aa]pr(il)?|[Jj]un(e)?|[Ss]ep(tember)?|[Nn]ov(ember)?)\s+(0?[1-9]|[12]\d|30))|((\d+)\,\s*[Ff]eb(ruary)?\s+(0?[1-9]|1\d|2[0-8]))$'

COLOR_REGEXP = '|'.join([rgb_color, hex_color, hsl_color])
PASSWORD_REGEXP = r'^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=(?:(?P<symbol>.)(?!(?P=symbol)))+$)(?=(.*(?P<spec_symbol>[\^$%@#&*!?]).*(?!(?P=spec_symbol))[\^$%@#&*!?])+)[A-Za-z0-9^$%@#&*!?]{8,}$'
EXPRESSION_REGEXP = '|'.join([num, const, func, operator, left_par, right_par, var])
DATES_REGEXP = '|'.join([date_ddmmyy, date_yymmdd, date_ddmmrusyy, date_mmengddyy, date_yymmengdd])

