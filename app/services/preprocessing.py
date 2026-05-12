import re
import nltk
from nltk.corpus import stopwords

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    STOP = (
        set(stopwords.words('russian')) |
        set(stopwords.words('english')) |
        set(stopwords.words('kazakh'))
    )
except OSError:
    STOP = (
        set(stopwords.words('russian')) |
        set(stopwords.words('english'))
    )

BOILER_PATS = [
    r'қазақстан\s+республикасы',
    r'министрлігі|министрлiгi',
    r'сәтбаев\s+университет|satbayev\s+university',
    r'бекітемін|бекiтемiн',
    r'қорғауға\s+жіберілді',
    r'ғылыми\s+жетекшінің\s+сын',
    r'программалық\s+инженерия\s+кафедрасы',
    r'отчет\s+подобия',
    r'scanned\s+with',
]
ABSTRACT_PATS = [r'аңдатпа', r'аннотация', r'\bannotation\b', r'\babstract\b']
TOC_PATS      = [r'мазмұны', r'содержание', r'table\s+of\s+contents']
BIBLIO_PATS   = [
    r'пайдаланылған\s+әдебиеттер',
    r'список\s+(использованн|литератур)',
    r'\bbibliography\b',
    r'\breferences\b',
]
APPENDIX_PATS = [
    r'(?:^|\n)\s*қосымша(?:лар)?\s*[а-дА-ДA-D1-9]',
    r'(?:^|\n)\s*қосымшалар\s*$',
    r'(?:^|\n)\s*приложени[ея]',
    r'(?:^|\n)\s*appendix\b',
]

MIN_CHARS  = 30
HEADER_LEN = 500

def extract_body(pages: list[str]) -> str:
    n = len(pages)
    if n == 0:
        return ''

    end_idx = n
    search_from = max(n // 2, 5)
    for i in range(n - 1, search_from - 1, -1):
        header = pages[i].lower().strip()[:HEADER_LEN]
        if len(header) < MIN_CHARS:
            continue
        if any(re.search(p, header) for p in BIBLIO_PATS):
            end_idx = i
            break
        if any(re.search(p, header, re.MULTILINE) for p in APPENDIX_PATS):
            end_idx = i
            break

    max_front_skip = min(10, n // 3 + 1)
    skip_set = set()
    for i in range(min(max_front_skip, end_idx)):
        text = pages[i].strip()
        if len(text) < MIN_CHARS:
            skip_set.add(i)
            continue
        header = text.lower()[:HEADER_LEN]
        if any(re.search(p, header) for p in BOILER_PATS + ABSTRACT_PATS + TOC_PATS):
            skip_set.add(i)

    body_parts = []
    for i in range(end_idx):
        if i in skip_set:
            continue
        text = pages[i].strip()
        if len(text) < MIN_CHARS:
            continue
        body_parts.append(text)

    return '\n'.join(body_parts)

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^0-9a-zа-яәіңғүұқөһё\s]', ' ', text)
    tokens = [w for w in text.split() if w not in STOP and len(w) > 1]
    return re.sub(r'\s+', ' ', ' '.join(tokens)).strip()
