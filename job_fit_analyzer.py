import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words and len(word) > 2]


def extract_keywords(text, top_n=50):
    tokens = preprocess_text(text)
    word_freq = Counter(tokens)
    return [word for word, freq in word_freq.most_common(top_n)]


def calculate_overall_fit(cv_text, jd_text):
    cv_processed = ' '.join(preprocess_text(cv_text))
    jd_processed = ' '.join(preprocess_text(jd_text))
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([cv_processed, jd_processed])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(similarity * 100, 1)


def calculate_skills_match(cv_text, jd_text):
    cv_keywords = set(extract_keywords(cv_text))
    jd_keywords = set(extract_keywords(jd_text))
    matching_skills = cv_keywords.intersection(jd_keywords)
    skills_match_rate = (len(matching_skills) / len(jd_keywords)) * 100 if jd_keywords else 0
    return round(skills_match_rate, 1), list(matching_skills)


def is_good_fit(overall_fit):
    if overall_fit >= 75:
        return 'YES - Excellent fit'
    elif overall_fit >= 60:
        return 'Maybe - Good fit'
    elif overall_fit >= 45:
        return 'Borderline - Needs work'
    else:
        return 'No - Poor fit'


def get_multiline_input(prompt):
    print(prompt)
    print('Paste content below. Press Enter on an empty line when done.')
    lines = []
    while True:
        line = input()
        if line.strip() == '' and lines:
            break
        lines.append(line)
    return '\n'.join(lines)


def job_fit_analyzer():
    print('BRUTALLY HONEST JOB FIT ANALYZER')
    print('=' * 50)
    cv_content = get_multiline_input('STEP 1: Paste your MASTER CV')
    print('CV captured!')
    jd_content = get_multiline_input('STEP 2: Paste the JOB DESCRIPTION')
    print('JD captured!')
    print('STEP 3: Analyzing...')
    overall_fit = calculate_overall_fit(cv_content, jd_content)
    skills_match_pct, matching_skills = calculate_skills_match(cv_content, jd_content)
    print('\n' + '=' * 50)
    print('JOB FIT RESULTS')
    print('=' * 50)
    print(f'Overall Fit Score: {overall_fit}%')
    print(f'Skills Match Score: {skills_match_pct}%')
    print(f'Fit Status: {is_good_fit(overall_fit)}')
    if matching_skills:
        print('\nMatching Skills:')
        print(', '.join(matching_skills[:10]))
    else:
        print('\nMatching Skills: None found')
    print('=' * 50)


if __name__ == '__main__':
    job_fit_analyzer()
