# JOB-FIT-ANALYST
It creates critical analysis whether your cv really fits in the analysis or not.




























import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# Download NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

def preprocess_text(text):
    """Clean and tokenize text"""
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words and len(word) > 2]

def extract_keywords(text, top_n=50):
    """Extract top keywords from text"""
    tokens = preprocess_text(text)
    word_freq = Counter(tokens)
    return [word for word, freq in word_freq.most_common(top_n)]

def calculate_overall_fit(resume_text, jd_text):
    """Calculate overall job fit percentage using TF-IDF cosine similarity"""
    resume_processed = ' '.join(preprocess_text(resume_text))
    jd_processed = ' '.join(preprocess_text(jd_text))
    
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1,2))
    tfidf_matrix = vectorizer.fit_transform([resume_processed, jd_processed])
    
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(similarity * 100, 1)

def calculate_skills_match(resume_text, jd_text):
    """Calculate skills match percentage based on keyword overlap"""
    resume_keywords = set(extract_keywords(resume_text))
    jd_keywords = set(extract_keywords(jd_text))
    
    matching_skills = resume_keywords.intersection(jd_keywords)
    skills_match_rate = (len(matching_skills) / len(jd_keywords)) * 100 if jd_keywords else 0
    return round(skills_match_rate, 1), list(matching_skills)

def read_file_content(file_path):
    """Read content from file with error handling"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def job_fit_analyzer():
    """Main interactive analyzer following exact flowchart"""
    print("🚀 BRUTALLY HONEST JOB FIT ANALYZER")
    print("=" * 50)
    
    # Step 1: Ask for Master CV
    print("\n📄 STEP 1: Upload your Master CV")
    cv_path = input("Enter path to your CV file (e.g., resume.pdf.txt or resume.txt): ").strip()
    
    try:
        cv_content = read_file_content(cv_path)
        print("✅ CV loaded successfully!")
    except Exception as e:
        print(f"❌ Error reading CV: {e}")
        return
    
    # Step 2: Ask for JD
    print("\n📋 STEP 2: Upload Job Description")
    jd_path = input("Enter path to JD file (e.g., job_description.txt): ").strip()
    
    try:
        jd_content = read_file_content(jd_path)
        print("✅ JD loaded successfully!")
    except Exception as e:
        print(f"❌ Error reading JD: {e}")
        return
    
    # Step 3: Analyze
    print("\n🔍 STEP 3: ANALYZING FIT...")
    print("Running NLP analysis...")
    
    # Calculate metrics
    overall_fit = calculate_overall_fit(cv_content, jd_content)
    skills_match_pct, matching_skills = calculate_skills_match(cv_content, jd_content)
    
    # Step 4: Return Results
    print("\n" + "="*50)
    print("🎯 JOB FIT RESULTS")
    print("="*50)
    print(f"📊 OVERALL FIT TO JOB: {overall_fit}%")
    print(f"🔧 SKILLS MATCH TO JD:  {skills_match_pct}%")
    print()
    
    # Quick interpretation
    if overall_fit >= 80:
        status = "🟢 EXCELLENT - APPLY IMMEDIATELY"
    elif overall_fit >= 65:
        status = "🟡 GOOD - TAILOR & APPLY"
    elif overall_fit >= 50:
        status = "🟠 FAIR - UPSKILL FIRST"
    else:
        status = "🔴 POOR - LOOK ELSEWHERE"
    
    print(f"💡 RECOMMENDATION: {status}")
    
    if matching_skills:
        print(f"\n✅ TOP MATCHING SKILLS ({len(matching_skills)}):")
        print(", ".join(matching_skills[:10]))
    else:
        print("\n⚠️  No clear skill matches found")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    job_fit_analyzer()
