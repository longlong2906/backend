"""
Flask API cho hệ thống gợi ý phim
Cung cấp các endpoint dữ liệu cho Dashboard
Sử dụng dữ liệu đã tiền xử lý (processed_dataset.csv)
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
from collections import Counter
import json
import requests
from dotenv import load_dotenv
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from pymongo import MongoClient
import bcrypt
from datetime import timedelta
import re

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)

# MongoDB Config
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
client = MongoClient(MONGO_URI)
db = client['movie_recommendation']
users_collection = db['users']
movies_collection = db['movies']

# ... (Previous code)

@app.route('/api/movies/<int:movie_id>/tmdb', methods=['GET'])
def get_tmdb_info(movie_id):
    """Lấy thông tin từ TMDB API (có cache vào MongoDB)"""
    
    # 1. Check Cache in MongoDB
    cached_movie = movies_collection.find_one({'_id': movie_id})
    if cached_movie:
        # Return cached data (exclude internal fields if any)
        return jsonify(cached_movie)
    
    # 2. Prepare Data from CSV (Base info)
    movie_csv = df[df['id'] == movie_id]
    if movie_csv.empty:
        # If not in CSV, we might still fetch from TMDB, but let's strictly rely on known movies for now or just proceed.
        # Proceeding allows catching new movies if system expands.
        base_info = {'id': movie_id}
    else:
        row = movie_csv.iloc[0]
        # Helper to safely getting genre list
        genre_list = []
        if pd.notna(row.get('genre_list')):
            try:
                val = row['genre_list']
                if isinstance(val, str):
                    clean_str = val.strip("[]")
                    genre_list_raw = [x.strip().strip("'\"") for x in clean_str.split(',') if x.strip()]
                else:
                    genre_list_raw = val
                    
                genre_list = [get_genre_name_vi(g) for g in genre_list_raw if g]
            except Exception as e:
                print(f"Error parsing genre_list in get_tmdb_info: {e}")
                
        base_info = {
            '_id': int(row['id']), # Use ID as MongoDB _id
            'id': int(row['id']),
            'title': str(row['title']),
            'overview': str(row['overview']) if pd.notna(row.get('overview')) else '',
            'vote_average': float(row['vote_average']) if pd.notna(row.get('vote_average')) else 0.0,
            'vote_count': int(row['vote_count']) if pd.notna(row.get('vote_count')) else 0,
            'weighted_rating': float(row['weighted_rating']) if pd.notna(row.get('weighted_rating')) else 0.0,
            'release_year': int(row['release_year']) if pd.notna(row.get('release_year')) else None,
            'genre': str(row.get('genre', '')),
            'genre_list': genre_list,
            'original_language': str(row.get('original_language', '')),
            'popularity': float(row['popularity']) if pd.notna(row.get('popularity')) else 0.0
        }

    # 3. Fetch from TMDB API
    tmdb_api_key = os.getenv('TMDB_API_KEY', '')
    tmdb_info = {
        'poster_path': None,
        'backdrop_path': None,
        'release_date': None,
        'runtime': None,
        'tagline': None,
        'homepage': None,
        'imdb_id': None,
        'status': None
    }

    if tmdb_api_key:
        try:
            url = f"https://api.themoviedb.org/3/movie/{movie_id}"
            params = {'api_key': tmdb_api_key, 'language': 'vi-VN'}
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                tmdb_info = {
                    'poster_path': data.get('poster_path'),
                    'backdrop_path': data.get('backdrop_path'),
                    'release_date': data.get('release_date'),
                    'runtime': data.get('runtime'),
                    'tagline': data.get('tagline'),
                    'homepage': data.get('homepage'),
                    'imdb_id': data.get('imdb_id'),
                    'status': data.get('status')
                }
                
                # Update base info with better overview/title from TMDB if available and CSV was poor?
                # For now, we prioritize CSV for stats coherence, but TMDB images are key.
                # Actually user requested to take movie info from TMDB. 
                # Let's overwrite CSV overview if TMDB has one (often better/localized).
                if data.get('overview'):
                    base_info['overview'] = data['overview']
                if data.get('title'):
                    base_info['title'] = data['title']
                    
        except Exception as e:
            print(f"Error fetching TMDB for {movie_id}: {e}")

    # 4. Merge and Save to MongoDB
    final_data = {**base_info, **tmdb_info}
    
    # Update or Insert (upsert=True just in case)
    try:
        movies_collection.replace_one({'_id': movie_id}, final_data, upsert=True)
    except Exception as e:
        print(f"Error saving to MongoDB: {e}")
    
    return jsonify(final_data)
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'super-secret-key-please-change')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=7)
jwt = JWTManager(app)

# Đường dẫn file dữ liệu đã tiền xử lý
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'processed_dataset.csv')

def load_data():
    """Load dữ liệu đã tiền xử lý"""
    df = pd.read_csv(DATA_PATH)
    return df

# Load dữ liệu một lần khi khởi động
df = load_data()

# Lấy danh sách các cột genre one-hot
GENRE_COLUMNS = [col for col in df.columns if col.startswith('genre_') and col != 'genre_list']

# Mapping tên thể loại từ tiếng Anh sang tiếng Việt
GENRE_NAMES_VI = {
    'action': 'Hành động',
    'adventure': 'Phiêu lưu',
    'animation': 'Hoạt hình',
    'comedy': 'Hài hước',
    'crime': 'Tội phạm',
    'documentary': 'Tài liệu',
    'drama': 'Chính kịch',
    'family': 'Gia đình',
    'fantasy': 'Giả tưởng',
    'history': 'Lịch sử',
    'horror': 'Kinh dị',
    'music': 'Âm nhạc',
    'mystery': 'Bí ẩn',
    'romance': 'Lãng mạn',
    'science fiction': 'Khoa học viễn tưởng',
    'tv movie': 'Phim truyền hình',
    'thriller': 'Giật gân',
    'war': 'Chiến tranh',
    'western': 'Miền tây',
    'unknown': 'Không xác định'
}

def get_genre_name_vi(genre_code: str) -> str:
    """Chuyển đổi mã thể loại sang tên tiếng Việt"""
    genre_code_lower = genre_code.lower().replace('_', ' ').replace('-', ' ')
    return GENRE_NAMES_VI.get(genre_code_lower, genre_code.replace('_', ' ').title())

# Mapping tên ngôn ngữ từ tiếng Anh sang tiếng Việt
LANGUAGE_NAMES_VI = {
    'en': 'Anh',
    'ja': 'Nhật Bản',
    'ko': 'Hàn Quốc',
    'fr': 'Pháp',
    'es': 'Tây Ban Nha',
    'it': 'Ý',
    'de': 'Đức',
    'hi': 'Ấn Độ',
    'zh': 'Trung Quốc',
    'cn': 'Trung Quốc',
    'pt': 'Bồ Đào Nha',
    'ru': 'Nga',
    'th': 'Thái Lan',
    'vi': 'Việt Nam',
    'ar': 'Ả Rập',
    'tr': 'Thổ Nhĩ Kỳ',
    'sv': 'Thụy Điển',
    'nl': 'Hà Lan',
    'pl': 'Ba Lan',
    'da': 'Đan Mạch',
    'no': 'Na Uy',
    'fi': 'Phần Lan',
    'cs': 'Séc',
    'hu': 'Hungary',
    'ro': 'Romania',
    'he': 'Do Thái',
    'id': 'Indonesia',
    'ms': 'Malaysia',
    'sk': 'Slovakia',
    'bg': 'Bulgaria',
    'hr': 'Croatia',
    'el': 'Hy Lạp',
    'uk': 'Ukraine',
    'is': 'Iceland',
    'et': 'Estonia',
    'lv': 'Latvia',
    'lt': 'Lithuania'
}

def get_language_name_vi(lang_code: str) -> str:
    """Chuyển đổi mã ngôn ngữ sang tên tiếng Việt"""
    return LANGUAGE_NAMES_VI.get(lang_code.lower(), lang_code.upper())


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Thống kê tổng quan"""
    stats = {
        'total_movies': len(df),
        'avg_rating': round(df['vote_average'].mean(), 2),
        'avg_weighted_rating': round(df['weighted_rating'].mean(), 2),
        'total_votes': int(df['vote_count'].sum()),
        'genres_count': len(GENRE_COLUMNS),
        'languages_count': df['original_language'].nunique(),
        'year_range': {
            'min': int(df['release_year'].min()),
            'max': int(df['release_year'].max())
        },
        'avg_movie_age': round(df['movie_age'].mean(), 1),
        'avg_overview_length': round(df['overview_length'].mean(), 1)
    }
    return jsonify(stats)


@app.route('/api/top-movies', methods=['GET'])
def get_top_movies():
    """Top 10 phim có weighted_rating cao nhất"""
    top = df.nlargest(10, 'weighted_rating')[
        ['id', 'title', 'vote_average', 'vote_count', 'genre', 'weighted_rating', 'release_year', 'decade']
    ].to_dict('records')
    return jsonify(top)


@app.route('/api/genre-distribution', methods=['GET'])
def get_genre_distribution():
    """Phân bố thể loại phim (từ one-hot encoding)"""
    genre_counts = {}
    for col in GENRE_COLUMNS:
        genre_name = col.replace('genre_', '').replace('_', ' ').title()
        count = int(df[col].sum())
        if count > 0:
            genre_counts[genre_name] = count
    
    # Sắp xếp theo số lượng
    result = [{'genre': k, 'count': v} for k, v in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)]
    return jsonify(result[:15])


@app.route('/api/rating-distribution', methods=['GET'])
def get_rating_distribution():
    """Phân bố điểm đánh giá (histogram)"""
    bins = np.arange(0, 10.5, 0.5)
    hist, edges = np.histogram(df['vote_average'].dropna(), bins=bins)
    result = [{'range': f'{edges[i]:.1f}-{edges[i+1]:.1f}', 'count': int(hist[i])} 
              for i in range(len(hist))]
    return jsonify(result)


@app.route('/api/weighted-rating-distribution', methods=['GET'])
def get_weighted_rating_distribution():
    """Phân bố weighted rating (histogram)"""
    bins = np.arange(4, 9, 0.25)
    hist, edges = np.histogram(df['weighted_rating'].dropna(), bins=bins)
    result = [{'range': f'{edges[i]:.2f}-{edges[i+1]:.2f}', 'count': int(hist[i])} 
              for i in range(len(hist))]
    return jsonify(result)


@app.route('/api/movies-by-year', methods=['GET'])
def get_movies_by_year():
    """Số lượng phim theo năm"""
    yearly = df.groupby('release_year').size().reset_index(name='count')
    yearly = yearly[yearly['release_year'] >= 1950]
    result = yearly.to_dict('records')
    return jsonify(result)


@app.route('/api/popularity-by-year', methods=['GET'])
def get_popularity_by_year():
    """Popularity và các chỉ số trung bình theo năm"""
    yearly = df.groupby('release_year').agg({
        'popularity': 'mean',
        'vote_average': 'mean',
        'vote_count': 'sum',
        'weighted_rating': 'mean',
        'overview_length': 'mean'
    }).reset_index()
    yearly = yearly[yearly['release_year'] >= 1980]
    yearly = yearly.round(2)
    result = yearly.to_dict('records')
    return jsonify(result)


@app.route('/api/language-distribution', methods=['GET'])
def get_language_distribution():
    """Phân bố ngôn ngữ"""
    # Hợp nhất cn và zh thành Chinese
    df_lang = df.copy()
    df_lang['original_language'] = df_lang['original_language'].replace('cn', 'zh')
    
    lang_counts = df_lang['original_language'].value_counts().head(10)
    
    # Mapping mã ngôn ngữ
    result = [{'language': get_language_name_vi(lang), 'code': lang, 'count': int(count)} 
              for lang, count in lang_counts.items()]
    return jsonify(result)


@app.route('/api/decade-distribution', methods=['GET'])
def get_decade_distribution():
    """Phân bố phim theo thập kỷ"""
    decade_counts = df['decade'].value_counts().sort_index()
    decade_counts = decade_counts[decade_counts.index >= 1950]
    result = [{'decade': f"{int(d)}s", 'year': int(d), 'count': int(c)} 
              for d, c in decade_counts.items()]
    return jsonify(result)


@app.route('/api/scatter-data', methods=['GET'])
def get_scatter_data():
    """Dữ liệu cho scatter plot (vote_count vs vote_average)"""
    sample = df.nlargest(200, 'vote_count')[
        ['title', 'vote_average', 'vote_count', 'popularity', 'genre', 
         'weighted_rating', 'release_year', 'movie_age', 'overview_length']
    ]
    result = sample.to_dict('records')
    return jsonify(result)


@app.route('/api/genre-rating', methods=['GET'])
def get_genre_rating():
    """Điểm trung bình và weighted rating theo thể loại"""
    genre_stats = []
    
    for col in GENRE_COLUMNS:
        genre_name = col.replace('genre_', '').replace('_', ' ').title()
        genre_df = df[df[col] == 1]
        
        if len(genre_df) >= 10:
            genre_stats.append({
                'genre': genre_name,
                'avg_rating': round(genre_df['vote_average'].mean(), 2),
                'avg_weighted_rating': round(genre_df['weighted_rating'].mean(), 2),
                'count': len(genre_df),
                'avg_popularity': round(genre_df['popularity'].mean(), 2),
                'avg_overview_length': round(genre_df['overview_length'].mean(), 1)
            })
    
    result = sorted(genre_stats, key=lambda x: x['avg_weighted_rating'], reverse=True)
    return jsonify(result)


@app.route('/api/heatmap-data', methods=['GET'])
def get_heatmap_data():
    """Dữ liệu tương quan cho heatmap (sử dụng các cột normalized)"""
    cols_for_corr = [
        'popularity_normalized', 'vote_average_normalized', 'vote_count_normalized', 
        'movie_age_normalized', 'overview_length_normalized', 'weighted_rating'
    ]
    
    # Đổi tên cột cho dễ đọc
    display_names = ['Popularity', 'Vote Avg', 'Vote Count', 'Movie Age', 'Overview Len', 'Weighted Rating']
    
    corr_df = df[cols_for_corr].copy()
    corr_df.columns = display_names
    corr_matrix = corr_df.corr()
    
    result = {
        'columns': display_names,
        'data': corr_matrix.values.tolist()
    }
    return jsonify(result)


@app.route('/api/treemap-data', methods=['GET'])
def get_treemap_data():
    """Dữ liệu cho treemap - thể loại theo ngôn ngữ"""
    result = []
    top_languages = df['original_language'].value_counts().head(5).index.tolist()
    
    for col in GENRE_COLUMNS[:10]:  # Top 10 genres
        genre_name = col.replace('genre_', '').replace('_', ' ').title()
        genre_df = df[df[col] == 1]
        
        for lang in top_languages:
            count = len(genre_df[genre_df['original_language'] == lang])
            if count > 0:
                result.append({
                    'genre': genre_name,
                    'language': lang,
                    'count': count
                })
    
    return jsonify(result)


@app.route('/api/wordcloud-data', methods=['GET'])
def get_wordcloud_data():
    """Dữ liệu cho word cloud từ overview_clean"""
    import re
    from collections import Counter
    
    # Stopwords cơ bản
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 
                 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
                 'dare', 'ought', 'used', 'it', 'its', 'this', 'that', 'these', 'those',
                 'i', 'you', 'he', 'she', 'we', 'they', 'what', 'which', 'who', 'whom',
                 'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
                 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
                 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now',
                 'her', 'his', 'him', 'their', 'them', 'my', 'your', 'our', 'into',
                 'after', 'before', 'between', 'under', 'over', 'out', 'up', 'down',
                 'about', 'against', 'during', 'through', 'above', 'below', 'while',
                 'find', 'finds', 'becomes', 'become', 'must', 'take', 'takes', 'life',
                 'new', 'young', 'one', 'two', 'first', 'last', 'back', 'make', 'made'}
    
    all_words = []
    for overview in df['overview_clean'].dropna():
        words = str(overview).split()
        words = [w for w in words if len(w) >= 4 and w not in stopwords]
        all_words.extend(words)
    
    word_counts = Counter(all_words).most_common(100)
    result = [{'text': word, 'value': count} for word, count in word_counts]
    return jsonify(result)


@app.route('/api/network-data', methods=['GET'])
def get_network_data():
    """Dữ liệu cho network graph - mối quan hệ giữa các thể loại"""
    from collections import defaultdict
    
    # Đếm số lần các thể loại xuất hiện cùng nhau
    genre_pairs = defaultdict(int)
    genre_counts = {}
    
    for col in GENRE_COLUMNS:
        genre_name = col.replace('genre_', '').replace('_', ' ').title()
        genre_counts[genre_name] = int(df[col].sum())
    
    # Tìm các cặp thể loại xuất hiện cùng nhau
    for _, row in df.iterrows():
        active_genres = [col.replace('genre_', '').replace('_', ' ').title() 
                        for col in GENRE_COLUMNS if row[col] == 1]
        for i in range(len(active_genres)):
            for j in range(i + 1, len(active_genres)):
                pair = tuple(sorted([active_genres[i], active_genres[j]]))
                genre_pairs[pair] += 1
    
    # Lấy top 12 genres
    top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:12]
    top_genre_names = [g[0] for g in top_genres]
    
    # Tạo nodes
    nodes = [{'id': genre, 'group': i % 5, 'size': count} 
             for i, (genre, count) in enumerate(top_genres)]
    
    # Tạo links
    links = []
    for (g1, g2), count in genre_pairs.items():
        if g1 in top_genre_names and g2 in top_genre_names and count > 50:
            links.append({'source': g1, 'target': g2, 'value': count})
    
    return jsonify({'nodes': nodes, 'links': links})


@app.route('/api/bubble-data', methods=['GET'])
def get_bubble_data():
    """Dữ liệu cho bubble chart"""
    sample = df.nlargest(50, 'popularity')[
        ['title', 'vote_average', 'vote_count', 'popularity', 'genre', 
         'release_year', 'weighted_rating', 'movie_age', 'decade']
    ]
    sample = sample.dropna()
    result = sample.to_dict('records')
    return jsonify(result)


@app.route('/api/genre-by-decade', methods=['GET'])
def get_genre_by_decade():
    """Thể loại theo thập kỷ (stacked bar)"""
    result = []
    decades = [1970, 1980, 1990, 2000, 2010, 2020]
    top_genres = ['Drama', 'Comedy', 'Action', 'Thriller', 'Romance', 'Horror', 'Adventure']
    
    for decade in decades:
        decade_df = df[df['decade'] == decade]
        genre_counts = {'decade': f"{decade}s"}
        
        for genre in top_genres:
            col_name = f"genre_{genre.lower()}"
            if col_name in df.columns:
                genre_counts[genre] = int(decade_df[col_name].sum())
            else:
                genre_counts[genre] = 0
        
        result.append(genre_counts)
    
    return jsonify(result)


@app.route('/api/rating-by-language', methods=['GET'])
def get_rating_by_language():
    """Điểm trung bình và weighted rating theo ngôn ngữ"""
    # Tạo bản sao để không ảnh hưởng đến dataframe gốc
    df_temp = df.copy()
    
    # Hợp nhất 'cn' và 'zh' thành 'zh' (Chinese)
    df_temp['original_language'] = df_temp['original_language'].replace('cn', 'zh')
    
    lang_stats = df_temp.groupby('original_language').agg({
        'vote_average': 'mean',
        'weighted_rating': 'mean',
        'vote_count': 'sum',
        'id': 'count',
        'popularity': 'mean'
    }).reset_index()
    lang_stats.columns = ['language', 'avg_rating', 'avg_weighted_rating', 'total_votes', 'movie_count', 'avg_popularity']
    lang_stats = lang_stats[lang_stats['movie_count'] >= 50]
    lang_stats = lang_stats.nlargest(10, 'movie_count')
    lang_stats = lang_stats.round(2)
    
    # Mapping tên ngôn ngữ
    lang_stats['language_name'] = lang_stats['language'].apply(get_language_name_vi)
    
    result = lang_stats.to_dict('records')
    return jsonify(result)


@app.route('/api/monthly-releases', methods=['GET'])
def get_monthly_releases():
    """Phân bố phim theo tháng"""
    monthly = df.groupby('release_month').agg({
        'id': 'count',
        'vote_average': 'mean',
        'weighted_rating': 'mean'
    }).reset_index()
    monthly.columns = ['release_month', 'count', 'avg_rating', 'avg_weighted_rating']
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly['month_name'] = monthly['release_month'].apply(
        lambda x: month_names[int(x)-1] if pd.notna(x) and 1 <= x <= 12 else ''
    )
    monthly = monthly.round(2)
    
    result = monthly.to_dict('records')
    return jsonify(result)


@app.route('/api/movie-age-analysis', methods=['GET'])
def get_movie_age_analysis():
    """Phân tích theo tuổi phim"""
    age_bins = [0, 5, 10, 20, 30, 50, 100]
    age_labels = ['0-5 years', '5-10 years', '10-20 years', '20-30 years', '30-50 years', '50+ years']
    
    df['age_group'] = pd.cut(df['movie_age'], bins=age_bins, labels=age_labels, right=False)
    
    age_stats = df.groupby('age_group', observed=True).agg({
        'id': 'count',
        'vote_average': 'mean',
        'weighted_rating': 'mean',
        'popularity': 'mean'
    }).reset_index()
    age_stats.columns = ['age_group', 'count', 'avg_rating', 'avg_weighted_rating', 'avg_popularity']
    age_stats = age_stats.round(2)
    
    result = age_stats.to_dict('records')
    return jsonify(result)


@app.route('/api/overview-length-analysis', methods=['GET'])
def get_overview_length_analysis():
    """Phân tích theo độ dài overview"""
    length_bins = [0, 20, 40, 60, 80, 100, 200]
    length_labels = ['0-20', '20-40', '40-60', '60-80', '80-100', '100+']
    
    df['length_group'] = pd.cut(df['overview_length'], bins=length_bins, labels=length_labels, right=False)
    
    length_stats = df.groupby('length_group', observed=True).agg({
        'id': 'count',
        'vote_average': 'mean',
        'weighted_rating': 'mean'
    }).reset_index()
    length_stats.columns = ['length_group', 'count', 'avg_rating', 'avg_weighted_rating']
    length_stats = length_stats.round(2)
    
    result = length_stats.to_dict('records')
    return jsonify(result)


@app.route('/api/normalized-comparison', methods=['GET'])
def get_normalized_comparison():
    """So sánh các giá trị normalized theo thập kỷ"""
    decade_stats = df.groupby('decade').agg({
        'popularity_normalized': 'mean',
        'vote_average_normalized': 'mean',
        'vote_count_normalized': 'mean',
        'overview_length_normalized': 'mean'
    }).reset_index()
    decade_stats = decade_stats[decade_stats['decade'] >= 1970]
    decade_stats = decade_stats.round(3)
    decade_stats['decade'] = decade_stats['decade'].apply(lambda x: f"{int(x)}s")
    
    result = decade_stats.to_dict('records')
    return jsonify(result)


@app.route('/api/top-by-genre/<genre>', methods=['GET'])
def get_top_by_genre(genre):
    """Top phim theo thể loại cụ thể (sử dụng weighted_rating)"""
    col_name = f"genre_{genre.lower()}"
    if col_name in df.columns:
        filtered = df[df[col_name] == 1]
        top = filtered.nlargest(10, 'weighted_rating')[
            ['title', 'vote_average', 'vote_count', 'release_year', 'weighted_rating']
        ]
        result = top.to_dict('records')
    else:
        result = []
    return jsonify(result)


@app.route('/api/dayofweek-analysis', methods=['GET'])
def get_dayofweek_analysis():
    """Phân tích theo ngày trong tuần phát hành"""
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    day_stats = df.groupby('release_dayofweek').agg({
        'id': 'count',
        'vote_average': 'mean',
        'weighted_rating': 'mean',
        'popularity': 'mean'
    }).reset_index()
    day_stats.columns = ['day', 'count', 'avg_rating', 'avg_weighted_rating', 'avg_popularity']
    day_stats['day_name'] = day_stats['day'].apply(lambda x: day_names[int(x)] if pd.notna(x) and 0 <= x <= 6 else '')
    day_stats = day_stats.round(2)
    
    result = day_stats.to_dict('records')
    return jsonify(result)


@app.route('/api/movies/top-rated', methods=['GET'])
def get_top_rated_movies():
    """Top phim có weighted rating cao nhất"""
    limit = request.args.get('limit', 20)
    
    if limit == 'all':
        query = df.nlargest(5000, 'weighted_rating')
    else:
        try:
            limit = int(limit)
        except:
            limit = 20
        query = df.nlargest(limit, 'weighted_rating')

    top_movies = query[
        ['id', 'title', 'genre', 'release_year', 'vote_average', 'vote_count', 
         'weighted_rating', 'overview', 'original_language', 'popularity']
    ]
    # Sanitise clean NaNs
    top_movies = top_movies.where(pd.notnull(top_movies), None)
    result = top_movies.to_dict('records')
    return jsonify(result)


@app.route('/api/movies/trending', methods=['GET'])
def get_trending_movies():
    """Top phim thịnh hành nhất (dựa trên popularity)"""
    limit = request.args.get('limit', 20)
    
    if limit == 'all':
        query = df.nlargest(5000, 'popularity')
    else:
        try:
            limit = int(limit)
        except:
            limit = 20
        query = df.nlargest(limit, 'popularity')

    trending_movies = query[
        ['id', 'title', 'genre', 'release_year', 'vote_average', 'vote_count', 
         'weighted_rating', 'overview', 'original_language', 'popularity']
    ]
    trending_movies = trending_movies.where(pd.notnull(trending_movies), None)
    result = trending_movies.to_dict('records')
    return jsonify(result)


@app.route('/api/movies/by-genre/<genre>', methods=['GET'])
def get_movies_by_genre(genre):
    """Lấy phim theo thể loại"""
    # Robust column finding (case-insensitive attempt)
    candidates = [f'genre_{genre.lower()}', f'genre_{genre.title()}', f'genre_{genre}']
    genre_col = next((c for c in candidates if c in df.columns), None)
    
    if not genre_col:
        # Fallback: scan all columns
        for col in df.columns:
            if col.lower() == f'genre_{genre.lower()}':
                genre_col = col
                break
    
    if not genre_col:
        return jsonify([]), 404
    
    limit = request.args.get('limit', 20)
    
    if limit == 'all':
        query = df[df[genre_col] == 1].sort_values('weighted_rating', ascending=False).head(5000)
    else:
        try:
            limit = int(limit)
        except:
            limit = 20
        query = df[df[genre_col] == 1].nlargest(limit, 'weighted_rating')
    
    genre_movies = query[
        ['id', 'title', 'genre', 'release_year', 'vote_average', 'vote_count', 
         'weighted_rating', 'overview', 'original_language', 'popularity']
    ]
    genre_movies = genre_movies.where(pd.notnull(genre_movies), None)
    result = genre_movies.to_dict('records')
    return jsonify(result)


@app.route('/api/movies/by-language/<language>', methods=['GET'])
def get_movies_by_language(language):
    """Lấy phim theo ngôn ngữ"""
    limit = request.args.get('limit', 20)
    
    filtered_df = df[df['original_language'] == language]
    
    if limit == 'all':
        query = filtered_df.sort_values('weighted_rating', ascending=False).head(5000)
    else:
        try:
            limit = int(limit)
        except:
            limit = 20
        query = filtered_df.nlargest(limit, 'weighted_rating')

    lang_movies = query[
        ['id', 'title', 'genre', 'release_year', 'vote_average', 'vote_count', 
         'weighted_rating', 'overview', 'original_language', 'popularity']
    ]
    lang_movies = lang_movies.where(pd.notnull(lang_movies), None)
    result = lang_movies.to_dict('records')
    return jsonify(result)


@app.route('/api/movies/genres', methods=['GET'])
def get_genres():
    """Lấy danh sách thể loại có sẵn"""
    genres = []
    for col in GENRE_COLUMNS:
        genre_code = col.replace('genre_', '')
        genre_name_vi = get_genre_name_vi(genre_code)
        count = int(df[col].sum())
        if count > 0:
            genres.append({'name': genre_name_vi, 'code': genre_code, 'count': count})
    genres.sort(key=lambda x: x['count'], reverse=True)
    return jsonify(genres)


@app.route('/api/movies/languages', methods=['GET'])
def get_languages():
    """Lấy danh sách ngôn ngữ có sẵn"""
    lang_counts = df['original_language'].value_counts()
    
    # Hợp nhất 'cn' và 'zh'
    if 'cn' in lang_counts.index and 'zh' in lang_counts.index:
        lang_counts['zh'] += lang_counts['cn']
        lang_counts = lang_counts.drop('cn')
    elif 'cn' in lang_counts.index:
        lang_counts = lang_counts.rename(index={'cn': 'zh'})
    
    languages = []
    for lang, count in lang_counts.items():
        if count >= 50:  # Chỉ lấy ngôn ngữ có ít nhất 50 phim
            languages.append({
                'code': lang,
                'name': get_language_name_vi(lang),
                'count': int(count)
            })
    
    languages.sort(key=lambda x: x['count'], reverse=True)
    return jsonify(languages)


@app.route('/api/movies/<int:movie_id>', methods=['GET'])
def get_movie_by_id(movie_id):
    """Lấy thông tin chi tiết phim theo ID"""
    movie = df[df['id'] == movie_id]
    if movie.empty:
        return jsonify({'error': 'Movie not found'}), 404
    
    movie_data = movie.iloc[0]
    
    # Lấy danh sách thể loại và chuyển sang tiếng Việt
    genre_list = []
    if pd.notna(movie_data.get('genre_list')):
        try:
            val = movie_data['genre_list']
            if isinstance(val, str):
                # Xử lý chuỗi dạng "[Drama, Action]" hoặc "['Drama', 'Action']"
                # Loại bỏ ngoặc vuông và tách bằng dấu phẩy
                clean_str = val.strip("[]")
                genre_list_raw = [x.strip().strip("'\"") for x in clean_str.split(',') if x.strip()]
            else:
                genre_list_raw = val
            
            # Chuyển đổi từng thể loại sang tiếng Việt
            genre_list = [get_genre_name_vi(genre) for genre in genre_list_raw if genre]
        except Exception as e:
            print(f"Error parsing genre_list for current movie: {e}")
            genre_list = []
    
    result = {
        'id': int(movie_data['id']),
        'title': str(movie_data['title']),
        'genre': str(movie_data.get('genre', '')),
        'genre_list': genre_list,
        'release_year': int(movie_data['release_year']) if pd.notna(movie_data.get('release_year')) else None,
        'vote_average': float(movie_data['vote_average']) if pd.notna(movie_data.get('vote_average')) else 0,
        'vote_count': int(movie_data['vote_count']) if pd.notna(movie_data.get('vote_count')) else 0,
        'weighted_rating': float(movie_data['weighted_rating']) if pd.notna(movie_data.get('weighted_rating')) else 0,
        'overview': str(movie_data['overview']) if pd.notna(movie_data.get('overview')) else '',
        'original_language': str(movie_data['original_language']) if pd.notna(movie_data.get('original_language')) else '',
        'language_name': get_language_name_vi(str(movie_data.get('original_language', ''))),
        'popularity': float(movie_data['popularity']) if pd.notna(movie_data.get('popularity')) else 0
    }
    
    return jsonify(result)






# ==========================================
# AUTHENTICATION & USER ROUTES
# ==========================================

@app.route('/api/auth/register', methods=['POST'])
def register():
    from flask import request
    data = request.get_json()

    required_fields = ['username', 'password', 'confirm_password', 'fullName', 'email']
    for field in required_fields:
        if field not in data or not data[field]:
            return jsonify({'error': f'Missing field: {field}'}), 400

    if data['password'] != data['confirm_password']:
        return jsonify({'error': 'Passwords do not match'}), 400

    if len(data['password']) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400

    # Check existing user
    if users_collection.find_one({'$or': [{'username': data['username']}, {'email': data['email']}]}):
        return jsonify({'error': 'Username or Email already exists'}), 400

    # Hash password
    hashed_password = bcrypt.hashpw(data['password'].encode('utf-8'), bcrypt.gensalt())

    new_user = {
        'username': data['username'],
        'password': hashed_password,
        'fullName': data['fullName'],
        'email': data['email'],
        'dob': data.get('dob', ''),
        'address': data.get('address', ''),
        'favorites': []
    }

    users_collection.insert_one(new_user)
    
    return jsonify({'message': 'User registered successfully'}), 201


@app.route('/api/auth/login', methods=['POST'])
def login():
    from flask import request
    data = request.get_json()

    if not data or 'username' not in data or 'password' not in data:
        return jsonify({'error': 'Missing username or password'}), 400

    user = users_collection.find_one({'username': data['username']})
    
    if not user or not bcrypt.checkpw(data['password'].encode('utf-8'), user['password']):
        return jsonify({'error': 'Invalid credentials'}), 401

    access_token = create_access_token(identity=str(user['_id']))
    
    return jsonify({
        'message': 'Login successful',
        'token': access_token,
        'user': {
            'username': user['username'],
            'fullName': user['fullName']
        }
    }), 200


@app.route('/api/user/profile', methods=['GET'])
@jwt_required()
def get_user_profile():
    from bson.objectid import ObjectId
    user_id = get_jwt_identity()
    user = users_collection.find_one({'_id': ObjectId(user_id)})
    
    if not user:
        return jsonify({'error': 'User not found'}), 404

    # Get favorite movies details
    favorites_details = []
    if 'favorites' in user and user['favorites']:
        fav_ids = [int(f) for f in user['favorites']]
        
        # 1. Fetch from Cache (Mongo)
        # This ensures we get poster_path and latest data
        cached_cursor = movies_collection.find({'_id': {'$in': fav_ids}})
        cached_map = {doc['_id']: doc for doc in cached_cursor}
        
        for mid in fav_ids:
            movie_data = {}
            
            if mid in cached_map:
                # Use Cached Data
                doc = cached_map[mid]
                movie_data = {
                    'id': doc['id'],
                    'title': doc['title'],
                    'poster_path': doc.get('poster_path'),
                    'release_year': doc.get('release_year'),
                    'vote_average': doc.get('vote_average'),
                    'genre': doc.get('genre'),
                    'overview': doc.get('overview')
                }
            else:
                # Fallback to CSV (No image)
                row = df[df['id'] == mid]
                if not row.empty:
                    r = row.iloc[0]
                    movie_data = {
                        'id': int(r['id']),
                        'title': str(r['title']),
                        'poster_path': None,
                        'release_year': int(r['release_year']) if pd.notna(r['release_year']) else None,
                        'vote_average': float(r['vote_average']) if pd.notna(r['vote_average']) else 0,
                        'genre': str(r.get('genre', '')),
                        'overview': str(r.get('overview', ''))
                    }
            
            if movie_data:
                favorites_details.append(movie_data)
            
    return jsonify({
        'username': user['username'],
        'fullName': user['fullName'],
        'email': user['email'],
        'dob': user.get('dob', ''),
        'address': user.get('address', ''),
        'favorites': favorites_details
    }), 200


@app.route('/api/user/favorites/ids', methods=['GET'])
@jwt_required()
def get_user_favorite_ids():
    from bson.objectid import ObjectId
    user_id = get_jwt_identity()
    user = users_collection.find_one({'_id': ObjectId(user_id)})
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    # Return list of IDs (ints)
    fav_ids = [int(f) for f in user.get('favorites', [])]
    return jsonify({'favorites': fav_ids}), 200


@app.route('/api/user/favorites', methods=['POST'])
@jwt_required()
def toggle_favorite():
    from flask import request
    from bson.objectid import ObjectId
    
    data = request.get_json()
    movie_id = data.get('movieId')
    
    if not movie_id:
        return jsonify({'error': 'Missing movieId'}), 400

    # Ensure movie_id is int
    try:
        movie_id = int(movie_id)
    except:
        return jsonify({'error': 'Invalid movieId'}), 400

    user_id = get_jwt_identity()
    user = users_collection.find_one({'_id': ObjectId(user_id)})
    
    if not user:
        return jsonify({'error': 'User not found'}), 404

    current_favorites = user.get('favorites', [])
    
    if movie_id in current_favorites:
        users_collection.update_one(
            {'_id': ObjectId(user_id)},
            {'$pull': {'favorites': movie_id}}
        )
        is_favorite = False
    else:
        users_collection.update_one(
            {'_id': ObjectId(user_id)},
            {'$addToSet': {'favorites': movie_id}}
        )
        is_favorite = True
        
    return jsonify({
        'success': True, 
        'isFavorite': is_favorite
    }), 200


@app.route('/api/user/favorites/check/<int:movie_id>', methods=['GET'])
@jwt_required()
def check_favorite(movie_id):
    from bson.objectid import ObjectId
    user_id = get_jwt_identity()
    user = users_collection.find_one({'_id': ObjectId(user_id)})
    
    if not user:
        return jsonify({'isFavorite': False}), 200
        
    # Ensure favorite list has ints
    favorites = [int(x) for x in user.get('favorites', [])]
    is_favorite = movie_id in favorites
    return jsonify({'isFavorite': is_favorite}), 200


if __name__ == '__main__':
    print("\n" + "🎬" * 20)
    print("  Movie Recommendation API")
    print("  Sử dụng dữ liệu đã tiền xử lý")
    print("🎬" * 20)
    print(f"\n📊 Loaded {len(df)} movies from processed_dataset.csv")
    print(f"📁 Columns: {len(df.columns)}")
    print(f"🎭 Genre columns: {len(GENRE_COLUMNS)}")
    print(f"\n🚀 Starting server at http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(debug=True, port=5000)
