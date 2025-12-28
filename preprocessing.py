"""
Tiền xử lý dữ liệu cho hệ thống gợi ý phim
Movie Recommendation System - Data Preprocessing
"""

import pandas as pd
import numpy as np
import re
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# Đường dẫn file
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'dataset.csv')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'data', 'processed_dataset.csv')


def load_data(filepath: str) -> pd.DataFrame:
    """Đọc dữ liệu từ file CSV"""
    print("=" * 60)
    print("1. ĐỌC DỮ LIỆU")
    print("=" * 60)
    
    df = pd.read_csv(filepath)
    print(f"✓ Đã đọc {len(df)} bản ghi")
    print(f"✓ Số cột: {len(df.columns)}")
    print(f"✓ Các cột: {list(df.columns)}")
    return df


def explore_data(df: pd.DataFrame) -> None:
    """Khám phá dữ liệu ban đầu"""
    print("\n" + "=" * 60)
    print("2. KHÁM PHÁ DỮ LIỆU")
    print("=" * 60)
    
    print("\n--- Thông tin tổng quan ---")
    print(df.info())
    
    print("\n--- Thống kê mô tả ---")
    print(df.describe())
    
    print("\n--- Kiểm tra giá trị null ---")
    null_counts = df.isnull().sum()
    print(null_counts)
    
    print("\n--- Kiểm tra giá trị trùng lặp ---")
    duplicates = df.duplicated().sum()
    print(f"Số bản ghi trùng lặp: {duplicates}")
    
    print("\n--- Phân bố ngôn ngữ ---")
    print(df['original_language'].value_counts().head(10))


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Xử lý giá trị null/missing"""
    print("\n" + "=" * 60)
    print("3. XỬ LÝ GIÁ TRỊ MISSING")
    print("=" * 60)
    
    initial_count = len(df)
    
    # Kiểm tra missing values trước khi xử lý
    print("\nMissing values trước khi xử lý:")
    print(df.isnull().sum())
    
    # Xử lý từng cột
    # 1. overview: điền bằng chuỗi rỗng nếu null
    df['overview'] = df['overview'].fillna('')
    
    # 2. genre: điền bằng 'Unknown' nếu null
    df['genre'] = df['genre'].fillna('Unknown')
    
    # 3. release_date: điền bằng giá trị phổ biến nhất hoặc xóa
    if df['release_date'].isnull().sum() > 0:
        df = df.dropna(subset=['release_date'])
    
    # 4. Các cột số: điền bằng giá trị trung bình
    numeric_cols = ['popularity', 'vote_average', 'vote_count']
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())
    
    # Kiểm tra missing values sau khi xử lý
    print("\nMissing values sau khi xử lý:")
    print(df.isnull().sum())
    
    final_count = len(df)
    print(f"\n✓ Đã xử lý missing values")
    print(f"✓ Số bản ghi ban đầu: {initial_count}")
    print(f"✓ Số bản ghi sau xử lý: {final_count}")
    print(f"✓ Số bản ghi bị xóa: {initial_count - final_count}")
    
    return df


def clean_text(text: str) -> str:
    """Làm sạch text: loại bỏ ký tự đặc biệt, chuyển thường"""
    if pd.isna(text) or text == '':
        return ''
    
    # Chuyển thành chữ thường
    text = text.lower()
    
    # Loại bỏ ký tự đặc biệt, giữ lại chữ cái, số và khoảng trắng
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Loại bỏ khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def process_genres(df: pd.DataFrame) -> pd.DataFrame:
    """Xử lý cột genre - tách thành list và tạo các cột one-hot"""
    print("\n" + "=" * 60)
    print("4. XỬ LÝ CỘT GENRE")
    print("=" * 60)
    
    # Tạo cột genre_list chứa list các thể loại
    df['genre_list'] = df['genre'].apply(lambda x: [g.strip() for g in str(x).split(',')])
    
    # Lấy tất cả các thể loại unique
    all_genres = set()
    for genres in df['genre_list']:
        all_genres.update(genres)
    
    all_genres = sorted(list(all_genres))
    print(f"✓ Tìm thấy {len(all_genres)} thể loại unique:")
    print(f"  {all_genres}")
    
    # Tạo các cột one-hot encoding cho từng thể loại
    for genre in all_genres:
        col_name = f'genre_{genre.lower().replace(" ", "_")}'
        df[col_name] = df['genre_list'].apply(lambda x: 1 if genre in x else 0)
    
    print(f"✓ Đã tạo {len(all_genres)} cột one-hot encoding cho genre")
    
    return df, all_genres


def process_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Xử lý cột overview - làm sạch text"""
    print("\n" + "=" * 60)
    print("5. XỬ LÝ CỘT OVERVIEW (TEXT)")
    print("=" * 60)
    
    # Làm sạch overview
    df['overview_clean'] = df['overview'].apply(clean_text)
    
    # Tạo cột độ dài overview (số từ)
    df['overview_length'] = df['overview_clean'].apply(lambda x: len(x.split()) if x else 0)
    
    print(f"✓ Đã làm sạch cột overview")
    print(f"✓ Đã tạo cột overview_clean")
    print(f"✓ Đã tạo cột overview_length")
    print(f"  - Độ dài trung bình: {df['overview_length'].mean():.2f} từ")
    print(f"  - Độ dài min: {df['overview_length'].min()} từ")
    print(f"  - Độ dài max: {df['overview_length'].max()} từ")
    
    return df


def process_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Xử lý cột release_date - trích xuất các đặc trưng thời gian"""
    print("\n" + "=" * 60)
    print("6. XỬ LÝ CỘT RELEASE_DATE")
    print("=" * 60)
    
    # Chuyển đổi sang datetime
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    
    # Trích xuất các đặc trưng
    df['release_year'] = df['release_date'].dt.year
    df['release_month'] = df['release_date'].dt.month
    df['release_day'] = df['release_date'].dt.day
    df['release_dayofweek'] = df['release_date'].dt.dayofweek  # 0=Monday, 6=Sunday
    
    # Tính tuổi của phim (số năm từ khi phát hành đến nay)
    current_year = pd.Timestamp.now().year
    df['movie_age'] = current_year - df['release_year']
    
    # Phân loại theo thập kỷ
    df['decade'] = (df['release_year'] // 10) * 10
    
    print(f"✓ Đã chuyển đổi release_date sang datetime")
    print(f"✓ Đã tạo các cột: release_year, release_month, release_day, release_dayofweek")
    print(f"✓ Đã tạo cột movie_age (tuổi phim)")
    print(f"✓ Đã tạo cột decade (thập kỷ)")
    print(f"\n  - Phạm vi năm: {df['release_year'].min()} - {df['release_year'].max()}")
    print(f"  - Tuổi phim trung bình: {df['movie_age'].mean():.1f} năm")
    
    return df


def normalize_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa các cột số về khoảng [0, 1]"""
    print("\n" + "=" * 60)
    print("7. CHUẨN HÓA CÁC CỘT SỐ")
    print("=" * 60)
    
    # Các cột cần chuẩn hóa
    numeric_cols = ['popularity', 'vote_average', 'vote_count', 'movie_age', 'overview_length']
    
    scaler = MinMaxScaler()
    
    for col in numeric_cols:
        if col in df.columns:
            # Tạo cột mới với suffix _normalized
            col_normalized = f'{col}_normalized'
            df[col_normalized] = scaler.fit_transform(df[[col]])
            print(f"✓ Đã chuẩn hóa cột {col} -> {col_normalized}")
    
    return df


def create_combined_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo các đặc trưng kết hợp cho recommendation"""
    print("\n" + "=" * 60)
    print("8. TẠO ĐẶC TRƯNG KẾT HỢP")
    print("=" * 60)
    
    # Tạo cột kết hợp title + genre + overview cho content-based filtering
    df['combined_features'] = (
        df['title'].fillna('') + ' ' + 
        df['genre'].fillna('').str.replace(',', ' ') + ' ' + 
        df['overview_clean'].fillna('')
    )
    
    # Tạo cột tags (genre + language)
    df['tags'] = df['genre'].fillna('') + ',' + df['original_language'].fillna('')
    
    # Tính điểm weighted rating (IMDB formula)
    # WR = (v/(v+m)) * R + (m/(v+m)) * C
    # v = vote_count, R = vote_average, m = minimum votes required, C = mean vote
    m = df['vote_count'].quantile(0.75)  # minimum votes = 75th percentile
    C = df['vote_average'].mean()  # mean vote across all movies
    
    df['weighted_rating'] = (
        (df['vote_count'] / (df['vote_count'] + m)) * df['vote_average'] + 
        (m / (df['vote_count'] + m)) * C
    )
    
    print(f"✓ Đã tạo cột combined_features")
    print(f"✓ Đã tạo cột tags")
    print(f"✓ Đã tạo cột weighted_rating (IMDB formula)")
    print(f"  - Minimum votes threshold (m): {m:.0f}")
    print(f"  - Mean vote (C): {C:.2f}")
    
    return df


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """Lưu dữ liệu đã xử lý"""
    print("\n" + "=" * 60)
    print("9. LƯU DỮ LIỆU ĐÃ XỬ LÝ")
    print("=" * 60)
    
    # Chuyển đổi release_date về string để lưu CSV
    df['release_date'] = df['release_date'].astype(str)
    
    # Chuyển đổi genre_list về string
    df['genre_list'] = df['genre_list'].apply(lambda x: ','.join(x) if isinstance(x, list) else x)
    
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✓ Đã lưu dữ liệu vào: {output_path}")
    print(f"✓ Số bản ghi: {len(df)}")
    print(f"✓ Số cột: {len(df.columns)}")


def print_summary(df: pd.DataFrame, all_genres: list) -> None:
    """In tổng kết quá trình tiền xử lý"""
    print("\n" + "=" * 60)
    print("📊 TỔNG KẾT TIỀN XỬ LÝ DỮ LIỆU")
    print("=" * 60)
    
    print(f"\n✅ Số bản ghi: {len(df)}")
    print(f"✅ Số cột ban đầu: 9")
    print(f"✅ Số cột sau xử lý: {len(df.columns)}")
    
    print(f"\n📁 Các cột mới được tạo:")
    new_cols = [
        'genre_list', 'overview_clean', 'overview_length',
        'release_year', 'release_month', 'release_day', 'release_dayofweek',
        'movie_age', 'decade', 'combined_features', 'tags', 'weighted_rating'
    ]
    
    # Cột one-hot encoding cho genre
    genre_cols = [col for col in df.columns if col.startswith('genre_')]
    
    # Cột normalized
    normalized_cols = [col for col in df.columns if col.endswith('_normalized')]
    
    print(f"  - Cột đặc trưng mới: {len(new_cols)}")
    print(f"  - Cột one-hot genre: {len(genre_cols)}")
    print(f"  - Cột đã chuẩn hóa: {len(normalized_cols)}")
    
    print(f"\n🎬 Thống kê dữ liệu:")
    print(f"  - Số thể loại phim: {len(all_genres)}")
    print(f"  - Phạm vi năm: {df['release_year'].min():.0f} - {df['release_year'].max():.0f}")
    print(f"  - Điểm trung bình: {df['vote_average'].mean():.2f}")
    print(f"  - Weighted rating trung bình: {df['weighted_rating'].mean():.2f}")
    
    print(f"\n📋 Danh sách tất cả các cột:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2}. {col}")


def main():
    """Hàm chính thực hiện toàn bộ pipeline tiền xử lý"""
    print("\n" + "🎬" * 30)
    print("  TIỀN XỬ LÝ DỮ LIỆU HỆ THỐNG GỢI Ý PHIM")
    print("🎬" * 30)
    
    # 1. Đọc dữ liệu
    df = load_data(DATA_PATH)
    
    # 2. Khám phá dữ liệu
    explore_data(df)
    
    # 3. Xử lý missing values
    df = handle_missing_values(df)
    
    # 4. Xử lý genre
    df, all_genres = process_genres(df)
    
    # 5. Xử lý text features
    df = process_text_features(df)
    
    # 6. Xử lý date features
    df = process_date_features(df)
    
    # 7. Chuẩn hóa các cột số
    df = normalize_numeric_features(df)
    
    # 8. Tạo đặc trưng kết hợp
    df = create_combined_features(df)
    
    # 9. Lưu dữ liệu
    save_processed_data(df, OUTPUT_PATH)
    
    # 10. Tổng kết
    print_summary(df, all_genres)
    
    print("\n" + "✅" * 30)
    print("  HOÀN THÀNH TIỀN XỬ LÝ DỮ LIỆU!")
    print("✅" * 30 + "\n")
    
    return df


if __name__ == "__main__":
    df = main()

