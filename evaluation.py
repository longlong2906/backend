import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_model(df, cosine_sim, indices, sample_size=1000):
    """
    Evaluate the Content-Based Recommendation Model.
    
    Metrics:
    1. RMSE & MAE: 
       - Predict movie rating based on weighted average of k-NN similar movies.
       - Compare with actual vote_average.
    2. Precision@K & Recall@K:
       - Ground Truth: Movies that share at least one genre.
    """
    print("Starting Model Evaluation...")
    
    # 1. Prepare Sample
    if len(df) > sample_size:
        sample_indices = np.random.choice(len(df), sample_size, replace=False)
    else:
        sample_indices = np.arange(len(df))
        
    actual_ratings = []
    predicted_ratings = []
    
    k_values = [5, 10, 20]
    precision_scores = {k: [] for k in k_values}
    recall_scores = {k: [] for k in k_values}
    
    for idx in sample_indices:
        # Get actual rating
        actual_rating = df.iloc[idx]['vote_average']
        if pd.isna(actual_rating):
            continue
            
        # Get Similar Movies
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Take Top 20 for calculation (exclude self)
        top_20 = sim_scores[1:21]
        
        # --- RMSE/MAE Calculation ---
        # Predict rating = Weighted average of neighbors
        weighted_sum = 0
        similarity_sum = 0
        
        for i, score in top_20:
            neighbor_rating = df.iloc[i]['vote_average']
            if pd.notna(neighbor_rating):
                weighted_sum += score * neighbor_rating
                similarity_sum += score
                
        if similarity_sum > 0:
            pred_rating = weighted_sum / similarity_sum
            actual_ratings.append(actual_rating)
            predicted_ratings.append(pred_rating)
            
        # --- Precision/Recall Calculation ---
        source_genres = set(df.iloc[idx]['genre_list'])
        
        for k in k_values:
            top_k = sim_scores[1:k+1]
            relevant_items = 0
            
            for i, score in top_k:
                target_genres = set(df.iloc[i]['genre_list'])
                # Relevance: Share at least 1 genre
                if not source_genres.isdisjoint(target_genres):
                    relevant_items += 1
            
            precision = relevant_items / k
            precision_scores[k].append(precision)
            
            # Recall is harder because we don't know total relevant items in entire DB easily
            # We approximate Recall relative to the retrieved set or assume a fixed pool usually
            # But for CBF evaluation standard, we can use Recall@K = (Relevant in TopK) / (Total Relevant in Top N or Ground Truth)
            # Here let's define simplified Recall: Relevant items found / K (which is same as precision?) NO.
            # Standard Recall@K = (Relevant in K) / (Total Relevant in Dataset)
            # Calculating Total Relevant in Dataset is expensive (checking all 45k movies).
            # Optimization: Estimate 'Total Relevant' as 'all movies with shared genre'. Too big.
            # Alternative: Use a smaller pool or skip Recall if too slow.
            # Let's use a proxy: Assume 'Total Relevant' is the count of movies with SAME PRIMARY genre in the Top 100 similar?
            # Or just strictly compute:
            # Let's compute Precision only for speed? Or approximate Recall using Top 100 as "Ground Truth Pool"
            
            # Approximation: Ground Truth = Relevant items in Top 100 most similar (assume algorithm is somewhat good)
            # This measures "Ranking Quality" within the localized neighborhood.
            top_100 = sim_scores[1:101]
            total_relevant_in_pool = 0
            for i, score in top_100:
                target_genres = set(df.iloc[i]['genre_list'])
                if not source_genres.isdisjoint(target_genres):
                    total_relevant_in_pool += 1
            
            if total_relevant_in_pool > 0:
                recall = relevant_items / total_relevant_in_pool
            else:
                recall = 0
            recall_scores[k].append(recall)

    # Compute Final Metrics
    rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
    mae = mean_absolute_error(actual_ratings, predicted_ratings)
    
    avg_precision = {k: np.mean(v) for k, v in precision_scores.items()}
    avg_recall = {k: np.mean(v) for k, v in recall_scores.items()}
    
    results = {
        'rmse': round(rmse, 4),
        'mae': round(mae, 4),
        'precision': avg_precision,
        'recall': avg_recall
    }
    
    print("Evaluation Complete:", results)
    return results
