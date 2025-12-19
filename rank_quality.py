import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import sqlite3
from tqdm import tqdm

DB_PATH = f"results.sqlite"

def bt_neg_log_likelihood(params, bt_df):
    abilities = np.append(params, 0.0)
    ll = 0.0
    for row in bt_df.itertuples():
        i = row.winner
        j = row.loser
        num = np.exp(abilities[i])
        denom = np.exp(abilities[i]) + np.exp(abilities[j])
        count = row.count
        ll += count * np.log(num / denom)
    return -ll

def fit_bt_model(df_win_rates, df_battle_counts):
    num_models = len(df_win_rates)
    mod_names = list(df_win_rates.index)
    wins = df_win_rates.values * df_battle_counts.values
    total_games = wins + wins.T

    pairs = [(i, j) for i in range(num_models) for j in range(num_models) if i != j and total_games[i, j] > 0]

    data = []
    for i, j in pairs:
        if wins[i, j] > 0:
            data.append({'winner': i, 'loser': j, 'count': wins[i, j]})
        if wins[j, i] > 0:
            data.append({'winner': j, 'loser': i, 'count': wins[j, i]})
    bt_df = pd.DataFrame(data)

    # Fit the model (N-1 parameters; last one fixed to zero)
    init_params = np.zeros(num_models-1)
    res = minimize(bt_neg_log_likelihood, init_params, args=(bt_df,), method='BFGS')

    # Get fitted abilities (last one is zero for identifiability)
    fitted_abilities = np.append(res.x, 0.0)

    # Compute fitted BT probabilites: P(i beats j) for all i, j
    bt_probs = np.zeros((num_models, num_models))
    for i in range(num_models):
        for j in range(num_models):
            if i != j:
                bt_probs[i, j] = np.exp(fitted_abilities[i]) / (np.exp(fitted_abilities[i]) + np.exp(fitted_abilities[j]))
            else:
                bt_probs[i, j] = np.nan  # undefined

    bt_probs_df = pd.DataFrame(bt_probs, index=mod_names, columns=mod_names)

    return fitted_abilities, bt_probs_df

def generate_binomial_matrix(bt_probs, avg_num_votes, model_names):
    N = len(bt_probs)
    binom_matrix = np.empty((N, N))
    for i in range(N):
        for j in range(i):
            if i != j:
                binom_matrix[i, j] = np.random.binomial(n=avg_num_votes, p=bt_probs.iloc[i, j]) / avg_num_votes
                binom_matrix[j, i] = 1 - binom_matrix[i, j]

    return pd.DataFrame(binom_matrix, index=model_names, columns=model_names)

def generate_binomial_matrix_w_clone(bt_probs, to_clone, avg_num_votes, model_names):
    num_models = len(bt_probs)
    binom_matrix = np.empty((num_models+1, num_models+1))
    for i in range(num_models):
        for j in range(i):
            if i != j:
                binom_matrix[i, j] = np.random.binomial(n=avg_num_votes, p=bt_probs.iloc[i, j]) / avg_num_votes
                binom_matrix[j, i] = 1 - binom_matrix[i, j]
    for i in range(num_models):
        if i != to_clone:
            binom_matrix[i, num_models] = np.random.binomial(n=avg_num_votes, p=bt_probs.iloc[i, to_clone]) / avg_num_votes
            binom_matrix[num_models, i] = 1 - binom_matrix[i, num_models]
        else:
            binom_matrix[i, num_models] = np.random.binomial(n=avg_num_votes, p=0.5) / avg_num_votes
            binom_matrix[num_models, i] = 1 - binom_matrix[i, num_models]
    binom_matrix[num_models, num_models] = np.nan

    return pd.DataFrame(binom_matrix, index=model_names + [f"{model_names[to_clone]}_clone"], columns=model_names + [f"{model_names[to_clone]}_clone"])

def compute_rank(estimated_abilities, idx):
    return np.argsort(estimated_abilities)[::-1][idx]

def simulate_clone(to_clone, B):
    avg_votes_mat_w_clone = pd.DataFrame(np.full((M+1, M+1), avg_num_votes), index=model_names + [f"{model_names[to_clone]}_clone"], columns=model_names + [f"{model_names[to_clone]}_clone"])

    rank_w_clone = []
    rank_wo_clone = []

    for _ in tqdm(range(B)):
        binom_matrix_df = generate_binomial_matrix(true_bt_probs, avg_num_votes, model_names)
        binom_matrix_df_w_clone = generate_binomial_matrix_w_clone(true_bt_probs, to_clone, avg_num_votes, model_names)

        estimated_abilities_with_clone = fit_bt_model(binom_matrix_df_w_clone, avg_votes_mat_w_clone)[0]
        estimated_abilities = fit_bt_model(binom_matrix_df, avg_votes_mat)[0]

        rank_original_with_clone = compute_rank(estimated_abilities_with_clone, to_clone)
        rank_clone = compute_rank(estimated_abilities_with_clone, M)

        rank_w_clone.append(min(rank_original_with_clone, rank_clone))
        rank_wo_clone.append(rank_original_with_clone)

    return np.array(rank_w_clone), np.array(rank_wo_clone)

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")   # better concurrency
        conn.execute("PRAGMA synchronous=NORMAL;")

def append_df(df, table):
    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql(table, conn, if_exists="append", index=False, method="multi")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, required=True, help="The type of data to use")
    parser.add_argument("--B", type=int, default=1, help="The number of simulations to run")
    args = parser.parse_args()
    type = args.type
    B = args.B

    df_win_rates = pd.read_csv(f"parsed_heatmaps/{type}_win_rates.csv", index_col=0)
    df_battle_counts = pd.read_csv(f"parsed_heatmaps/{type}_battle_counts.csv", index_col=0)

    M = len(df_win_rates)
    model_names = list(df_win_rates.index)

    avg_num_votes = int(df_battle_counts.values.mean())
    avg_votes_mat = pd.DataFrame(np.full((M, M), avg_num_votes), index=model_names, columns=model_names)

    true_abilities, true_bt_probs = fit_bt_model(df_win_rates, df_battle_counts)

    rank_w_clone = np.zeros((B, M))
    rank_wo_clone = np.zeros((B, M))

    true_abilities, true_bt_probs = fit_bt_model(df_win_rates, df_battle_counts)
    init_db()

    for to_clone in range(M):
        print(f"Simulating clone {to_clone}")
        rank_w_clone[:,to_clone], rank_wo_clone[:,to_clone] = simulate_clone(to_clone, B)


    TABLE_WITH_CLONE = f"{type}_with_clone"
    TABLE_WITHOUT_CLONE = f"{type}_without_clone"

    append_df(pd.DataFrame(rank_w_clone, columns=model_names), TABLE_WITH_CLONE)
    append_df(pd.DataFrame(rank_wo_clone, columns=model_names), TABLE_WITHOUT_CLONE)