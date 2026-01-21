import argparse
import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import kendalltau
import sqlite3
from tqdm import tqdm
import os

def win_rate_matrix(abilities, model_names):
    M = len(abilities)
    win_rates = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            win_rates[i, j] = 1 / (1 + math.exp(-(abilities[i] - abilities[j])))

    win_rates = pd.DataFrame(win_rates, index=model_names, columns=model_names)
    return win_rates

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

    init_params = np.zeros(num_models-1)
    res = minimize(bt_neg_log_likelihood, init_params, args=(bt_df,), method='BFGS')

    fitted_abilities = np.append(res.x, 0.0)

    bt_probs = np.zeros((num_models, num_models))
    for i in range(num_models):
        for j in range(num_models):
            if i != j:
                bt_probs[i, j] = np.exp(fitted_abilities[i]) / (np.exp(fitted_abilities[i]) + np.exp(fitted_abilities[j]))
            else:
                bt_probs[i, j] = np.nan  

    bt_probs_df = pd.DataFrame(bt_probs, index=mod_names, columns=mod_names)

    return fitted_abilities, bt_probs_df

def generate_binomial_matrix(bt_probs, avg_num_votes, model_names):
    N = len(bt_probs)
    binom_matrix = np.nan * np.ones((N, N))
    for i in range(N):
        for j in range(i):
            if i != j:
                binom_matrix[i, j] = np.random.binomial(n=avg_num_votes, p=bt_probs.iloc[i, j]) / avg_num_votes
                binom_matrix[j, i] = 1 - binom_matrix[i, j]

    return pd.DataFrame(binom_matrix, index=model_names, columns=model_names)

def generate_binomial_matrix_w_clone(bt_probs, to_clone, avg_num_votes, model_names, num_clones):
    num_models = len(bt_probs)
    binom_matrix = np.nan * np.ones((num_models+num_clones, num_models+num_clones))
    for i in range(num_models):
        for j in range(i):
            if i != j:
                binom_matrix[i, j] = np.random.binomial(n=avg_num_votes, p=bt_probs.iloc[i, j]) / avg_num_votes
                binom_matrix[j, i] = 1 - binom_matrix[i, j]
    for i in range(num_models):
        for j in range(num_clones):
            if i != to_clone:
                binom_matrix[i, num_models+j] = np.random.binomial(n=avg_num_votes, p=bt_probs.iloc[i, to_clone]) / avg_num_votes
                binom_matrix[num_models+j, i] = 1 - binom_matrix[i, num_models+j]
            else:
                binom_matrix[i, num_models+j] = np.random.binomial(n=avg_num_votes, p=0.5) / avg_num_votes
                binom_matrix[num_models+j, i] = 1 - binom_matrix[i, num_models+j]
    for i in range(num_clones):
        for j in range(i):
            binom_matrix[num_models+i, num_models+j] = np.random.binomial(n=avg_num_votes, p=0.5) / avg_num_votes
            binom_matrix[num_models+j, num_models+i] = 1 - binom_matrix[num_models+i, num_models+j]

    return pd.DataFrame(binom_matrix, 
        index=model_names + [f"{model_names[to_clone]}_clone_{i}" for i in range(num_clones)], 
        columns=model_names + [f"{model_names[to_clone]}_clone_{i}" for i in range(num_clones)])

def compute_yrwr_scores(estimated_abilities, to_clone, num_clones):
    corrected_estimated_abilities = np.array(estimated_abilities, copy=True)
    for i in range(num_clones):
        corrected_estimated_abilities[M+i] = min(estimated_abilities.iloc[to_clone], np.min(estimated_abilities.iloc[M:M+i+1]))
    return corrected_estimated_abilities


def compute_borda(binom_matrix_df_w_clone):
    return np.mean(binom_matrix_df_w_clone, axis=1)

def compute_rank(estimated_abilities, idx):
    return np.where(np.argsort(estimated_abilities)[::-1] == idx)[0][0]

def simulate_clone(to_clone, B, num_clones, compute_without_clone=True, use_borda=True, yrwr=False, org_to_models=None, model_to_org=None):
    avg_votes_mat_w_clone = pd.DataFrame(np.full((M+num_clones, M+num_clones), avg_num_votes), 
        index=model_names + [f"{model_names[to_clone]}_clone_{i}" for i in range(num_clones)], 
        columns=model_names + [f"{model_names[to_clone]}_clone_{i}" for i in range(num_clones)])
    rank_w_clone = []
    rank_wo_clone = []
    yrwr_rank_w_clone = []
    yrwr_rank_wo_clone = []
    kt_distance_w_clone = []
    kt_distance_yrwr = []
    kt_dist_bw_mechanisms = []
    order_violations = []
    for _ in range(B):
        binom_matrix_df = generate_binomial_matrix(true_bt_probs, avg_num_votes, model_names)
        binom_matrix_df_w_clone = generate_binomial_matrix_w_clone(true_bt_probs, to_clone, avg_num_votes, model_names, num_clones)

        if use_borda:
            estimated_abilities_with_clone = compute_borda(binom_matrix_df_w_clone)
        else:
            estimated_abilities_with_clone = fit_bt_model(binom_matrix_df_w_clone, avg_votes_mat_w_clone)[0]
        rank_original_with_clone = compute_rank(estimated_abilities_with_clone, to_clone)
        rank_clones = [compute_rank(estimated_abilities_with_clone, M+i) for i in range(num_clones)]
        rank_w_clone.append(min(rank_original_with_clone, min(rank_clones)))
        if yrwr:
            rank_tmp = compute_rank(estimated_abilities_with_clone[:M], to_clone)
            clone_org = model_to_org[model_names[to_clone]]
            models_ahead = []
            for m in org_to_models[clone_org]:
                models_ahead.append(m)
                if m == to_clone:
                    break

            models_ahead_ranks = [compute_rank(estimated_abilities_with_clone[:M], m) for m in models_ahead]
            if rank_tmp < max(models_ahead_ranks):
                order_violations.append((models_ahead[np.argmax(models_ahead_ranks)], to_clone))
            yrwr_rank_w_clone.append(max(rank_tmp, max(models_ahead_ranks)))
        true_rank = np.concatenate([true_abilities, np.array([true_abilities.iloc[to_clone]] * num_clones)])
        kt = kendalltau(estimated_abilities_with_clone, true_rank)[0]
        kt_distance_w_clone.append(kt)
        yrwr_scores = compute_yrwr_scores(estimated_abilities_with_clone, to_clone, num_clones)
        kt_yrwr = kendalltau(yrwr_scores, true_rank)[0]
        kt_distance_yrwr.append(kt_yrwr)

        clone_inds = [to_clone] + [M+i for i in range(num_clones)] 
        total_distance = kendalltau(estimated_abilities_with_clone, yrwr_scores)[0] * (M + num_clones) * (M + num_clones - 1) / 4
        clone_distance = kendalltau(estimated_abilities_with_clone[clone_inds], yrwr_scores[clone_inds])[0] * (num_clones + 1) * num_clones / 4
        kt_dist_bw_mechanisms.append(total_distance - clone_distance)
        
        if compute_without_clone:
            if use_borda:
                estimated_abilities = compute_borda(binom_matrix_df)
            else:
                estimated_abilities = fit_bt_model(binom_matrix_df, avg_votes_mat)[0]
            rank_original = compute_rank(estimated_abilities, to_clone)
            rank_wo_clone.append(rank_original)
            if yrwr:
                yrwr_rank_wo_clone.append(rank_original)

    result_dict = {
        "rank_w_clone": np.array(rank_w_clone),
    }
    if yrwr:
        result_dict["yrwr_rank_w_clone"] = np.array(yrwr_rank_w_clone)
        result_dict["kt_distance_w_clone"] = np.array(kt_distance_w_clone)
        result_dict["kt_distance_yrwr"] = np.array(kt_distance_yrwr)
        result_dict["order_violations"] = np.array(order_violations)
        result_dict["kt_dist_bw_mechanisms"] = np.array(kt_dist_bw_mechanisms)
    if compute_without_clone:
        result_dict["rank_wo_clone"] = np.array(rank_wo_clone)
        if yrwr:
            result_dict["yrwr_rank_wo_clone"] = np.array(yrwr_rank_wo_clone)
    return result_dict

def init_db(db_path):
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")   # better concurrency
        conn.execute("PRAGMA synchronous=NORMAL;")


def append_df(df, table, db_path):
    with sqlite3.connect(db_path) as conn:
        df.to_sql(table, conn, if_exists="append", index=False, method="multi")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=1, help="The number of simulations to run")
    parser.add_argument("--job_id", type=int, required=True, help="The job ID")
    parser.add_argument("--max_num_clones", type=int, default=1, help="The maximum number of clones to simulate")
    parser.add_argument("--keep_existing_results", action="store_true", help="Whether to keep existing results")
    args = parser.parse_args()
    B = args.B
    max_num_clones = args.max_num_clones
    keep_existing_results = args.keep_existing_results
    DB_PATH = f"results/job_{args.job_id}.sqlite"

    if not keep_existing_results:
        try:
            os.remove(DB_PATH)
        except FileNotFoundError:
            pass

    init_db(DB_PATH)
    TYPES = ["text", "vision", "webdev", "texttoimage", "imageedit", "imagetovideo"]
    text_types = ["coding","expert", "hard_prompts", "instruction_following", "longer_query", "multiturn"]
    text_types = ["text_" + t for t in text_types]
    TYPES = TYPES + text_types

    # TYPES = ["imageedit", "imagetovideo"]

    for type in TYPES:
        print(f"Processing {type}")
        type_no_dashes = type.replace("-", "")

        df = pd.read_csv(f"parsed_tables/{type}.csv", index_col="Model")

        alpha = 400 / math.log(10)
        true_abilities = (df['Score'] - 1000) / alpha # convert to unscaled abilities
        M = len(true_abilities)
        model_names = list(true_abilities.index)
        true_bt_probs = win_rate_matrix(true_abilities.values, model_names)

        org_to_models = (
            df.reset_index()
              .sort_values('Score', ascending=False)
              .groupby('Organization')['Model']
              .apply(lambda models: [model_names.index(m) for m in models])
              .to_dict()
        )
        model_to_org = df.reset_index().set_index("Model")["Organization"].to_dict()

        avg_num_votes = df['Votes'].values.sum() / (M * (M-1) / 2)
        avg_votes_mat = pd.DataFrame(np.full((M, M), avg_num_votes), index=model_names, columns=model_names)

        rank_w_clone = np.nan * np.zeros((B, M, max_num_clones))
        rank_wo_clone = np.nan * np.zeros((B, M))
        yrwr_rank_w_clone = np.nan * np.zeros((B, M, max_num_clones))
        yrwr_rank_wo_clone = np.nan * np.zeros((B, M))
        kt_distance_w_clone = np.nan * np.zeros((B, M, max_num_clones))
        kt_distance_yrwr = np.nan * np.zeros((B, M, max_num_clones))
        kt_dist_bw_mechanisms = np.nan * np.zeros((B, M, max_num_clones))
        order_violations = []
        for to_clone in tqdm(range(M)):
            for num_clones in range(1, max_num_clones+1):
                result_dict = simulate_clone(to_clone, B, num_clones, compute_without_clone=True, yrwr=True, org_to_models=org_to_models, model_to_org=model_to_org)
                rank_w_clone[:,to_clone,num_clones-1] = result_dict["rank_w_clone"]
                yrwr_rank_w_clone[:,to_clone,num_clones-1] = result_dict["yrwr_rank_w_clone"]
                kt_distance_w_clone[:,to_clone,num_clones-1] = result_dict["kt_distance_w_clone"]
                kt_distance_yrwr[:,to_clone,num_clones-1] = result_dict["kt_distance_yrwr"]
                kt_dist_bw_mechanisms[:,to_clone,num_clones-1] = result_dict["kt_dist_bw_mechanisms"]
                order_violations.extend(result_dict["order_violations"])
                if num_clones == 1:
                    rank_wo_clone[:,to_clone] = result_dict["rank_wo_clone"]
                    yrwr_rank_wo_clone[:,to_clone] = result_dict["yrwr_rank_wo_clone"]

        TABLE_WITH_CLONE = f"{type_no_dashes}_with_clone"
        TABLE_WITHOUT_CLONE = f"{type_no_dashes}_without_clone"
        TABLE_YRWR_WITH_CLONE = f"{type_no_dashes}_yrwr_with_clone"
        TABLE_YRWR_WITHOUT_CLONE = f"{type_no_dashes}_yrwr_without_clone"
        TABLE_KT_DISTANCE_WITH_CLONE = f"{type_no_dashes}_kt_distance_with_clone"
        TABLE_KT_DISTANCE_YRWR_WITH_CLONE = f"{type_no_dashes}_kt_distance_yrwr_with_clone"
        TABLE_KT_DISTANCE_BW_MECHANISMS = f"{type_no_dashes}_kt_distance_bw_mechanisms"
        TABLE_ORDER_VIOLATIONS = f"{type_no_dashes}_order_violations"

        df_wo_clone = pd.DataFrame(rank_wo_clone, columns=model_names)
        append_df(df_wo_clone, TABLE_WITHOUT_CLONE, DB_PATH)
        df_yrwr_wo_clone = pd.DataFrame(yrwr_rank_wo_clone, columns=model_names)
        append_df(df_yrwr_wo_clone, TABLE_YRWR_WITHOUT_CLONE, DB_PATH)
        df_order_violations = pd.DataFrame(order_violations, columns=["model_ahead", "model_to_clone"])
        append_df(df_order_violations, TABLE_ORDER_VIOLATIONS, DB_PATH)
        for num_clones in range(1, max_num_clones+1):
            df_w_clone = pd.DataFrame(rank_w_clone[:,:,num_clones-1], columns=model_names)
            df_w_clone["num_clones"] = num_clones
            append_df(df_w_clone, TABLE_WITH_CLONE, DB_PATH)
            df_yrwr_w_clone = pd.DataFrame(yrwr_rank_w_clone[:,:,num_clones-1], columns=model_names)
            df_yrwr_w_clone["num_clones"] = num_clones
            append_df(df_yrwr_w_clone, TABLE_YRWR_WITH_CLONE, DB_PATH)
            df_kt_distance_w_clone = pd.DataFrame(kt_distance_w_clone[:,:,num_clones-1], columns=model_names)
            df_kt_distance_w_clone["num_clones"] = num_clones
            append_df(df_kt_distance_w_clone, TABLE_KT_DISTANCE_WITH_CLONE, DB_PATH)
            df_kt_distance_yrwr = pd.DataFrame(kt_distance_yrwr[:,:,num_clones-1], columns=model_names)
            df_kt_distance_yrwr["num_clones"] = num_clones
            append_df(df_kt_distance_yrwr, TABLE_KT_DISTANCE_YRWR_WITH_CLONE, DB_PATH)
            df_kt_dist_bw_mechanisms = pd.DataFrame(kt_dist_bw_mechanisms[:,:,num_clones-1], columns=model_names)
            df_kt_dist_bw_mechanisms["num_clones"] = num_clones
            append_df(df_kt_dist_bw_mechanisms, TABLE_KT_DISTANCE_BW_MECHANISMS, DB_PATH)