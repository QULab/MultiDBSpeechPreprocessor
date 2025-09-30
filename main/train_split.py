#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from datetime import datetime
from time import time
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpConstraint, LpStatusOptimal, COIN_CMD
from scipy.spatial.distance import jensenshannon


def bhattacharyya(a, b):
    return -np.log(np.sum(np.sqrt(a * b)))


do_plot = False

# fixed random seed
random_seed = 17
#random_seed = None
rng = np.random.default_rng(seed=random_seed)

###############################################################################
# prepare data
###############################################################################

db = pd.read_csv("db.csv")
db["global_pid"] = db["db"] + db["pid"].apply(lambda x: f"{x:04d}")
n_segments_total = len(db.index)


speakers = [p for p, _ in db.groupby("global_pid")]
n_speakers = len(speakers)
segments = db.groupby("global_pid").size()  # number of segments per speaker
db["count"] = db["global_pid"].apply(lambda x: segments[x])

possible_sexes = [s for s, _ in db.groupby("sex")]

possible_scores = [p for p, _ in db.groupby("phq-9")]
score_min = min(possible_scores)
score_max = max(possible_scores)
score_bin_edges = np.arange(score_min, score_max + 2)


scores = pd.Series([t[1] for t, _ in db.groupby(["global_pid", "phq-9"])], index=speakers)
sexes = pd.Series([t[1] for t, _ in db.groupby(["global_pid", "sex"])], index=speakers)

# speaker_info = pd.DataFrame({"score": scores, "segments": segments}, index=speakers)

# count number of segments that each speaker has for each possible PHQ-9 score
score_counts = db.groupby(["global_pid", "phq-9"]).size().unstack(fill_value=0)


# ## Optimization notes:
# - 85% train and 15% validation
#     - take 10% of segments (randomly) from each speaker for test
# - sexes as balanced as possible between train and validation
# - as equal distribution as possible of scores in train and validation
# - assignment to the train group or test group is done one the per-speaker level,
#     not the per-segment level
#     - this is indicated by the use of the binary decision variable


# define desired train set size (train_fraction +/- tolerance)
train_fraction = 0.85
tolerance = 0.05

# number of segments desired in training set
# (just a constraint, doesn't need to be an integer)
n_segments_train_desired = train_fraction * n_segments_total
n_segments_train_min = (train_fraction - tolerance) * n_segments_total
n_segments_train_max = (train_fraction + tolerance) * n_segments_total

n_segments_val_desired = (1 - train_fraction) * n_segments_total

# compute desired score counts in train set (balanced)
# PHQ-9 scores can range from 0 to 27 (inclusive). thus in the ideal case,
# there will be a perfectly uniform distribution of scores in the set.
n_segments_train_desired_ps = n_segments_train_desired / len(possible_scores)
n_segments_val_desired_ps = n_segments_val_desired / len(possible_scores)

###############################################################################
# solve distribution problem using integer linear programming
# train/validation split as an optimization problem
# goals: 
# - train/validation split of approximately 85/15
# - both sets have a balanced representation of PHQ-9 scores
# - both sets have a balanced representation of sexes
###############################################################################

# define the optimization problem
prob = LpProblem("train_validation_split", LpMinimize)

x = LpVariable.dicts("in_train_set", speakers, cat="Binary")

# auxiliary vars for score deviations
diff = LpVariable.dicts("diff", possible_scores, cat="Continuous")
# auxiliary vars for sex deviations
diffs = LpVariable.dicts("diffs", possible_sexes, cat="Continuous")

# objective: minimize the total deviation across all scores between training
# and validation sets
prob += lpSum([[diff[p] for p in possible_scores], [diffs[s] for s in possible_sexes]]), "total_score_deviation"

# constraint 1: define score distribution deviation
for p in possible_scores:
   try: 
       # calculate total number of segments with score p in train set
       score_in_train = lpSum([(segments[s] if scores[s] == p else 0) * x[s] for s in speakers])
       score_in_val = lpSum([(segments[s] if scores[s] == p else 0) * (1 - x[s]) for s in speakers])
       
       # define deviation between train and val for each score
       prob += (score_in_train / train_fraction - score_in_val / (1 - train_fraction) <= diff[p], f"upper_deviation_{p}")
       prob += (score_in_val / (1 - train_fraction) - score_in_train / train_fraction <= diff[p], f"lower_deviation_{p}")
       
       prob += (score_in_train - n_segments_train_desired_ps <= diff[p], f"upper_deviation_train_{p}")
       prob += (n_segments_train_desired_ps - score_in_train <= diff[p], f"lower_deviation_train_{p}")
       
       prob += (score_in_val - n_segments_val_desired_ps <= diff[p], f"upper_deviation_val_{p}")
       prob += (n_segments_val_desired_ps - score_in_val <= diff[p], f"lower_deviation_val_{p}")
       
   except KeyError:
       print(f"No speakers have a PHQ-9 score of {p}")

# constraint 2: define sex distribution deviation
for p in possible_sexes:
   try: 
       # calculate total number of segments with sex p in train and val set
       n_sex_in_train = lpSum([(segments[s] if sexes[s] == p else 0) * x[s] for s in speakers])
       n_sex_in_val = lpSum([(segments[s] if sexes[s] == p else 0) * (1 - x[s]) for s in speakers])
       
       # define deviation between train and val for each score
       prob += (n_sex_in_train / train_fraction - n_sex_in_val / (1 - train_fraction) <= diffs[p], f"upper_deviation_sex_{p}")
       prob += (n_sex_in_val / (1 - train_fraction) - n_sex_in_train / train_fraction <= diffs[p], f"lower_deviation_sex_{p}")
       
       #prob += (score_in_train - n_segments_train_desired_ps <= diff[p], f"upper_deviation_train_{p}")
       #prob += (n_segments_train_desired_ps - score_in_train <= diff[p], f"lower_deviation_train_{p}")
       
       #prob += (score_in_val - n_segments_val_desired_ps <= diff[p], f"upper_deviation_val_{p}")
       #prob += (n_segments_val_desired_ps - score_in_val <= diff[p], f"lower_deviation_val_{p}")
       
   except KeyError:
       print(f"No speakers have a PHQ-9 score of {p}")

# constraint 3: train set size within desired range
prob += (
    lpSum([segments[s] * x[s] for s in speakers]) >= n_segments_train_min,
    "min_training_size"
)
prob += (
    lpSum([segments[s] * x[s] for s in speakers]) <= n_segments_train_max,
    "max_training_size"
)

###############################################################################
# solve
###############################################################################
prob.solve(COIN_CMD(msg=True, timeLimit=120, threads=4))

# check the status
assert prob.status == LpStatusOptimal, "no optimal solution found. consider adjusting the tolerances or reviewing the data."

# assign databases to train or train sets based on the solution
speakers_train = [s for s in speakers if x[s].varValue == 1]
speakers_val = [s for s in speakers if x[s].varValue == 0]



###############################################################################
# print results
###############################################################################
print("\nselected training speakers:")
for s in speakers_train:
    print(f"- {s}")

print("\nselected validation speakers:")
for s in speakers_val:
    print(f"- {s}")


# validate the split
mask_train = db["global_pid"].isin(speakers_train)
mask_val = db["global_pid"].isin(speakers_val)

n_segments_train = np.count_nonzero(mask_train)
n_segments_val = np.count_nonzero(mask_val)

print(f"\ntotal segments: {n_segments_total}")
print(f"train set segments: {n_segments_train} ({(n_segments_train / n_segments_total):.2%})")
print(f"validation set segments: {n_segments_val} ({(n_segments_val / n_segments_total):.2%})")

# validate score distribution in train set
train_score_distribution = db[mask_train]["phq-9"].value_counts().sort_index()
print("\ntrain set score distribution:")
for p in possible_scores:
    count = train_score_distribution.get(p, 0)
    swsp = np.asarray(speakers_train)[[(segments[s] if scores[s] == p else 0) > 0 for s in speakers_train]]
    print(f"PHQ-9 = {p}: {count} files ({(count / n_segments_train):.2%}) from {len(swsp)} speaker(s)")

# validate score distribution in validation set
val_score_distribution = db[mask_val]["phq-9"].value_counts().sort_index()
print("\nval set score distribution:")
for p in possible_scores:
    count = val_score_distribution.get(p, 0)
    swsp = np.asarray(speakers_val)[[(segments[s] if scores[s] == p else 0) > 0 for s in speakers_val]]
    print(f"PHQ-9 = {p}: {count} files ({(count / n_segments_val):.2%}) from {len(swsp)} speaker(s)")

# statstical distances
dist_train, bins = np.histogram(db[db["global_pid"].isin(speakers_train)]["phq-9"], bins=score_bin_edges, density=True)
dist_val, _ = np.histogram(db[db["global_pid"].isin(speakers_val)]["phq-9"], bins=score_bin_edges, density=True)

bd_train_val = bhattacharyya(dist_train, dist_val)
bd_train_unif = bhattacharyya(dist_train, np.ones_like(dist_train) / len(possible_scores))
bd_val_unif = bhattacharyya(dist_val, np.ones_like(dist_val) / len(possible_scores))
jsd_train_val = jensenshannon(dist_train, dist_val)
jsd_train_unif = jensenshannon(dist_train, np.ones_like(dist_train) / len(possible_scores))
jsd_val_unif = jensenshannon(dist_val, np.ones_like(dist_val) / len(possible_scores))

print("\nStatistical distances of score distributions:")
print(f"Bhattacharyya distance train and val = {bd_train_val:.4f}")
print(f"Bhattacharyya distance train and uniform = {bd_train_unif:.4f}")
print(f"Bhattacharyya distance val and uniform = {bd_val_unif:.4f}")
print()
print(f"Jensen-Shannon distance train and val = {jsd_train_val:.4f}")
print(f"Jensen-Shannon distance train and uniform = {jsd_train_unif:.4f}")
print(f"Jensen-Shannon distance val and uniform = {jsd_val_unif:.4f}")

###############################################################################
# select 10% of validation segements to use for testing
###############################################################################
train_val = ["train" if x[s] else "val" for s in db["global_pid"]]

for speaker in speakers:
    if speaker in speakers_val:
        val_indices = db[db["global_pid"] == speaker].index
        if len(val_indices > 10):
            test_indices = rng.choice(val_indices, len(val_indices) // 10,
                                    replace=False)
            for i in test_indices:
                train_val[i] = "test"

###############################################################################
# save updated db with train/val (test) labels
###############################################################################
db = pd.read_csv("db.csv", dtype={"pid": "str", "file_num": "str"})
db["db"] = db["db"] + "_" + train_val
db.to_csv("train-val-test-db.csv", index=False)

###############################################################################
# inspect results visually
###############################################################################
if do_plot:
    # complete distribution
    fig, ax = plt.subplots()
    bottom = np.zeros(len(possible_scores))
    c = plt.cm.jet(np.linspace(0, 1, n_speakers))
    for i, speaker in enumerate(speakers):
        p = ax.bar(
            possible_scores,
            score_counts.loc[speaker],
            # fc=(i / n_speakers, 0, (n_speakers - i) / n_speakers),
            fc = c[i],
            ec="k",
            lw=0.5,
            width=1,
            bottom=bottom
        )
        bottom += score_counts.loc[speaker]

    ax.set(
        xlabel="PHQ-9",
        ylabel="number of segments",
        title="Distribution of all segments",
    )
    ylim = ax.get_ylim()
    #plt.show(block=False)


    # histogram of segments
    alpha = 0.6
    lw = 0.5
    width = 1
    fig, ax = plt.subplots()
    bottom = np.zeros(len(possible_scores))
    for i, speaker in enumerate(speakers_train):
        p = ax.bar(
            possible_scores,
            score_counts.loc[speaker],# / n_segments_train,
            fc = "tab:blue",
            ec="k",
            alpha=alpha,
            lw=lw,
            width=width,
            label="training" if i == 0 else "_",
            bottom=bottom
        )
        bottom += score_counts.loc[speaker]# / n_segments_train  
        
    bottom = np.zeros(len(possible_scores))
    for i, speaker in enumerate(speakers_val):
        p = ax.bar(
            np.array(possible_scores) + 0.,
            score_counts.loc[speaker],# / n_segments_val,
            fc="tab:orange",
            ec="k",
            alpha=alpha,
            lw=lw,
            width=width,
            label="validation" if i == 0 else "_",
            bottom=bottom
        )
        bottom += score_counts.loc[speaker]# / n_segments_val

    ax.axhline(n_segments_train_desired_ps, c="0.5", ls="--", label="ideal uniform dist.", zorder=-1)
    ax.axhline(n_segments_val_desired_ps, c="0.5", ls="--", zorder=-1)
    ax.set(
        xlabel="PHQ-9",
        ylabel="number of segments",
        title="Score distribution comparison (per segment)"
        # ylim=ylim
    )

    ax.legend()
    #plt.show(block=False)


    # histogram of segments (density)
    alpha = 0.6
    lw = 0.5
    width = 1
    fig, ax = plt.subplots()
    bottom = np.zeros(len(possible_scores))
    for i, speaker in enumerate(speakers_train):
        p = ax.bar(
            possible_scores,
            score_counts.loc[speaker] / n_segments_train,
            fc = "tab:blue",
            ec="k",
            alpha=alpha,
            lw=lw,
            width=width,
            label="training" if i == 0 else "_",
            bottom=bottom
        )
        bottom += score_counts.loc[speaker] / n_segments_train  
        
    bottom = np.zeros(len(possible_scores))
    for i, speaker in enumerate(speakers_val):
        p = ax.bar(
            np.array(possible_scores) + 0.,
            score_counts.loc[speaker] / n_segments_val,
            fc="tab:orange",
            ec="k",
            alpha=alpha,
            lw=lw,
            width=width,
            label="validation" if i == 0 else "_",
            bottom=bottom
        )
        bottom += score_counts.loc[speaker] / n_segments_val

    ax.axhline(1 / len(possible_scores), c="0.5", ls="--", label="ideal uniform dist.", zorder=-1)
    # ax.axhline(n_segments_val_desired_ps / n_segments_val, c="0.5", ls="--", zorder=-1)
    ax.set(
        xlabel="PHQ-9",
        ylabel="proportion of segments",
        # ylim=ylim,
        title="Score distribution comparison (density; per segment)"
    )

    ax.legend()
    #plt.show(block=False)



    alpha = 0.6

    fig, ax = plt.subplots()
    ax.bar(score_bin_edges[:-1], dist_train, width=width, alpha=alpha, edgecolor="k", hatch="..", label="training")
    ax.bar(score_bin_edges[:-1], dist_val, width=width, alpha=alpha, edgecolor="k", hatch="///", label="validation")

    ax.set(
        xlabel="PHQ-9",
        ylabel="propotion of segments",
        title="Score distribution comparison (density)"
    )
    # ax.axhline(n_segments_train_desired_ps, c="0.5", ls="--", label="ideal uniform dist.", zorder=-1)
    # ax.axhline(n_segments_val_desired_ps, c="0.5", ls="--", label="ideal uniform dist.", zorder=-1)
    ax.grid(visible=False)
    ax.legend()
    #plt.show(block=False)


    # gender identity distribution
    fig, ax = plt.subplots()
    tick_labels = ["f", "m", "d", "u"]
    width = 0.35
    ind = np.arange(len(tick_labels))
    ax.bar(ind,
        np.array([(mask_train & (db["sex"] == "f")).sum(),
            (mask_train & (db["sex"] == "m")).sum(),
            (mask_train & (db["sex"] == "d")).sum(),
            (mask_train & (db["sex"] == "u")).sum()]) / n_segments_train,
        width=width,
        label="train")
    ax.bar(ind + width,
        np.array([(mask_val & (db["sex"] == "f")).sum(),
            (mask_val & (db["sex"] == "m")).sum(),
            (mask_val & (db["sex"] == "d")).sum(),
            (mask_val & (db["sex"] == "u")).sum()]) / n_segments_val,
        width=width,
        label="val")

    ax.set_xticks(ind + width / 2, labels=tick_labels)
    ax.set(ylabel="proportion of segments",
        ylim=(0, 1),
        title="Distribution of self-reported sex")
    ax.legend()

    plt.show(block=True)
