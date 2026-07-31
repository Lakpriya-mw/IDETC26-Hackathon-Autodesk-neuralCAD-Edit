from src.utils.args import parse_args
from src.utils.process_config import load_config
import os
import os.path as osp
import json
from src.utils.db import DatabaseManager
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def parse_rating(rating: dict) -> dict:
    """
    Parses a rating dictionary to extract relevant information.
    
    Args:
        rating (dict): The rating dictionary to parse.
        
    Returns:
        dict: A dictionary of metric: score containing the parsed information.
    """

    if rating["user"] == "similarity_eval":
        return_dict = {}
        if "dino similarity gt" in rating:
            return_dict["dino-v2_similarity"] = rating["dino similarity gt"]
        if "chamfer similarity norm gt" in rating:
            return_dict["chamfer_similarity_norm"] = rating["chamfer similarity norm gt"]
        if "diff f1 gt" in rating:
            return_dict["diff_f1"] = rating["diff f1 gt"]
        return return_dict

    return None

def plot_ratings(config, scores: dict, difficulty: str = "all", request_type: str = "edit", mode="ratings"):

    fig, ax = plt.subplots(figsize=(12, 6))

    out_dir = osp.join(config["storage_dir"]["path"], "results")
    os.makedirs(out_dir, exist_ok=True)
    out_fn = osp.join(out_dir, f"{mode}_{request_type}_{difficulty}.png")

    
    # Prepare data for grouped bar chart
    # users = list(scores.keys())
    users = config["benchmark_eval_users"][request_type]
    metrics = set()
    for user_scores in scores.values():
        metrics.update(user_scores.keys())
    metrics = list(metrics)
    metrics.sort()  # Sort metrics for consistent ordering

    print(metrics)


    # Position bars - group by user first, then by metric
    bar_width = 0.8 / len(metrics)
    x = np.arange(len(users))

    print(scores)

    for i, metric in enumerate(metrics):
        metric_means = []
        for user in users:
            values = scores.get(user, {}).get(metric, {})

            values = [v if v is not None else 0.0 for v in values.values()]

            mean_score = np.mean(values) if values else 0.0
            metric_means.append(mean_score)
            # print(f"{user} - {metric}: {mean_score:.4f} (n={len(values)})")
        
        pos = x + i * bar_width - (len(metrics) - 1) * bar_width / 2
        ax.bar(pos, metric_means, width=bar_width, label=metric)
    
    ax.set_ylabel("Mean Score")
    ax.set_xlabel("Model")
    ax.set_title(f"Task: {request_type} {mode}, Difficulty: {difficulty}.")
    ax.set_xticks(x)
    ax.set_xticklabels(users)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_fn)
    # plt.show()

def display_rating_results(config: dict, dbm: DatabaseManager, difficulty: str = "all", request_fields={"eval_vis_multi": True, "eval_geometric": True}, request_type="edit", save_plot=True):

    request_fields["request_type"] = request_type

    if difficulty == "all":
        pass
    else:
        request_fields["difficulty"] = difficulty
        
    request_ids = dbm.requests.find(request_fields)
    request_ids = [request["_id"] for request in request_ids]
    request_ids.sort()


    # scores = {u: [] for u in config["benchmark_eval_users"]}
    scores = {}

    # loop over all ratings in the database
    ratings_iterator = dbm.ratings.find()
    for rating in ratings_iterator:

        edit = dbm.edits.find_one({"_id": rating["edit"]})

        if not edit:
            # print(f"rating {rating['_id']} has no associated edit , skipping.")
            # print(rating)
            continue

        if "request" not in edit:
            # print(f"Edit {edit['_id']} has no request field, skipping.")
            # print(edit)
            continue
        edit_request_id = edit["request"]
        request = dbm.requests.find_one({"_id": edit_request_id})

        if not request:
            # print(f"Edit {edit['_id']} has no associated request , skipping.")
            # print(edit)
            continue

        user = dbm.users.find_one({"_id": edit["user"]})

        if edit_request_id not in request_ids:
            continue

        if request["request_type"] != request_type:
            continue

        valid_user = False
        if user["_id"] in config["benchmark_eval_users"][request_type]:
            valid_user = True
            valid_user_id = user["_id"]
        if "other human" in config["benchmark_eval_users"][request_type] and user.get("is_human", True) and edit["user"] != request["user"]:
            valid_user = True
            valid_user_id = "other human"
        if "gt human" in config["benchmark_eval_users"][request_type] and edit["user"] == request["user"]:
            valid_user = True
            valid_user_id = "gt human"



        if not valid_user:
            continue

        print(rating)


        metrics = parse_rating(rating)

        if not metrics:
            continue

        for k, v in metrics.items():
            user_scores = scores.get(valid_user_id, {})
            scores_dict = user_scores.get(k, {})
            scores_dict[edit_request_id] = v
            # scores_list.append(v)
            user_scores[k] = scores_dict
            scores[valid_user_id] = user_scores

    all_metrics = set()
    for user_scores in scores.values():
        all_metrics.update(user_scores.keys())

    print(all_metrics)

    # for every request_id without a score in scores, add a placeholder
    for request_id in request_ids:
        for user_id in config["benchmark_eval_users"][request_type]:
            if user_id not in scores:
                print(f"User {user_id} not in scores, adding placeholder.")
                scores[user_id] = {}
            for metric in all_metrics:
                if metric not in scores.get(user_id, {}):
                    scores[user_id][metric] = {}
                if request_id not in scores.get(user_id, {}).get(metric, {}):
                    scores[user_id][metric][request_id] = None

    if save_plot:
        plot_ratings(config, scores, difficulty=difficulty, request_type=request_type, mode="ratings")

    return scores




METRIC_DISPLAY_NAMES = {
    "chamfer_similarity_norm": "Chamfer similarity (norm)",
    "diff_f1": "Diff F1",
    "dino-v2_similarity": "DINOv2 similarity",
}

# Baseline models from the paper are shown in grey; any other (newly added)
# model is highlighted in blue.
BASELINE_MODELS = {
    "gemini-3-pro_cadquery-script",
    "gpt-5.2_cadquery-script",
    "claude-sonnet-4.5_cadquery-script",
}
BASELINE_COLOR = "#999999"
NEW_MODEL_COLOR = "#1f77b4"

# Users that are treated as reference lines rather than bars.
HUMAN_BASELINE_KEY = "other human"
HUMAN_BASELINE_LABEL = "human baseline"
HUMAN_BASELINE_COLOR = "#444444"


def _model_color(model, index=0):
    return BASELINE_COLOR if model in BASELINE_MODELS else NEW_MODEL_COLOR


def _bar_models(config, request_type):
    """Models rendered as bars: config order, excluding humans (gt/baseline)."""
    return [
        m for m in config["benchmark_eval_users"][request_type]
        if m not in ("gt human", HUMAN_BASELINE_KEY)
    ]


def _score_or_zero(v):
    """Map missing or failed scores to 0.0 so models are penalized for failures."""
    if v is None or v != v:
        return 0.0
    return float(v)


def _aggregate_scores_by_metric(results: dict):
    """Collapse a results dict (task -> model -> metric -> {edit_id: score}) into
    metric -> model -> list_of_scores, pooling across all tasks/difficulties.

    Missing scores (``None`` placeholders for failed/unrated edits) and NaN
    values (e.g. from ``diff_f1`` on unloadable meshes) are counted as 0.0
    rather than dropped, so a model is penalized for failed runs. This
    matches the leaderboard notebook, which averages every edit with a 0.0
    default for missing metrics.
    """
    metric_model_scores = {}
    for _task, model_data in results.items():
        for model, metric_data in model_data.items():
            for metric, score_dict in metric_data.items():
                if isinstance(score_dict, dict):
                    values = [_score_or_zero(v) for v in score_dict.values()]
                elif isinstance(score_dict, list):
                    values = [_score_or_zero(v) for v in score_dict]
                else:
                    values = [_score_or_zero(score_dict)]
                metric_model_scores.setdefault(metric, {}).setdefault(model, []).extend(values)
    return metric_model_scores


def faceted_bar_plot(config: dict, results: dict, request_type: str = "edit", metrics=None, save=True):
    """
    Bar chart with one facet (subplot) per metric, showing the mean score for
    every model. Each facet is taller than it is wide. The human baseline is
    drawn as a dashed reference line rather than a bar. Replaces the previous
    radar/ratings plots.

    Args:
        config: config dict (provides benchmark_eval_users and storage_dir).
        results: task -> model -> metric -> {edit_id: score} results dict.
        request_type: which benchmark_eval_users list to use for model ordering.
        metrics: optional explicit metric ordering; defaults to the enabled three.
    """
    metric_model_scores = _aggregate_scores_by_metric(results)

    if metrics is None:
        metrics = [m for m in ["chamfer_similarity_norm", "diff_f1", "dino-v2_similarity"] if m in metric_model_scores]
        # include any additional metrics that showed up, for robustness
        metrics += [m for m in metric_model_scores if m not in metrics]

    models = _bar_models(config, request_type)
    colors = [_model_color(m, i) for i, m in enumerate(models)]

    n = len(metrics)
    # Portrait facets: each is taller than wide.
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 6.5), squeeze=False)
    axes = axes[0]

    x = np.arange(len(models))
    baseline_present = False
    for ax, metric in zip(axes, metrics):
        means = []
        for model in models:
            values = metric_model_scores.get(metric, {}).get(model, [])
            means.append(float(np.mean(values)) if values else 0.0)
        ax.bar(x, means, color=colors)

        # human baseline as a dashed horizontal reference line
        baseline_values = metric_model_scores.get(metric, {}).get(HUMAN_BASELINE_KEY, [])
        if baseline_values:
            baseline_mean = float(np.mean(baseline_values))
            ax.axhline(baseline_mean, linestyle="--", linewidth=1.5,
                       color=HUMAN_BASELINE_COLOR, label=HUMAN_BASELINE_LABEL)
            baseline_present = True

        ax.set_title(METRIC_DISPLAY_NAMES.get(metric, metric), fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Mean score")
        ax.set_ylim(0, 1)
        for xi, mean in zip(x, means):
            ax.text(xi, mean, f"{mean:.2f}", ha="center", va="bottom", fontsize=8)

    if baseline_present:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")

    fig.suptitle(f"Metric comparison across models ({request_type})")
    plt.tight_layout()

    if save:
        out_dir = osp.join(config["storage_dir"]["path"], "results")
        os.makedirs(out_dir, exist_ok=True)
        fig_fn = osp.join(out_dir, "metric_bar_facets.png")
        plt.savefig(fig_fn, dpi=200, bbox_inches="tight")
        print(f"Saved faceted bar plot to {fig_fn}")

    return fig, axes


def cost_barplot(config: dict, dbm: DatabaseManager, request_type: str = "edit", save=True):
    """
    Bar plot of the mean per-edit cost estimate for each model, with error bars
    (standard deviation across that model's edits). Styling is kept consistent
    with ``faceted_bar_plot`` (same per-model colours, rotated labels, value
    labels). Cost is read from each edit's ``token_counts.cost_estimate``.

    Only models present in ``benchmark_eval_users`` that have cost data (i.e.
    non-human harness runs) are shown.
    """
    models = _bar_models(config, request_type)

    model_costs = {}
    for edit in dbm.edits.find({}):
        user = edit.get("user")
        if user not in models:
            continue
        token_counts = edit.get("token_counts") or {}
        cost = token_counts.get("cost_estimate")
        if cost is None:
            continue
        model_costs.setdefault(user, []).append(float(cost))

    ordered_models = [m for m in models if model_costs.get(m)]
    means = [float(np.mean(model_costs[m])) for m in ordered_models]
    stds = [float(np.std(model_costs[m])) for m in ordered_models]
    colors = [_model_color(m, i) for i, m in enumerate(ordered_models)]

    fig, ax = plt.subplots(figsize=(3.2 * max(1, len(ordered_models)), 6.5))

    if ordered_models:
        x = np.arange(len(ordered_models))
        ax.bar(x, means, yerr=stds, color=colors, capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(ordered_models, rotation=45, ha="right", fontsize=8)
        for xi, mean in zip(x, means):
            ax.text(xi, mean, f"{mean:.2f}", ha="center", va="bottom", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No cost data available", ha="center", va="center", transform=ax.transAxes)

    ax.set_ylabel("Estimated cost per edit ($)")
    ax.set_title(f"Mean cost per edit ({request_type})", fontsize=10)
    plt.tight_layout()

    if save:
        out_dir = osp.join(config["storage_dir"]["path"], "results")
        os.makedirs(out_dir, exist_ok=True)
        fig_fn = osp.join(out_dir, "cost_barplot.png")
        plt.savefig(fig_fn, dpi=200, bbox_inches="tight")
        print(f"Saved cost bar plot to {fig_fn}")

    return fig, ax


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` Axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding Axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def all_tasks_radar_plot(config: dict, dbm: DatabaseManager, results: dict, chosen_models=None, primary_result_keys_override=None, save=True):
    """
    Create a radar plot showing performance across different benchmark types.
    
    Args:
        config: Configuration dictionary with primary_result_keys
        dbm: DatabaseManager instance
        results: Dictionary organized as benchmark_type -> users -> metric -> scores_dict. scores_dict is a dictionary mapping edit_request_id to score.

    Returns:
        tuple: (fig, ax) matplotlib figure and axes objects, or (None, None) if plot cannot be created
    """

    if chosen_models:
        filtered_results = {}
        for task in results:
            filtered_results[task] = {model: data for model, data in results[task].items() if model in chosen_models}
        results = filtered_results

    # Select just the primary result keys
    if primary_result_keys_override:
        primary_result_keys = primary_result_keys_override
    else:
        primary_result_keys = config.get("primary_result_keys", None)
    if primary_result_keys is None:
        print("No primary result keys specified in config, using all keys.")

    # Get benchmark types and collect all users across all benchmarks
    benchmark_types = list(results.keys())
    if len(benchmark_types) < 3:
        print(f"Need at least 3 benchmark types for radar plot, only have {len(benchmark_types)}")
        return None, None
    
    all_users = set()
    for benchmark_data in results.values():
        all_users.update(benchmark_data.keys())
    all_users = sorted(list(all_users))
    
    if len(all_users) == 0:
        print("No users found in results data")
        return None, None

    # Set up radar chart
    N = len(benchmark_types)
    theta = radar_factory(N, frame='polygon')
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='radar'))
    
    # Colors for different users
    colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k', 'orange', 'purple', 'brown']
    
    # Prepare data for each user
    for i, user in enumerate(all_users):
        user_scores = []
        
        for benchmark_type in benchmark_types:
            if benchmark_type in results and user in results[benchmark_type]:
                # Get the primary metric for this benchmark type
                primary_metric = primary_result_keys.get(benchmark_type)
                user_data = results[benchmark_type][user]

                if not primary_metric or primary_metric not in user_data:
                    print(f"No primary metric specified for benchmark type '{benchmark_type}', using first available metric.")
                    primary_metric = list(user_data.keys())[0] if user_data else None
                
                # Calculate mean score for this metric
                scores = user_data[primary_metric]
                if isinstance(scores, list) and len(scores) > 0:
                    mean_score = np.mean(scores)
                elif isinstance(scores, (int, float)):
                    mean_score = scores
                elif isinstance(scores, dict) and len(scores) > 0:
                    mean_score = np.mean([v if v is not None else 0.0 for v in list(scores.values())])
                else:
                    mean_score = 0.0

            else:
                mean_score = 0.0
                print(f"Warning: No data for user '{user}' in benchmark '{benchmark_type}'")
            
            user_scores.append(mean_score)
        
        # Plot the user's scores
        color = colors[i % len(colors)]
        ax.plot(theta, user_scores, 'o-', linewidth=2, color=color, label=user, markersize=6)
        ax.fill(theta, user_scores, facecolor=color, alpha=0.15)
        
        print(f"User '{user}' scores: {[f'{score:.3f}' for score in user_scores]}")
    
    # Set up the plot
    ax.set_varlabels(benchmark_types)
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0, 1.35)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()

    if save:
        out_dir = osp.join(config["storage_dir"]["path"], "results")
        os.makedirs(out_dir, exist_ok=True)
        fig_fn = osp.join(out_dir, "radar_plot_all_tasks.png")
        plt.savefig(fig_fn, dpi=300, bbox_inches='tight')
    
    return fig, ax





def main():
    # Parse command-line arguments
    args = parse_args()
    # Load configuration
    config = load_config(args.config)

    dbm = DatabaseManager(config)

    dbm.print_db_summary()

    display_rating_results(config, dbm, difficulty="all", request_type="edit")
  

    out_dir = osp.join(config["storage_dir"]["path"], "results")
    result_path = osp.join(out_dir, "all_results.json")
    with open(result_path, "r") as f:
        results = json.load(f)

    all_tasks_radar_plot(config=config, dbm=None, results=results)



if __name__ == "__main__":
    main()