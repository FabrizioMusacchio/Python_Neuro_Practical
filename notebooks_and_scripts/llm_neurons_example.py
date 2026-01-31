"""
Linear mixed models in a neuroscience style grouped data setting.

Goal
-----
Simulate trial level responses with repeated measurements within animals and
demonstrate why treating all trials as independent observations can be misleading.

We use a simple continuous predictor x (stimulus strength) and a continuous response y
(neural response). Measurements are grouped by animal, which induces within animal
dependence that violates the independence assumption of ordinary least squares (OLS).

The script compares several approaches:

1) Global OLS on all trials (ignores grouping)
   Model: y ~ x
   This fit treats all trials as independent and therefore uses standard errors that
   are typically too optimistic when within animal correlations are present.

2) ANCOVA with fixed animal intercept shifts (conditions on the observed animals)
   Model: y ~ x + C(animal)
   This allows different baselines per animal but enforces a common slope.

3) Aggregation and two stage alternatives
   a) OLS on per animal means: average x and y within each animal, then fit y ~ x
      This is valid but discards trial level variability and often reduces power.
   b) Two stage hierarchical slope test: fit y ~ x within each animal, then test the
      mean slope across animals.
      This highlights the loss of information when collapsing to one estimate per animal.

4) Linear mixed model with random intercepts (animals as a sample from a population)
   Model: y ~ x + (1 | animal)
   This models animal specific baselines as random draws from a population distribution
   and estimates the between animal variance component.

5) Linear mixed model with random intercepts and random slopes
   Model: y ~ x + (1 + x | animal)
   This allows both baseline and slope to vary by animal and yields group specific
   conditional estimates via BLUPs. The group specific effects are partially pooled,
   leading to shrinkage toward the population mean when per animal information is limited.

6) ANCOVA with interaction (fixed animal specific slopes)
   Model: y ~ x * C(animal)
   This assigns a separate slope to each animal without pooling, which can produce
   highly variable group slopes when the number of observations per animal is small.

We keep the response Gaussian to focus on mixed effects structure and interpretation.

Acknowledgments
---------------
The structure and several diagnostic plot ideas are adapted from Edouard Duchesnay's
LMM tutorial:

https://duchesnay.github.io/pystatsml/statistics/lmm/lmm.html

The main change is the framing in neuroscience terms, with stimulus strength as a
continuous predictor and neural response as the dependent variable. Many thanks to
Edouard Duchesnay for making this material openly available.

Authorship
----------
Author: Fabrizio Musacchio
Date: Jan 31, 2026
License: choose a license file in the repository (e.g., MIT, BSD-3, or CC BY 4.0 for text)
"""
# %% IMPORTS
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import scipy.stats as spstats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# remove spines right and top for better aesthetics:
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.bottom'] = False
plt.rcParams.update({'font.size': 12})
# %% FUNCTIONS

def simulate_animal_data(
    seed=0,
    n_animals=6,
    n_obs_per_animal=40,
    beta0=8.0,             # population intercept
    beta1=0.25,            # population slope for x
    sigma_animal_intercept=2.5,
    sigma_animal_slope=0.12,
    sigma_noise=1.0):
    """ 
    Simulate animal-level data with random intercepts and slopes.
    """
    rng = np.random.default_rng(seed)
    animals = [f"subj_{i}" for i in range(n_animals)]

    # animal random intercepts and slopes:
    b0 = rng.normal(0.0, sigma_animal_intercept, size=n_animals)
    b1 = rng.normal(0.0, sigma_animal_slope, size=n_animals)

    rows = []
    for i, a in enumerate(animals):
        # choose a plausible x distribution:
        # e.g. stimulus strength 0..10
        x = rng.integers(0, 11, size=n_obs_per_animal).astype(float)

        # conditional mean with animal specific intercept and slope
        mu = (beta0 + b0[i]) + (beta1 + b1[i]) * x
        y = mu + rng.normal(0.0, sigma_noise, size=n_obs_per_animal)

        for xx, yy in zip(x, y):
            rows.append({"animal": a, "x": xx, "y": yy})

    return pd.DataFrame(rows)

def simulate_animal_data_lmm_friendly(
    seed=0,
    n_animals=40,
    n_obs_min=3,
    n_obs_max=8,
    beta0=8.0,
    beta1=0.25,
    sigma_animal_intercept=2.5,
    sigma_animal_slope=0.35,
    rho_intercept_slope=-0.6,
    sigma_noise=1.0,
    x_mode="group_shifted"):
    """
    LMM-friendly simulation.

    Key features:
    * many groups, few observations per group (and unbalanced)
    * random intercepts AND random slopes
    * correlated random effects (b0,b1)
    * optional group-shifted x distributions

    x_mode:
      "shared"        -> all groups draw x from the same distribution
      "group_shifted" -> each group sees its own x-range, causing confounding risk
    """
    rng = np.random.default_rng(seed)
    animals = [f"subj_{i}" for i in range(n_animals)]

    # correlated random effects (b0,b1) per animal:
    cov = np.array([
        [sigma_animal_intercept**2, rho_intercept_slope*sigma_animal_intercept*sigma_animal_slope],
        [rho_intercept_slope*sigma_animal_intercept*sigma_animal_slope, sigma_animal_slope**2]])
    b = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov, size=n_animals)
    b0 = b[:, 0]
    b1 = b[:, 1]

    rows = []
    for i, a in enumerate(animals):
        n_obs = int(rng.integers(n_obs_min, n_obs_max + 1))

        if x_mode == "shared":
            x = rng.integers(0, 11, size=n_obs).astype(float)
        else:
            # each animal sees a shifted x-window
            center = float(rng.integers(0, 11))
            x = center + rng.normal(0.0, 1.0, size=n_obs)
            x = np.clip(x, 0.0, 10.0)

        mu = (beta0 + b0[i]) + (beta1 + b1[i]) * x
        y = mu + rng.normal(0.0, sigma_noise, size=n_obs)

        for xx, yy in zip(x, y):
            rows.append({"animal": a, "x": float(xx), "y": float(yy)})

    return pd.DataFrame(rows)

def plot_qq_diagnostics(resid, fitted, groups, title_prefix, outpath=None):
    """ 
    QQ plot of residuals from linear model.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    sm.qqplot(resid, line="45", fit=True, ax=ax, alpha=0.7, lw=0)
    ax.set_title(f"{title_prefix}:\nQQ plot of residuals")
    plt.tight_layout()
    if outpath is not None:
        plt.savefig(os.path.join(outpath, f"{title_prefix}_qqplot.png"), dpi=300)
        plt.close()
    else:
        plt.show()

def plot_residual_diagnostics(residual, prediction, group=None, group_boxplot=False,
                      title_prefix="", outpath=None, fname_prefix=""):
    """
    Plots three panels for linear model diagnostics:
      1) Residuals vs prediction, colored by group (if provided)
      2) Residual density (overall)
      3) Residual density by group (or boxplot by group)
    """
    diag_df = pd.DataFrame(dict(prediction=np.asarray(prediction), residual=np.asarray(residual)))
    if group is not None:
        diag_df["group"] = np.asarray(group)

        fig, axes = plt.subplots(1, 3, figsize=(7, 3.5), sharey=True)

        sns.scatterplot(
            x="prediction", y="residual", hue="group", data=diag_df,
            ax=axes[0], s=35, alpha=0.85, legend=False
        )
        axes[0].axhline(0.0, linewidth=1.0)
        axes[0].set_title("Residuals vs pred")

        sns.kdeplot(y="residual", data=diag_df, fill=True, ax=axes[1])
        axes[1].set_title("Residuals")

        if group_boxplot:
            sns.boxplot(y="residual", x="group", data=diag_df, ax=axes[2])
            axes[2].set_title("Residuals by group")
        else:
            sns.kdeplot(y="residual", hue="group", data=diag_df, fill=True, ax=axes[2])
            axes[2].set_title("Residuals by group")

        # if group/animal size is >4, don't show legend:
        if diag_df["group"].nunique() <= 4:
            # ensure legend is visible and tidy
            leg = axes[2].get_legend()
            if leg is None:
                axes[2].legend(title="group", frameon=False)
            else:
                leg.set_frame_on(False)
        else:
            # remove legend for many groups
            leg = axes[2].get_legend()
            if leg is not None:
                leg.remove()

        if title_prefix:
            fig.suptitle(title_prefix)
        plt.tight_layout()

        if outpath is not None:
            os.makedirs(outpath, exist_ok=True)
            fn = f"{fname_prefix}_lm_diagnosis.png" if fname_prefix else "lm_diagnosis.png"
            plt.savefig(os.path.join(outpath, fn), dpi=300)
            plt.close()
        else:
            plt.show()

    else:
        fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), sharey=True)
        sns.scatterplot(x="prediction", y="residual", data=diag_df, ax=axes[0], s=35, alpha=0.85)
        axes[0].axhline(0.0, linewidth=1.0)
        axes[0].set_title("Residuals vs pred")

        sns.kdeplot(y="residual", data=diag_df, fill=True, ax=axes[1])
        axes[1].set_title("Residuals")

        if title_prefix:
            fig.suptitle(title_prefix)
        plt.tight_layout()

        if outpath is not None:
            os.makedirs(outpath, exist_ok=True)
            fn = f"{fname_prefix}_lm_diagnosis.png" if fname_prefix else "lm_diagnosis.png"
            plt.savefig(os.path.join(outpath, fn), dpi=300)
            plt.close()
        else:
            plt.show()

def rmse_coef_tstat_pval(mod, var: str):
    """
    Return RMSE, coefficient, t-stat and p-value for a fitted statsmodels model.

    Works for OLS. For MixedLM, statsmodels reports z-values (Wald z) rather than t.
    In that case, we still return (RMSE, coef, z, p).
    """
    resid = np.asarray(mod.resid)
    sse = float(np.sum(resid ** 2))
    df_resid = float(getattr(mod, "df_resid", len(resid) - 1))

    rmse = np.sqrt(sse / df_resid)

    coef = float(mod.params[var])

    # OLS: tvalues, MixedLM: zvalues
    if hasattr(mod, "tvalues") and (var in getattr(mod, "tvalues", {})):
        stat = float(mod.tvalues[var])
    elif hasattr(mod, "tvalues") and isinstance(mod.tvalues, (pd.Series, np.ndarray)):
        stat = float(mod.tvalues[var])
    else:
        # MixedLMResults: use z-values
        stat = float(mod.tvalues[var]) if hasattr(mod, "tvalues") else float(mod.params[var] / mod.bse[var])

    pval = float(mod.pvalues[var])

    return rmse, coef, stat, pval

def plot_ancova_oneslope_grpintercept(x, y, group, df, model, outpath=None, fname="ancova_oneslope_grpintercept.png"):
    """
    ANCOVA: one common slope, group specific fixed intercept shifts.

    Model form: y ~ x + C(group)
    Plot: scatter by group + black lines with same slope, shifted intercept.
    """
    legend_on = True if df[group].nunique() <= 6 else False
    fig, ax = plt.subplots(figsize=(6.5, 4))
    
    """ g = sns.lmplot(x=x, y=y, hue=group, data=df, fit_reg=False, height=4, aspect=1.4,
                   legend=legend_on, ax=ax) """
    sns.scatterplot(x=x, y=y, hue=group, data=df, s=35, alpha=0.8, lw=0,
        legend=legend_on,ax=ax)
    #ax = g.ax

    palette = itertools.cycle(sns.color_palette())
    x_jitter = -0.2

    base_intercept = float(model.params["Intercept"])
    slope = float(model.params[x])

    for group_lab, group_df in df.groupby(group):
        x_ = group_df[x].to_numpy()
        color = next(palette)

        # fixed intercept shift for this group, reference group has 0 shift
        key = f"C({group})[T.{group_lab}]"
        group_offset = float(model.params[key]) if key in model.params else 0.0

        y_pred = base_intercept + slope * x_ + group_offset

        # draw the common-slope line for this group
        order = np.argsort(x_)
        ax.plot(x_[order], y_pred[order], color="k", linewidth=2.0)

        # arrow indicating the intercept shift
        ax.arrow(0 + x_jitter, base_intercept, 0, group_offset,
                 head_width=0.25, length_includes_head=True, color=color)
        x_jitter += 0.2

    # deactivate legend if too many groups:
    if df[group].nunique() > 6:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    else:
        ax.legend(frameon=False, bbox_to_anchor=(1.00, 1), loc='upper left')

    # deactivate left and bottom spines for aesthetics:
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # set proper x and y labels:
    ax.set_xlabel(x + ": stimulus strength")
    ax.set_ylabel(y + ": neural response")

    ax.set_title("ANCOVA: one slope, group fixed intercepts")
    plt.tight_layout()

    if outpath is not None:
        os.makedirs(outpath, exist_ok=True)
        plt.savefig(os.path.join(outpath, fname), dpi=300)
        plt.close()
    else:
        plt.show()

def plot_lmm_oneslope_randintercept_OLD(x, y, group, df, model, outpath=None, fname="lmm_oneslope_randintercept.png"):
    """
    LMM: one common slope, random intercept per group.

    Model form: y ~ x + (1 | group)

    statsmodels stores random intercept BLUPs in:
      model.random_effects[group_lab]
    For random-intercept-only, that is a length-1 vector/series.
    """
    g = sns.lmplot(x=x, y=y, hue=group, data=df, fit_reg=False, height=4, aspect=1.4)
    ax = g.ax

    palette = itertools.cycle(sns.color_palette())
    x_jitter = -0.2

    base_intercept = float(model.params["Intercept"])
    slope = float(model.params[x])

    for group_lab, group_df in df.groupby(group):
        x_ = group_df[x].to_numpy()
        color = next(palette)

        re = model.random_effects[group_lab]
        # robustly get intercept BLUP
        if isinstance(re, (pd.Series, dict)):
            group_offset = float(re.iloc[0]) if isinstance(re, pd.Series) else float(list(re.values())[0])
        else:
            group_offset = float(re[0])

        y_pred = base_intercept + slope * x_ + group_offset

        order = np.argsort(x_)
        ax.plot(x_[order], y_pred[order], color=color, linewidth=2.0)

        ax.arrow(0 + x_jitter, base_intercept, 0, group_offset,
                 head_width=0.25, length_includes_head=True, color=color)
        x_jitter += 0.2

    ax.set_title("LMM: one slope, random intercepts")
    plt.tight_layout()

    if outpath is not None:
        os.makedirs(outpath, exist_ok=True)
        plt.savefig(os.path.join(outpath, fname), dpi=300)
        plt.close()
    else:
        plt.show()

def plot_lmm_oneslope_randintercept(
    x, y, group, df, model,
    outpath=None, fname="lmm_oneslope_randintercept.png",
    add_text=True):
    """
    LMM: one common slope, random intercept per group.

    Model form: y ~ x + (1 | group)

    Arrows indicate the group-specific random intercept BLUP:
        beta0  ->  beta0 + b0_g

    If add_text=True, annotate the population-level distribution assumption:
        (beta0 + b0_g) ~ N(beta0, Var(b0))
    """

    fig, ax = plt.subplots(figsize=(6.5, 4))
    legend_on = True if df[group].nunique() <= 6 else False
    """ g = sns.lmplot(x=x, y=y, hue=group, data=df, fit_reg=False, height=4, aspect=1.4,
                   legend=legend_on) """
    sns.scatterplot(x=x, y=y, hue=group, data=df, s=35, alpha=0.8, lw=0,
        legend=legend_on,ax=ax)
    #ax = g.ax

    palette = itertools.cycle(sns.color_palette())
    x_jitter = -0.2

    base_intercept = float(model.params["Intercept"])
    slope = float(model.params[x])

    # Random-intercept variance estimate (sigma_b^2)
    # For random-intercept-only models, cov_re is 1x1
    try:
        var_b0 = float(model.cov_re.iloc[0, 0])
    except Exception:
        # fallback: sometimes it is exposed as "Group Var" in params
        var_b0 = float(model.params.get("Group Var", np.nan))

    for group_lab, group_df in df.groupby(group):
        x_ = group_df[x].to_numpy()
        color = next(palette)

        re = model.random_effects[group_lab]

        # robustly get intercept BLUP
        if isinstance(re, pd.Series):
            group_offset = float(re.iloc[0])
        elif isinstance(re, dict):
            group_offset = float(list(re.values())[0])
        else:
            group_offset = float(re[0])

        y_pred = base_intercept + slope * x_ + group_offset

        order = np.argsort(x_)
        ax.plot(x_[order], y_pred[order], color=color, linewidth=2.0)

        ax.arrow(
            0 + x_jitter, base_intercept, 0, group_offset,
            head_width=0.25, length_includes_head=True, color=color)

        if add_text and np.isfinite(var_b0) and legend_on:
            ax.text(
                0.15, base_intercept + group_offset,
                f"~N({base_intercept:.3f}, {var_b0:.2f})",
                fontsize=9, color="k", va="center")

        x_jitter += 0.2

    if legend_on:
        ax.legend(frameon=False, bbox_to_anchor=(1.00, 1), loc='upper left')

    # set some proper labels:
    ax.set_xlabel(x + ": stimulus strength")
    ax.set_ylabel(y + ": neural response")

    ax.set_title("LMM: one slope, random intercepts")
    plt.tight_layout()

    if outpath is not None:
        os.makedirs(outpath, exist_ok=True)
        plt.savefig(os.path.join(outpath, fname), dpi=300)
        plt.close()
    else:
        plt.show()

def plot_lmm_randintercept_randslope(
    x, y, group, df, model,
    outpath=None, fname="lmm_random_intercept_and_slope_fit.png",
    add_arrows=True):
    """
    LMM with random intercept and random slope:
        y ~ x + (1 + x | group)

    Plot:
      * points colored by group
      * dashed black: fixed effects only (population line)
      * solid colored: group-specific lines using BLUPs
    """
    
    fig, ax = plt.subplots(figsize=(6.5, 4))
    legend_on = True if df[group].nunique() <= 6 else False
    """ g = sns.lmplot(x=x, y=y, hue=group, data=df, fit_reg=False, height=4, aspect=1.4,
                   legend=legend_on) """
    sns.scatterplot(x=x, y=y, hue=group, data=df, s=35, alpha=0.8, lw=0,
                    legend=legend_on,ax=ax)
    #ax = g.ax

    xline = np.linspace(df[x].min(), df[x].max(), 200)

    beta0 = float(model.params["Intercept"])
    beta1 = float(model.params[x])

    # population line (fixed effects only)
    ax.plot(xline, beta0 + beta1 * xline, color="k", linestyle="--", linewidth=2.0)

    palette = itertools.cycle(sns.color_palette())
    x_jitter = -0.2

    for group_lab, group_df in df.groupby(group):
        color = next(palette)

        re = model.random_effects[group_lab]
        # For re_formula="1 + x", statsmodels returns two entries (intercept, slope)
        if isinstance(re, pd.Series):
            b0 = float(re.iloc[0])
            b1 = float(re.iloc[1])
        elif isinstance(re, dict):
            vals = list(re.values())
            b0 = float(vals[0])
            b1 = float(vals[1])
        else:
            b0 = float(re[0])
            b1 = float(re[1])

        ax.plot(xline, (beta0 + b0) + (beta1 + b1) * xline, color=color, linewidth=2.5)

        if add_arrows:
            # intercept arrow at x=0
            ax.arrow(
                0 + x_jitter, beta0, 0, b0,
                head_width=0.25, length_includes_head=True, color=color)
            # small slope indication: show delta y over delta x=1 at x=0
            ax.arrow(
                0 + x_jitter, beta0 + b0, 1.0, b1,
                head_width=0.25, length_includes_head=True, color=color)
            x_jitter += 0.2

    ax.set_title("LMM: random intercept and random slope")
    ax.set_xlabel(x + ": stimulus strength")
    ax.set_ylabel(y + ": neural response")

    if legend_on:
        ax.legend(frameon=False, bbox_to_anchor=(1.00, 1), loc='upper left')

    # deactivate left and bottom spines for aesthetics:
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    plt.tight_layout()
    if outpath is not None:
        os.makedirs(outpath, exist_ok=True)
        plt.savefig(os.path.join(outpath, fname), dpi=300)
        plt.close()
    else:
        plt.show()

def plot_ancova_fullmodel(x, y, group, df, model, outpath=None, fname="ANCOVA_fullmodel.png"):
    """
    ANCOVA full model with interaction:
      y ~ x * C(group)
    Equivalent to:
      y ~ x + C(group) + x:C(group)

    Plot:
      * points per group
      * dashed black lines: "no-interaction" reference with group intercept offsets
            y_ref_g(x) = Intercept + beta_x * x + offset_g
        where beta_x is the slope of the reference group (from the fitted interaction model)
        and offset_g is the group intercept shift (C(group)[T.g]).
      * colored lines: full interaction model predictions within each group
      * arrows: show offset_g at x=0
    """
    
    legend_on = True if df[group].nunique() <= 6 else False
    g = sns.lmplot(x=x, y=y, hue=group, data=df, fit_reg=False, height=4, aspect=1.6,
                   legend=False)
    ax = g.ax

    palette = itertools.cycle(sns.color_palette())
    x_jitter = -0.2

    base_intercept = float(model.params["Intercept"])
    base_slope = float(model.params[x])  # slope for the reference group in the interaction model

    for group_lab, group_df in df.groupby(group):
        color = next(palette)

        x_ = group_df[x].to_numpy()
        order = np.argsort(x_)
        x_sorted = x_[order]

        # --- fixed intercept offset for this group (reference group has 0) ---
        key = f"C({group})[T.{group_lab}]"
        group_offset = float(model.params[key]) if key in model.params else 0.0

        # --- dashed reference: same slope, different intercepts (NO interaction slopes) ---
        y_ref = base_intercept + base_slope * x_sorted + group_offset
        ax.plot(x_sorted, y_ref, color="k", linestyle="--", linewidth=2.0, zorder=5)

        # --- colored: full interaction predictions (group specific slope + intercept) ---
        y_pred = np.asarray(model.predict(group_df))[order]
        ax.plot(x_sorted, y_pred, color=color, linewidth=2.5)

        # --- arrow indicates the intercept shift at x=0 ---
        ax.arrow(0 + x_jitter, base_intercept, 0, group_offset,
                 head_width=0.25, length_includes_head=True, color=color)
        x_jitter += 0.2

    # deactivate left and bottom spines for aesthetics:
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # set proper x and y labels:
    ax.set_xlabel(x + ": stimulus strength")
    ax.set_ylabel(y + ": neural response")
    
    # if legend_on, put the legend right outside the plot
    if legend_on:
        ax.legend(frameon=False, bbox_to_anchor=(1.00, 1), loc='upper left')

    ax.set_title("ANCOVA full: group specific intercepts\nand slopes(interaction)")
    plt.tight_layout()

    if outpath is not None:
        os.makedirs(outpath, exist_ok=True)
        plt.savefig(os.path.join(outpath, fname), dpi=300)
        plt.close()
    else:
        plt.show()

def extract_group_slopes_from_ancova_full(model, df, group_col="animal", x_col="x"):
    """
    For y ~ x * C(group), compute the slope per group:
      slope_group = beta_x + beta_{x:C(group)[T.group]}
    Reference group gets just beta_x.
    """
    beta_x = float(model.params[x_col])
    slopes = {}
    groups = df[group_col].unique()
    for g in groups:
        key = f"{x_col}:C({group_col})[T.{g}]"
        slopes[g] = beta_x + (float(model.params[key]) if key in model.params else 0.0)
    return slopes

def extract_group_slopes_from_lmm(model, df, group_col="animal", x_col="x"):
    """
    For MixedLM with re_formula="1 + x", slope per group is:
      slope_group = beta_x + b1_group
    """
    beta_x = float(model.params[x_col])
    slopes = {}
    for g in df[group_col].unique():
        re = model.random_effects[g]
        if isinstance(re, pd.Series):
            b1 = float(re.iloc[1])
        else:
            b1 = float(re[1])
        slopes[g] = beta_x + b1
    return slopes

def plot_slope_comparison(slopes_ancova, slopes_lmm, outpath, fname="slope_comparison.png"):
    keys = sorted(set(slopes_ancova.keys()) & set(slopes_lmm.keys()))
    y1 = np.array([slopes_ancova[k] for k in keys])
    y2 = np.array([slopes_lmm[k] for k in keys])

    plt.figure(figsize=(6, 5))
    for subject_i, subject in enumerate(keys):
        plt.scatter(y1[subject_i], y2[subject_i], s=40, alpha=0.8)
    lo = min(y1.min(), y2.min())
    hi = max(y1.max(), y2.max())
    plt.plot([lo, hi], [lo, hi], linewidth=2.0)
    plt.xlabel("Group slope, ANCOVA full")
    plt.ylabel("Group slope, LMM BLUP")
    plt.title("Group-specific slopes\nANCOVA full versus LMM (shrinkage)")
    plt.tight_layout()
    plt.savefig(os.path.join(outpath, fname), dpi=300)
    plt.close()

# %% MAIN SCRIPT

outpath = "llm_results"
os.makedirs(outpath, exist_ok=True)

df = simulate_animal_data(seed=41,
                          n_animals=3,
                          n_obs_per_animal=40,
                          beta0=8.0,
                          beta1=0.25,
                          sigma_animal_intercept=2.5,
                          sigma_animal_slope=0.12,
                          sigma_noise=1.0)


""" outpath = "llm_results_lmm_preferred"
os.makedirs(outpath, exist_ok=True)
df = simulate_animal_data_lmm_friendly(
        seed=41,
        n_animals=40,
        n_obs_min=3,
        n_obs_max=8,
        beta0=8.0,
        beta1=0.25,
        sigma_animal_intercept=2.5,
        sigma_animal_slope=0.35,
        rho_intercept_slope=-0.6,
        sigma_noise=1.0,
        x_mode="group_shifted") """


# ----------------------------
# plotting: raw data
# ----------------------------
plt.figure(figsize=(6.5, 4))
for a, g in df.groupby("animal"):
    plt.scatter(g["x"], g["y"], s=35, alpha=0.8, label=a, lw=0)
plt.xlabel("x: stimulus strength (=continuous predictor)")
plt.ylabel("y: neural response (=dependent variable)")
plt.title("Raw data, grouped by animal")
# show a legend only if few animals
if df["animal"].nunique() <= 6:
    plt.legend(frameon=False, bbox_to_anchor=(1.0, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(outpath, "raw_data_by_animal.png"), dpi=300)
plt.close()


# ----------------------------
# 1) global OLS (ignores grouping)
# ----------------------------
lm_global = smf.ols("y ~ x", data=df).fit()
print("\nGlobal OLS: y ~ x")
print(lm_global.summary().tables[1])

plot_qq_diagnostics(
    resid=lm_global.resid.to_numpy(),
    fitted=lm_global.fittedvalues.to_numpy(),
    groups=df["animal"].to_numpy(),
    title_prefix="Global_OLS",
    outpath=outpath)

# global fit plot:
legend_on = df["animal"].nunique() <= 6
fig, ax = plt.subplots(figsize=(6.5, 4))
# scatter by animal:
sns.scatterplot(x="x", y="y", hue="animal", data=df, s=35, alpha=0.8, lw=0,
    legend=legend_on,ax=ax)
# global OLS line
xline = np.linspace(df["x"].min(), df["x"].max(), 200)
yline = lm_global.params["Intercept"] + lm_global.params["x"] * xline
ax.plot(xline, yline, color="k", linewidth=2.0)
ax.set_xlabel("x: stimulus strength")
ax.set_ylabel("y: neural response")
ax.set_title("Global OLS fit (ignores grouping)")
# remove left and bottom spines for aesthetics:
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
if legend_on:
    ax.legend(frameon=False, bbox_to_anchor=(1.00, 1), loc="upper left")
else:
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
plt.tight_layout()
plt.savefig(os.path.join(outpath, "Global_OLS_fit.png"), dpi=300)
plt.close()

plot_residual_diagnostics(
    residual=lm_global.resid,
    prediction=lm_global.predict(df),
    group=df["animal"],
    title_prefix="Global OLS",
    outpath=outpath,
    fname_prefix="Global_OLS")

# create and store first summary row:
results = pd.DataFrame(columns=["Model", "RMSE", "Coef_x", "Stat_x", "Pval_x"])
rmse, coef, stat, pval = rmse_coef_tstat_pval(lm_global, "x")
results.loc[len(results)] = ["OLS global (biased SE)", rmse, coef, stat, pval]


# ----------------------------
# 2) ANCOVA with animal as fixed intercept shifts
# ----------------------------
ancova = smf.ols("y ~ x + C(animal)", data=df).fit()
print("\nANCOVA fixed animal intercepts: y ~ x + C(animal)")
print(ancova.summary().tables[1])

plot_qq_diagnostics(
    resid=ancova.resid.to_numpy(),
    fitted=ancova.fittedvalues.to_numpy(),
    groups=df["animal"].to_numpy(),
    title_prefix="ANCOVA fixed intercepts",
    outpath=outpath)

# ANCOVA one slope, fixed intercept shifts:
plot_ancova_oneslope_grpintercept(
    x="x", y="y", group="animal", df=df, model=ancova,
    outpath=outpath, fname="ANCOVA_oneslope_fixed_intercepts.png")

plot_residual_diagnostics(
    residual=ancova.resid,
    prediction=ancova.predict(df),
    group=df["animal"],
    title_prefix="ANCOVA fixed intercepts",
    outpath=outpath,
    fname_prefix="ANCOVA_fixed_intercepts")

rmse, coef, stat, pval = rmse_coef_tstat_pval(ancova, "x")
results.loc[len(results)] = ["ANCOVA fixed intercepts", rmse, coef, stat, pval]

# ----------------------------
# 3) aggregation: average within animal, then OLS
# ----------------------------
agg = df.groupby("animal")[["x", "y"]].mean().reset_index()
lm_agg = smf.ols("y ~ x", data=agg).fit()
print("\nAggregation: OLS on animal means")
print(lm_agg.summary().tables[1])

# plot aggregated points and regression
plt.figure(figsize=(6, 4))
for i, row in agg.iterrows():
    plt.scatter(row["x"], row["y"], s=120, alpha=0.9, label=row["animal"], lw=0)
xline = np.linspace(agg["x"].min(), agg["x"].max(), 100)
plt.plot(xline, lm_agg.params["Intercept"] + lm_agg.params["x"] * xline, linewidth=2.0, 
         c="k", label="OLS on means")
plt.xlabel("mean x per animal")
plt.ylabel("mean y per animal")
plt.title("Aggregation level analysis")
if agg["animal"].nunique() <= 6:
    plt.legend(frameon=False, bbox_to_anchor=(1.0, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(outpath, "aggregation_level_analysis.png"), dpi=300)
plt.close()


# ----------------------------
# 3b) hierarchical two-stage approach:
# level 1: fit y ~ x within each animal, extract slope
# level 2: test mean slope across animals
# ----------------------------
lv1 = []
for animal_lab, animal_df in df.groupby("animal"):
    m = smf.ols("y ~ x", data=animal_df).fit()
    lv1.append([animal_lab, float(m.params["x"])])

lv1 = pd.DataFrame(lv1, columns=["animal", "beta_x"])
lm_level2 = smf.ols("beta_x ~ 1", data=lv1).fit()

print("\nHierarchical two-stage: test mean within-animal slopes")
print(lm_level2.summary().tables[1])

# plot: within-animal regressions + barplot of slopes:
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for animal_lab, animal_df in df.groupby("animal"):
    sns.regplot(x="x", y="y", data=animal_df, ax=axes[0], scatter=True, ci=None, scatter_kws={"s":35, "alpha":0.8})
axes[0].set_title("Level 1: regressions within animal")
axes[0].set_xlabel("x (stimulus strength; per-animal data)")
axes[0].set_ylabel("y (neural response)")
sns.barplot(x="animal", y="beta_x", hue="animal", data=lv1, ax=axes[1], legend=False)
axes[1].axhline(0.0, linestyle="--")
axes[1].set_title("Level 2: slopes by animal")
axes[1].set_xlabel("animal")
axes[1].set_ylabel("beta_x")
# rotate x-ticks if many animals
if lv1["animal"].nunique() > 5:
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right", fontsize=6)
plt.tight_layout()
plt.savefig(os.path.join(outpath, "hierarchical_two_stage.png"), dpi=300)
plt.close()

# store results for level2 (here Intercept is mean slope)
rmse, coef, stat, pval = rmse_coef_tstat_pval(lm_level2, "Intercept")
results.loc[len(results)] = ["Two-stage hierarchical (mean slope)", rmse, coef, stat, pval]

# ----------------------------
# 4) LMM random intercept: y ~ x + (1 | animal)
# ----------------------------
lmm_int = smf.mixedlm("y ~ x", data=df, groups=df["animal"], re_formula="1")
lmm_int_fit = lmm_int.fit(reml=True, method="lbfgs")
print("\nLMM random intercept: y ~ x + (1 | animal)")
print(lmm_int_fit.summary())

plot_qq_diagnostics(
    resid=lmm_int_fit.resid.to_numpy(),
    fitted=lmm_int_fit.fittedvalues.to_numpy(),
    groups=df["animal"].to_numpy(),
    title_prefix="LMM random intercept",
    outpath=outpath)

# LMM random intercept plot:
plot_lmm_oneslope_randintercept(x="x", y="y", group="animal", df=df, 
                                model=lmm_int_fit,
                                outpath=outpath, 
                                add_text=False,
                                fname="LMM_random_intercept_fit.png")

plot_residual_diagnostics(
    residual=lmm_int_fit.resid,
    prediction=lmm_int_fit.fittedvalues,
    group=df["animal"],
    title_prefix="LMM random intercept",
    outpath=outpath,
    fname_prefix="LMM_random_intercept")

rmse, coef, stat, pval = rmse_coef_tstat_pval(lmm_int_fit, "x")
results.loc[len(results)] = ["LMM random intercept", rmse, coef, stat, pval]

# ----------------------------
# 5) LMM random intercept and slope: y ~ x + (1 + x | animal)
# ----------------------------
lmm_slope = smf.mixedlm("y ~ x", data=df, groups=df["animal"], re_formula="1 + x")
lmm_slope_fit = lmm_slope.fit(reml=True, method="lbfgs")
print("\nLMM random intercept and slope: y ~ x + (1 + x | animal)")
print(lmm_slope_fit.summary())

plot_qq_diagnostics(
    resid=lmm_slope_fit.resid.to_numpy(),
    fitted=lmm_slope_fit.fittedvalues.to_numpy(),
    groups=df["animal"].to_numpy(),
    title_prefix="LMM random intercept and\nslope",
    outpath=outpath)
    
plot_lmm_randintercept_randslope(
    x="x", y="y", group="animal", df=df, model=lmm_slope_fit,
    outpath=outpath, fname="LMM_random_intercept_and_slope_fit.png",
    add_arrows=True)

plot_residual_diagnostics(
    residual=lmm_slope_fit.resid,
    prediction=lmm_slope_fit.fittedvalues,
    group=df["animal"],
    title_prefix="LMM random intercept and slope",
    outpath=outpath,
    fname_prefix="LMM_random_intercept_slope")
    
# print random effects (BLUPs) per animal:
print("\nRandom effects (BLUPs) from random slope model:")
for k, v in lmm_slope_fit.random_effects.items():
    # v is a Series with entries like "Group" (intercept) and "x"
    print(k, dict(v))
    
rmse, coef, stat, pval = rmse_coef_tstat_pval(lmm_slope_fit, "x")
results.loc[len(results)] = ["LMM random intercept + slope", rmse, coef, stat, pval]
    
# ----------------------------
# 6) ANCOVA full model with interaction (fixed group slopes)
# y ~ x * C(animal)
# ----------------------------
ancova_full = smf.ols("y ~ x * C(animal)", data=df).fit()
print("\nANCOVA full: y ~ x * C(animal)")
print(ancova_full.summary().tables[1])

plot_ancova_fullmodel(
    x="x", y="y", group="animal", df=df, model=ancova_full,
    outpath=outpath, fname="ANCOVA_full_interaction.png")

plot_residual_diagnostics(
    residual=ancova_full.resid,
    prediction=ancova_full.predict(df),
    group=df["animal"],
    title_prefix="ANCOVA full (interaction)",
    outpath=outpath,
    fname_prefix="ANCOVA_full")

rmse, coef, stat, pval = rmse_coef_tstat_pval(ancova_full, "x")
results.loc[len(results)] = ["ANCOVA full (interaction)", rmse, coef, stat, pval]
    
    
# compare group-specific slopes from ANCOVA full vs LMM
slopes_ancova = extract_group_slopes_from_ancova_full(ancova_full, df, group_col="animal", x_col="x")
slopes_lmm = extract_group_slopes_from_lmm(lmm_slope_fit, df, group_col="animal", x_col="x")
plot_slope_comparison(slopes_ancova, slopes_lmm, outpath=outpath)
    
# ----------------------------
# 7) Results summary
# ----------------------------
print("\nModel comparison table")
print(results)

results.to_csv(os.path.join(outpath, "model_comparison_table.csv"), index=False)
# %% END