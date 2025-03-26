import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

plt.style.use("default")
plt.rcParams.update({'font.size': 22})


def linlaw(x, a, b):
    """
    Linear function.

    Parameters:
    x (float or array-like): The input value(s).
    a (float): The slope of the line.
    b (float): The y-intercept of the line.

    Returns:
    float or array-like: The result of the linear function a * x + b.
    """
    return a * x + b


def powerlaw(x, a, b, c):
    """
    Power law function.

    Parameters:
    x (float or array-like): The input value(s).
    a (float): The exponent of the power law.
    b (float): The coefficient of the power law.
    c (float): The constant term added to the power law.

    Returns:
    float or array-like: The result of the power law function b * x ** a + c.
    """
    return b * x ** a + c


def curve_fit_log_linear(xdata, ydata):
    """
    Perform a linear fit on the logarithm of the x data and return the fit parameters and fitted y data.

    Parameters:
    xdata (array-like): The input x data.
    ydata (array-like): The input y data.

    Returns:
    tuple: A tuple containing the optimal values for the parameters (popt_log), the estimated covariance of popt_log (pcov_log), and the fitted y data (ydatafit).
    """
    xdata_log = np.log10(xdata)
    # Fit linear
    popt_log, pcov_log = curve_fit(linlaw, xdata_log, ydata)
    # y variables as results from fitting
    ydatafit = linlaw(xdata_log, *popt_log)
    return popt_log, pcov_log, ydatafit


def curve_fit_log_log(xdata, ydata):
    """
    Perform a linear fit on the logarithm of both the x and y data and return the fit parameters and fitted y data.

    Parameters:
    xdata (array-like): The input x data.
    ydata (array-like): The input y data.

    Returns:
    tuple: A tuple containing the optimal values for the parameters (popt_log), the estimated covariance of popt_log (pcov_log), and the fitted y data (ydatafit).
    """
    xdata_log = np.log10(xdata)
    ydata_log = np.log10(ydata)
    # Fit linear
    popt_log, pcov_log = curve_fit(linlaw, xdata_log, ydata_log)
    # y variables as results from fitting
    ydatafit = pow(10, linlaw(xdata_log, *popt_log))
    return popt_log, pcov_log, ydatafit


def find_fit(X, Y, color):
    """
    Perform a linear regression on the given data and plot the fit line with confidence interval.

    Parameters:
    X (array-like): The input x data.
    Y (array-like): The input y data.
    color (str): The color to use for the plot.

    Returns:
    None
    """
    slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
    print('Slope: {:.6f}'.format(slope), 'Intercept: {:.6f}'.format(intercept),
          'R-squared: {:.6f}'.format(r_value ** 2), 'Std_err: {:.6f}'.format(std_err))
    x_dummy = np.linspace(start=0, stop=8000, num=100)
    plt.plot(x_dummy, slope * x_dummy + intercept, color=color)
    plt.fill_between(x_dummy, slope * x_dummy + intercept - std_err, slope * x_dummy + intercept + std_err, color=color,
                     alpha=0.2)


def find_powerlaw_fit(x, y, color):
    """
    Perform a power law fit on the given data and plot the fit line.

    Parameters:
    x (float or array-like): The input x data.
    y (float or array-like): The input y data.
    color (str): The color to use for the plot.

    Returns:
    None
    """
    if x == 0:
        x = 0.000001
    if y == 0:
        y = 0.000001
    X = np.log10(x)
    Y = np.log10(y)

    slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
    print('Slope: {:.6f}'.format(slope), 'Intercept: {:.6f}'.format(intercept),
          'R-squared: {:.6f}'.format(r_value ** 2), 'Std_err: {:.6f}'.format(std_err))
    x_dummy = np.linspace(start=-500, stop=8000, num=100)
    plt.plot(x_dummy, (10 ** intercept) * (x_dummy ** slope), color=color)


def plot_reaxys_violin(kegg, chnops, mws):
    """
    Plot a violin plot comparing the chirality of compounds in KEGG and CHNOPS datasets.

    Parameters:
    kegg (array-like): The KEGG dataset containing chirality data.
    chnops (array-like): The CHNOPS dataset containing chirality data.
    mws (array-like): The molecular weights corresponding to the datasets.

    Returns:
    None
    """
    chnops_color = "#477C7D"
    kegg_color = "#AA0E35"

    fig, ax = plt.subplots(figsize=(11, 8))

    handles = []  # Handle array used for legend compatibility with Reaxys sampling

    kegg_violins = ax.violinplot(kegg, positions=mws, widths=75, showmeans=True)
    chnops_violins = ax.violinplot(chnops, positions=mws, widths=75, showmeans=True)

    # Color all CHNOPS violins
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
        vp = chnops_violins[partname]
        vp.set_edgecolor(chnops_color)
        vp.set_linewidth(2)

    for pc in chnops_violins['bodies']:
        pc.set_facecolor(chnops_color)
        pc.set_edgecolor(chnops_color)
        pc.set_alpha(0.5)

    # Now do all KEGG violins
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
        vp = kegg_violins[partname]
        vp.set_edgecolor(kegg_color)
        vp.set_linewidth(2)

    for pc in kegg_violins['bodies']:
        pc.set_facecolor(kegg_color)
        pc.set_edgecolor(kegg_color)
        pc.set_alpha(0.5)

    ax.set_ylabel("Chirality")
    ax.set_xlabel("Sample Size")

    # Legend
    chnops_patch = plt.Line2D([], [], color=chnops_color, marker="o", linewidth=0, label='Reaxys: CHNOPS Elements')
    kegg_patch = plt.Line2D([], [], color=kegg_color, marker="o", linewidth=0, label='Reaxys: KEGG Elements')

    ## Add Reaxys patches to handles
    handles.extend([chnops_patch, kegg_patch])

    # Legend
    lgnd = ax.legend(handles=handles, bbox_to_anchor=(0.62, 0.98), loc=2, borderaxespad=0., shadow=True,
                     prop={"size": 14})

    # Labels
    font_size = 14
    ax.set_xlabel("Number of Compounds")
    ax.set_ylabel("Ratio of Chiral Compounds")

    # # Axis
    # ax.set_ylim(0.45,0.85)
    # ax.set_xlim(-300, 7900)

    # Layout
    plt.tight_layout()
    plt.savefig("ReaxysSampling.png", dpi=600)
    plt.savefig("ReaxysSampling.pdf")
    plt.show()
    return


def plot_ratio_chiral_compounds(df, mws, chnops, kegg):
    """
    Plot the ratio of chiral compounds for different domains and Reaxys datasets.

    Parameters:
    df (dict): A dictionary containing dataframes for different domains.
    mws (array-like): An array of molecular weights corresponding to the datasets.
    chnops (array-like): The CHNOPS dataset containing chirality data.
    kegg (array-like): The KEGG dataset containing chirality data.

    Returns:
    None
    """
    # Get means of Reaxys data
    chnops_avgs = [np.mean(s) for s in chnops]
    kegg_avgs = [np.mean(s) for s in kegg]

    chnops_color = "dimgrey"
    kegg_color = "darkgrey"

    fig, ax = plt.subplots(figsize=(11, 8))

    color_number = {"LUCA": "#10D410", "Archaea": "#D65E00", "Bacteria": "#0B77B5",
                    "Eukarya": "#E69F03", "Metagenome": "#009E73", "Biosphere": "#4B0092"}

    list_domain = ["Metagenome", "Bacteria", "Archaea", "Eukarya"]

    handles = []  # Handle array used for legend compatibility with Reaxys sampling

    for domain in list_domain:
        x = df[domain].total_size
        y = df[domain].ratio_chiral
        s = ax.scatter(x, y, label=domain, s=5, color=color_number[domain], alpha=0.8)
        handles.append(s)  # Adding lines to an array for legend labeling
        find_fit(x, y, color_number[domain])

    # Add Reaxys averages to scatter plot
    ax.scatter(mws, chnops_avgs, color=chnops_color, s=15)
    ax.annotate("Reaxys CHNOPS", (6000, 0.575), color=chnops_color, weight='bold',
                fontsize=16, ha='center', va='center')

    ax.scatter(mws, kegg_avgs, color=kegg_color, s=15)
    ax.annotate("Reaxys KEGG", (6000, 0.55), color=kegg_color, weight='bold',
                fontsize=16, ha='center', va='center')

    # Add scaling fits (fit a polyfit line to averages)
    find_fit(mws, chnops_avgs, chnops_color)
    find_fit(mws, kegg_avgs, kegg_color)

    # Add Reaxys to Legend
    chnops_patch = plt.Line2D([], [], color=chnops_color, marker="o", linewidth=0, label='Reaxys: CHNOPS Elements')
    kegg_patch = plt.Line2D([], [], color=kegg_color, marker="o", linewidth=0, label='Reaxys: KEGG Elements')

    # Line for biosphere
    x = df["Biosphere"].total_size
    y = df["Biosphere"].ratio_chiral
    print("Biosphere ratio:", y)
    s = ax.scatter(x, y, s=100, color="k")  # , label="Biosphere")
    ax.annotate("Biosphere", (x - 150, y + 0.01), color="k", weight='bold',
                fontsize=16, ha='center', va='center')
    handles.append(s)
    plt.plot([-500, 8000], [y, y], ls='--', color="k", alpha=0.75)

    # Line for Luca
    x = df["LUCA"].total_size
    y = df["LUCA"].ratio_chiral
    print("LUCA ratio:", y)
    s = ax.scatter(x, y, s=100, color="k")  # , label="LUCA")
    ax.annotate("LUCA", (x - 300, y + 0.01), color="k", weight='bold',
                fontsize=16, ha='center', va='center')
    handles.append(s)
    plt.plot([-500, 8000], [y, y], ls='--', color="k", alpha=1)

    ## Add Reaxys patches to handles
    # handles.extend([chnops_patch, kegg_patch])

    # Legend
    lgnd = ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0., shadow=True, prop={"size": 18})

    for n in range(len(list_domain)):
        lgnd.legendHandles[n]._sizes = [40]

    # Labels
    ax.set_xlabel("Compounds Per Sample")
    ax.set_ylabel("Ratio of Chiral Compounds")

    # Axis
    ax.set_ylim(0.50, 0.90)
    ax.set_xlim(-300, 7900)

    ### Inset distribution axes
    left, bottom, width, height = [0.4, 0.525, 0.3, 0.4]
    ax2 = fig.add_axes([left, bottom, width, height])

    list_domain = ["Bacteria", "Metagenome", "Archaea", "Eukarya"]

    for domain in list_domain:
        x = df[domain].total_size
        y = df[domain].ratio_chiral
        num_bins = 100
        n, bins, patches = plt.hist(y, num_bins, label=domain, facecolor=color_number[domain], alpha=0.8,
                                    orientation="horizontal")

    # Flattening & adding Reaxys samples
    num_bins = 100
    ax2.hist([x for sub in kegg for x in sub], num_bins, label="Reaxys:\nKEGG Elements", facecolor=kegg_color,
             alpha=0.8, orientation="horizontal")
    ax2.hist([x for sub in chnops for x in sub], num_bins, label="Reaxys:\nCHNOPS Elements", facecolor=chnops_color,
             alpha=0.8, orientation="horizontal")

    # Line for biosphere
    x = df["Biosphere"].total_size
    y = df["Biosphere"].ratio_chiral
    # ax2.plot([x, x], [-10, 1300], label = "Biosphere", ls='--', color="k", alpha=1)
    # ax2.annotate("Biosphere", (y - 0.05, 1450), color="k", weight='bold',
    #             fontsize=18, ha='center', va='center')

    # Line for Luca
    x = df["LUCA"].total_size
    y = df["LUCA"].ratio_chiral
    # ax2.plot([x, x], [-10, 1300], label = "LUCA", ls='--', color="k", alpha=1)
    # ax2.annotate("LUCA", (y + 0.03, 1450), color="k", weight='bold',
    #             fontsize=18, ha='center', va='center')

    # Labels
    font_size = 14
    ax2.set_ylabel("Ratio of Chiral Compounds", fontsize=font_size)
    ax2.set_xlabel("Samples", fontsize=font_size)

    ax2.set_xscale('log')
    ax2.tick_params(axis="both", which="major", labelsize=font_size)
    ax2.set_ylim(0.4, 0.8)

    # Layout
    plt.tight_layout()
    plt.show()

    # Save Figure
    # file_plot = "../../Figures/Fig2.svg"
    # plt.savefig(file_plot)


def plot_ratio_achiral_compounds(df, mws, chnops_achiral, kegg_achiral):
    """
    Plot the ratio of achiral compounds for different domains and Reaxys datasets.

    Parameters:
    df (dict): A dictionary containing dataframes for different domains.
    mws (array-like): An array of molecular weights corresponding to the datasets.
    chnops_achiral (array-like): The CHNOPS dataset containing achirality data.
    kegg_achiral (array-like): The KEGG dataset containing achirality data.

    Returns:
    None
    """
    chnops_avgs = [np.mean(s) for s in chnops_achiral]
    kegg_avgs = [np.mean(s) for s in kegg_achiral]

    chnops_color = "dimgrey"
    kegg_color = "darkgrey"
    fig, ax = plt.subplots(figsize=(11, 8))

    color_number = {"LUCA": "#10D410", "Archaea": "#D55E00", "Bacteria": "#0072B2",
                    "Eukarya": "#E69F00", "Metagenome": "#009473", "Biosphere": "#4B0092"}

    list_domain = ["Metagenome", "Bacteria", "Archaea", "Eukarya"]

    handles = []

    for domain in list_domain:
        x = df[domain].total_size
        y = df[domain].ratio_achiral
        s = ax.scatter(x, y, label=domain, s=5, color=color_number[domain], alpha=0.8)
        handles.append(s)  # Adding lines to an array for legend labeling
        find_fit(x, y, color_number[domain])

    # Add Reaxys averages to scatter plot
    ax.scatter(mws, chnops_avgs, color=chnops_color, s=15)
    ax.scatter(mws, kegg_avgs, color=kegg_color, s=15)

    # Add scaling fits (fit a polyfit line to averages)
    find_fit(mws, chnops_avgs, chnops_color)
    find_fit(mws, kegg_avgs, kegg_color)

    ax.set_ylabel("Chirality")
    ax.set_xlabel("Sample Size")

    # Legend for KEGG samples
    chnops_patch = plt.Line2D([], [], color=chnops_color, marker="o", linewidth=0, label='Reaxys: CHNOPS Elements')
    kegg_patch = plt.Line2D([], [], color=kegg_color, marker="o", linewidth=0, label='Reaxys: KEGG Elements')

    # Line for biosphere
    x = df["Biosphere"].total_size
    y = df["Biosphere"].ratio_achiral
    s = ax.scatter(x, y, label="Biosphere", s=70, color=color_number["Biosphere"])
    handles.append(s)
    plt.plot([-500, 8000], [y, y], ls='--', color=color_number["Biosphere"], alpha=1)

    # Line for Luca
    x = df["LUCA"].total_size
    y = df["LUCA"].ratio_achiral
    s = ax.scatter(x, y, label="LUCA", s=70, color=color_number["LUCA"])
    handles.append(s)
    plt.plot([-500, 8000], [y, y], ls='--', color=color_number["LUCA"], alpha=1)

    ## Add Reaxys patches to handles
    handles.extend([chnops_patch, kegg_patch])

    # # Legend
    # lgnd = ax.legend(handles = handles, bbox_to_anchor=(0.02, 0.99), loc=2, borderaxespad=0., shadow=True, prop={"size": 13})

    # for n in range(len(list_domain) + 2):
    #     lgnd.legendHandles[n]._sizes = [30]

    # Labels
    font_size = 14
    ax.set_xlabel("Number of Compounds")
    ax.set_ylabel("Ratio of Achiral Compounds")

    # Axis
    ax.set_ylim(0.15, 0.5)
    ax.set_xlim(-300, 7900)

    # Layout
    plt.tight_layout()

    # Save Figure
    file_plot = "scale_ratio_Achiral.png"
    plt.savefig(file_plot)
    plt.show()

    return None


def plot_histogram_ratio_chiral_compounds(df, chnops, kegg):
    """
    Plot a histogram of the ratio of chiral compounds for different domains and Reaxys datasets.

    Parameters:
    df (dict): A dictionary containing dataframes for different domains.
    chnops (array-like): The CHNOPS dataset containing chirality data.
    kegg (array-like): The KEGG dataset containing chirality data.

    Returns:
    None
    """
    chnops_color = "dimgrey"
    kegg_color = "darkgrey"

    fig, ax = plt.subplots(figsize=(8, 6))

    color_number = {"LUCA": "#10D410",
                    "Archaea": "#D55E00",
                    "Bacteria": "#0072B2",
                    "Eukarya": "#E69F00",
                    "Metagenome": "#009473",
                    "Biosphere": "#4B0092"}

    list_domain = ["Bacteria", "Metagenome", "Archaea", "Eukarya"]

    num_bins = 100

    for domain in list_domain:
        x = df[domain].total_size
        y = df[domain].ratio_chiral

        n, bins, patches = plt.hist(y, num_bins, label=domain, facecolor=color_number[domain], alpha=0.8)

    plt.hist([x for sub in kegg for x in sub], num_bins,
             label="Reaxys:\nKEGG Elements",
             facecolor=kegg_color,
             alpha=0.8)
    plt.hist([x for sub in chnops for x in sub], num_bins,
             label="Reaxys:\nCHNOPS Elements",
             facecolor=chnops_color,
             alpha=0.8)

    # Line for biosphere
    x = df["Biosphere"].total_size
    y = df["Biosphere"].ratio_chiral
    plt.plot([y, y], [-10, 1300], label="Biosphere", ls='--', color="k", alpha=1)
    plt.annotate("Biosphere", (y - 0.05, 1450),
                 color="k",
                 weight='bold',
                 fontsize=18,
                 ha='center',
                 va='center')

    # Line for Luca
    x = df["LUCA"].total_size
    y = df["LUCA"].ratio_chiral
    plt.plot([y, y], [-10, 1300], label="LUCA", ls='--', color="k", alpha=1)
    plt.annotate("LUCA", (y + 0.03, 1450),
                 color="k",
                 weight='bold',
                 fontsize=18,
                 ha='center',
                 va='center')

    # Labels
    font_size = 14
    ax.set_xlabel("Ratio of Chiral Compounds", fontsize=font_size)
    ax.set_ylabel("Number of Samples", fontsize=font_size)

    # Legend
    # ax.legend(bbox_to_anchor=(0.67, 0.98), loc=2, borderaxespad=0., shadow=True, prop={"size":8})
    # ax.legend(loc = "upper left", shadow=True, prop={"size":10})

    # Axis
    # ax.set_ylim(0,10000)
    ax.set_yscale('log')
    ax.tick_params(axis="both", which="major", labelsize=font_size)

    # Layout
    plt.tight_layout()

    # Save Figure
    file_plot = "histogram_ratio_chiral.png"
    plt.savefig(file_plot)
    plt.show()


def plot_histogram_ratio_achiral_compounds(df, chnops_achiral, kegg_achiral):
    """
    Plot a histogram of the ratio of achiral compounds for different domains and Reaxys datasets.

    Parameters:
    df (dict): A dictionary containing dataframes for different domains.
    chnops_achiral (array-like): The CHNOPS dataset containing achirality data.
    kegg_achiral (array-like): The KEGG dataset containing achirality data.

    Returns:
    None
    """
    chnops_color = "dimgrey"
    kegg_color = "darkgrey"
    num_bins = 100

    fig, ax = plt.subplots(figsize=(8, 6))

    color_number = {"LUCA": "#10D410",
                    "Archaea": "#D55E00",
                    "Bacteria": "#0072B2",
                    "Eukarya": "#E69F00",
                    "Metagenome": "#009473",
                    "Biosphere": "#4B0092"}

    list_domain = ["Bacteria", "Metagenome", "Archaea", "Eukarya"]

    for domain in list_domain:
        x = df[domain].total_size
        y = df[domain].ratio_achiral

        n, bins, patches = plt.hist(y, num_bins, label=domain, facecolor=color_number[domain], alpha=0.8)

    # JM: flattening & adding Reaxys samples
    plt.hist([x for sub in kegg_achiral for x in sub], num_bins,
             label="Reaxys:\nKEGG Elements",
             facecolor=kegg_color,
             alpha=0.8)
    plt.hist([x for sub in chnops_achiral for x in sub], num_bins,
             label="Reaxys:\nCHNOPS Elements",
             facecolor=chnops_color,
             alpha=0.8)

    # Line for biosphere
    x = df["Biosphere"].total_size
    y = df["Biosphere"].ratio_achiral
    plt.plot([y, y], [-10, 1300], label="Biosphere", ls='--', color=color_number["Biosphere"], alpha=1)

    # Line for Luca
    x = df["LUCA"].total_size
    y = df["LUCA"].ratio_achiral
    plt.plot([y, y], [-10, 1300], label="LUCA", ls='--', color=color_number["LUCA"], alpha=1)

    # Labels
    font_size = 14
    ax.set_xlabel("Ratio of Achiral Compounds", fontsize=font_size)
    ax.set_ylabel("Number of Samples", fontsize=font_size)

    # Legend
    # ax.legend(bbox_to_anchor=(0.02, 0.98), loc=2, borderaxespad=0., shadow=True)
    ax.legend(loc="upper right", shadow=True, prop={"size": 10})

    # Axis
    # ax.set_ylim(0,10000)
    ax.set_yscale('log')
    ax.tick_params(axis="both", which="major", labelsize=font_size)

    # Layout
    plt.tight_layout()

    # Save Figure
    file_plot = "histogram_ratio_Achiral.png"
    plt.savefig(file_plot)
    plt.show()


if __name__ == "__main__":
    # Load the data
    data_dir = ""
    df = dict()
    for domain in ["Archaea", "Metagenome", "Bacteria", "Eukarya", "Biosphere", "LUCA"]:
        df[domain] = pd.read_csv(data_dir + "species_chirality_in_%s.csv" % domain)

    # Reaxys Data
    chnops = pickle.load(file=open("../ReaxysSampling_Fig6/Data/chiral_chnops_sampling_FULL.p", "rb"))
    kegg = pickle.load(file=open("../ReaxysSampling_Fig6/Data/chiral_kegg_sampling_FULL.p", "rb"))

    mws = np.arange(100, 7100, 100)

    # Switch around Reaxys data for achiral
    chnops_achiral = list(1 - np.array(chnops))
    kegg_achiral = list(1 - np.array(kegg))

    # Plot the violin plot
    plot_reaxys_violin(kegg, chnops, mws)

    # Plot the ratio of chiral compounds
    plot_ratio_chiral_compounds(df, mws, chnops, kegg)

    # Plot the ratio of achiral compounds
    plot_ratio_achiral_compounds(df, mws, chnops_achiral, kegg_achiral)

    # Plot the histogram of the ratio of chiral compounds
    plot_histogram_ratio_chiral_compounds(df, chnops, kegg)

    # Plot the histogram of the ratio of achiral compounds
    plot_histogram_ratio_achiral_compounds(df, chnops_achiral, kegg_achiral)

    # Biosphere compared to metagenome and eukarya distributions, using percentiles
    print("Metagenomes:", stats.percentileofscore(df["Metagenome"].ratio_chiral, 0.632, kind="rank"))
    print("Eukarya:", stats.percentileofscore(df["Eukarya"].ratio_chiral, 0.632, kind="rank"))

    # LUCA compared to all distributions except archaea
    print("Metagenomes:", stats.percentileofscore(df["Metagenome"].ratio_chiral, 0.653, kind="rank"))
    print("Eukarya:", stats.percentileofscore(df["Eukarya"].ratio_chiral, 0.653, kind="rank"))
    print("Bacteria:", stats.percentileofscore(df["Bacteria"].ratio_chiral, 0.653, kind="rank"))
