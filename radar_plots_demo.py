import matplotlib.pyplot as plt
import pandas as pd
from math import pi
import numpy as np

RADAR_NAMES = {
    'Acc.': 'Avg Acc',
    'Mem.': f'            Mem.\n              Efficiency',
    'FLOPs': f'                Compute \n             Efficiency',
    'Forgetting': f'               $-$Forgetting',
    'S_minus': ' \n\n        Direct \n        Transfer',
    'S_plus': ' \n\nKnowledge       \nUpdate      ',
    'S_in': 'Transfer           \nInput dist.          ',
    'S_out': 'Transfer            \nOutput dist.              ',
    'S_pl': 'Plasticity            ',
}


def main():
    data = pd.read_csv('results/data.csv')
    data = data.set_index('Approach')
    print(data)

    chart = radar_chart(data, False, ['Independent', 'Finetune', 'Online EWC'], data.columns)
    chart.savefig('results/radar_chart_stack.pdf')

    chart = radar_chart(data, True, None, data.columns)
    chart.savefig('results/radar_chart_all.pdf')



def normalize_df(df, offset=.2, inv=None):
    """
    Normalize the columns of a dataframe.
    :param df: Pandas Dataframe to normalize
    :param offset: New minimum. Data will be normalized between offset and 1.
    default is 0.2 to prevent the minimum value(s) to collapse to the center
    of the plot.
    :param inv: columns that should be inverted (for example if some dimensions
    should be minimized but we want to display only "bigger is better" metrics.
    :return: The normalized DataFrame.
    """
    if inv is None:
        inv = []
    for col in inv:
        if col in df:
            df[col] = -df[col]
    df -= df.min()
    df /= df.max()
    df = df * (1 - offset) + offset
    return df


def radar_chart(full_res, multi=True, rows=None, columns=None):
    if rows is None:
        rows = full_res.index
    plot_df = full_res[columns].loc[rows]
    plot_df = normalize_df(plot_df, inv=['FLOPs', 'Mem.'])

    if multi:
        res = radar_chart_matplotlib_multi(plot_df)
    else:
        res = radar_chart_matplotlib_simple(plot_df)
    # plt.ioff()
    # res.show()
    return res


def radar_chart_matplotlib_simple(df, size=23):
    categories = [RADAR_NAMES[cat] for cat in df.columns]
    N = len(categories)

    # Create the plot
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    ax = plt.subplot(111, polar=True)

    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], categories, size=size)

    ax.set_rlabel_position(0)
    plt.yticks([.2, .4, .6, .8, 1], [".2", ".4", ".6", ".8", "1"],
               color="grey", size=size * 0.8, visible=False)
    plt.ylim(0, 1)

    # Styling
    my_palette = plt.cm.get_cmap("Set2", 6)  # len(df.index)+1)
    line_styles = ['-', '--', '-.', ':', (-2.5, (7, 2)), (0, (1, 4))]
    for i, (name, trace) in enumerate(df.iterrows()):
        values = trace.values.flatten().tolist()
        values += values[:1]
        w = 2
        if name.startswith('MNTDP'):
            w = 2.5
        c = my_palette(i)
        ax.plot(angles, values, color=c, linewidth=w,
                linestyle=line_styles[i], label=name)
        ax.fill(angles, values, color=c, alpha=0.1)

    # Tune the layout (can require lot of trials ...)
    plt.legend(loc='lower left',
               bbox_to_anchor=(-0.7, 0.8),
               prop={'size': size})

    # plt.gcf().set_size_inches(9, 5)
    plt.gcf().set_size_inches(12, 7)
    plt.tight_layout(1)
    return plt


def radar_chart_matplotlib_multi(df):
    categories = [RADAR_NAMES[cat] for cat in df.columns]
    N = len(categories)
    plt.cla()
    plt.clf()

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    num_traces = len(df.index)
    n_cols = 3
    n_rows = np.ceil(num_traces / n_cols)

    def make_spider(idx, title, trace, color):
        ax = plt.subplot(n_rows, n_cols, idx + 1, polar=True, )

        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)

        if idx == 0:
            plt.xticks(angles[:-1], categories, color='grey', size=8)
        else:
            plt.xticks(angles[:-1], [''] * len(categories), color='grey', size=1)

        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([.2, .4, .6, .8, 1], [".2", ".4", ".6", ".8", "1"],
                   color="grey", size=7, visible=idx == 0)
        plt.ylim(0, 1)

        values = trace.values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
        ax.fill(angles, values, color=color, alpha=0.4)

        plt.title(title, size=11, color=color)

    # plt.gcf().set_size_inches(1920/DPI, 1080/DPI)
    plt.gcf().set_size_inches(15, 10)

    my_palette = plt.cm.get_cmap("Set2", len(df.index))

    # Loop to plot
    for idx, (name, trace) in enumerate(df.iterrows()):
        make_spider(idx=idx, title=name, trace=trace,
                    color=my_palette(idx))

    plt.gca().get_legend()
    plt.tight_layout()
    return plt




if __name__ == '__main__':
    main()
