import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import statsmodels.api as sm

c_list = ['#D9DEE7','blue','red']
alpha_list = [1,0.6,1]
range_c1 = 'grey'
range_c2 = 'blue'
line_c1 = 'black'
line_c2 =  'blue'
line_c3 =  'red'
alp = 0.2
font_legend = {'family': 'Arial', 'size': 13}


def lambda_wrapper_rmean(data, ws, rws=10):
    if np.count_nonzero(np.isnan(data)) == 0:
        try:
            data = np.nan_to_num(data)
            data_low = runmean(data, rws)
            if data.sum() != 0:
                lamb = run_fit_a_ar1((data - data_low), ws)
            else:
                lamb = np.full(len(data), np.nan)
        except:
            print('failed to get lambda')
            print(data - data_low)
            lamb = np.full(len(data), np.nan)
    else:
        lamb = np.full(len(data), np.nan)

    return lamb, data_low

def runmean(x, w):
    ## running mean of timeseries x with window size w
    n = x.shape[0]
    xs = np.zeros_like(x)
    for i in range(w // 2):
        xs[i] = np.nanmean(x[: i + w // 2 + 1])
    for i in range(n - w // 2, n):
        xs[i] = np.nanmean(x[i - w // 2 + 1:])

    for i in range(w // 2, n - w // 2):
        xs[i] = np.nanmean(x[i - w // 2: i + w // 2 + 1])
    return xs

def run_fit_a_ar1(x, w):
    ## calculate the restoring rate of timeseries x in running windows w
    n = x.shape[0]
    xs = np.zeros_like(x)

    for i in range(w // 2):
        xs[i] = np.nan

    for i in range(n - w // 2, n):
        xs[i] = np.nan

    for i in range(w // 2, n - w // 2):
        xw = x[i - w // 2: i + w // 2 + 1]
        xw = xw - xw.mean()  # variations in the window

        p0, p1 = np.polyfit(np.arange(xw.shape[0]), xw, 1)

        xw = xw - p0 * np.arange(xw.shape[0]) - p1  # remove linear trend

        dxw = xw[1:] - xw[:-1]

        xw = sm.add_constant(xw)
        model = sm.GLSAR(dxw, xw[:-1], rho=1)
        results = model.iterative_fit(maxiter=10)

        a = results.params[1]

        xs[i] = a
    return xs


def linear_fit_image(input_x, input_y, ax_input, alpha, color, lw=1.5, k_show=False):
    yy = input_y[~np.isnan(input_y)]
    xx = input_x[~np.isnan(input_y)]

    delta_x = xx[-1] - xx[-2]

    k, b = np.polyfit(xx, yy, deg=1)

    extend = int(len(xx) * 0.1)
    xx = np.array([xx[0] - delta_x * i for i in range(extend, 0, -1)] + list(xx) + [xx[-1] + delta_x * i for i in
                                                                                    range(1, extend, 1)])
    if k_show == True:
        ax_input.plot(xx, xx * k + b, c=color, ls='--', lw=lw, alpha=alpha, label=f'k={round(k, 3)}')
    else:
        ax_input.plot(xx, xx * k + b, c=color, ls='--', lw=lw, alpha=alpha)

def set_axis_label_value(ax,value_list,Decimal,judge='y'):
    if judge == 'x':
        ax.set_xticks(value_list)
        ax.set_xticklabels([format(i, f'.{Decimal}f') for i in value_list])
    else:
        ax.set_yticks(value_list)
        ax.set_yticklabels([format(i, f'.{Decimal}f') for i in value_list])

def change_axis_scale(ax):
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter)

def draw_gris(input_axs1,input_axs2,gris_time,gris,gris_lambda,ws_list):
    input_axs1.plot(gris_time, gris, c=line_c1, lw=2)
    for i in range(len(gris_lambda)):
        input_axs2.plot(gris_time, gris_lambda[i], ls='-', lw=2, c=c_list[i], alpha=alpha_list[i],
                    label=f'window width = {ws_list[i]} years')
        linear_fit_image(gris_time, gris_lambda[i], input_axs2, 1, c_list[i], lw=2)

    input_axs2.set_xlabel(r'Time [yr]', **font_legend)
    input_axs2.set_xlim(1855, 2020)
    input_axs2.set_ylim(-0.67,-0.25)
    # input_axs1.set_title(r'CW GrIS', **font_legend)

    input_axs1.set_ylabel(r'Ice sheet' + '\n' + 'height [a.u.]', **font_legend)
    input_axs2.legend(fontsize=11, frameon=False)

    plt.setp(input_axs1.get_xticklabels(), visible=False)
    # set_axis_label_value(input_axs1, [], 0, 'x')
    set_axis_label_value(input_axs2, [1860, 1900, 1940, 1980, 2020], 0, 'x')
    set_axis_label_value(input_axs1, [0, -20, -40, -60], 0, 'y')
    input_axs2.set_ylabel(r'$\lambda$', **font_legend)



def draw_AMOC(input_axs1,input_axs2,amoc_1_list,amoc_2_list):
    [amoc_time1, amoc_max_1, amoc_min_1, amoc_1, amoc_lab_max_1, amoc_lab_min_1,amoc_lab_1] = amoc_1_list
    [amoc_time2, amoc_max_2, amoc_min_2, amoc_2, amoc_lab_max_2, amoc_lab_min_2,amoc_lab_2] = amoc_2_list

    input_axs1.fill_between(amoc_time1, amoc_max_1, amoc_min_1, color=range_c1, alpha=alp, edgecolor=None)
    input_axs1.plot(amoc_time1, amoc_1, c=line_c1, lw=2)
    input_axs1.fill_between(amoc_time2, amoc_max_2, amoc_min_2, color=range_c2, alpha=alp, edgecolor=None)
    input_axs1.plot(amoc_time2, amoc_2, c=line_c2, lw=2)

    input_axs2.fill_between(amoc_time1, amoc_lab_max_1, amoc_lab_min_1, color=range_c1, alpha=alp, edgecolor=None,
                        label='HadCRUT5 ensemble range')
    input_axs2.plot(amoc_time1, amoc_lab_1, c=line_c1, lw=2, label='HadCRUT5 Mean', alpha=0.6)
    linear_fit_image(amoc_time1, amoc_lab_1, input_axs2, 1, line_c1, lw=1.8)
    input_axs2.fill_between(amoc_time2, amoc_lab_max_2, amoc_lab_min_2, color=range_c2, alpha=alp, edgecolor=None,
                        label='ERSSTv5 ensemble range')
    input_axs2.plot(amoc_time2, amoc_lab_2, c=line_c2, lw=2, label='ERSSTv5 Mean', alpha=0.6)
    linear_fit_image(amoc_time2, amoc_lab_2,input_axs2, 1, line_c2, lw=1.8)

    input_axs2.set_xlabel(r'Time [yr]', **font_legend)
    input_axs2.set_xlim(1846, 2024)
    # input_axs1.set_title(r'AMOC', **font_legend)
    input_axs1.set_ylabel(r'AMOC SST' + '\n' + r'fingerprint [°C]', **font_legend)

    input_axs2.legend(fontsize=11, frameon=False)
    set_axis_label_value(input_axs2, [1860, 1900, 1940, 1980, 2020], 0, 'x')
    input_axs1.set_ylim(-2, 2)
    set_axis_label_value(input_axs1, [-2, -1, 0, 1, 2], 0, 'y')
    input_axs2.set_ylim(-0.93,-0.1)

    plt.setp(input_axs1.get_xticklabels(), visible=False)
    input_axs2.set_ylabel(r'$\lambda$', **font_legend)

def draw_Amaz(input_axs1,input_axs2,amazon_list1,amazon_list2,amazon_list3):

    [amazon_time, amazon_ds,amazon_la] = amazon_list1
    [amazon_time1, amazon_single1_ds,lambda_single_1] = amazon_list2
    [amazon_time2, amazon_single2_ds, lambda_single_2] = amazon_list3


    twin_axis_position = 1.10
    ax_amaz_tw_11 = input_axs1.twinx()
    ax_amaz_tw_12 = input_axs1.twinx()
    input_axs1.plot(amazon_time, amazon_ds, ls='-', lw=2, c=line_c1, alpha=0.8)
    ax_amaz_tw_11.plot(amazon_time1, amazon_single1_ds, lw=2, c=line_c2)
    ax_amaz_tw_12.spines["right"].set_position(("axes", twin_axis_position))
    ax_amaz_tw_12.plot(amazon_time2, amazon_single2_ds, lw=2, c=line_c3)

    set_axis_label_value(ax_amaz_tw_12, [0.86,0.87,0.88,0.89], 2, 'y')
    ax_amaz_tw_12.set_ylim(0.855,0.895)

    ax_amaz_tw_11.tick_params(axis='y', labelcolor=line_c2, color=line_c2, which='major', direction='out')
    ax_amaz_tw_12.tick_params(axis='y', labelcolor=line_c3, color=line_c3, which='major', direction='out')
    ax_amaz_tw_11.spines['right'].set_color(line_c2)
    ax_amaz_tw_12.spines['right'].set_color(line_c3)

    ax_amaz_tw_21 = input_axs2.twinx()
    ax_amaz_tw_22 = input_axs2.twinx()

    ax_amaz_tw_22.spines["right"].set_position(("axes", twin_axis_position))

    l1, = input_axs2.plot(amazon_time, amazon_la, lw=2, ls='-', c=line_c1, alpha=0.8)
    l2, = ax_amaz_tw_21.plot(amazon_time1, lambda_single_1, lw=2, ls='-', c=line_c2)
    l3, = ax_amaz_tw_22.plot(amazon_time2, lambda_single_2, lw=2, ls='-', c=line_c3)

    linear_fit_image(amazon_time, amazon_la, input_axs2, 0.8, 'black', lw=2)
    linear_fit_image(amazon_time1, lambda_single_1, ax_amaz_tw_21, 1, line_c2, lw=2)
    linear_fit_image(amazon_time2, lambda_single_2, ax_amaz_tw_22, 1, line_c3, lw=2)

    ax_amaz_tw_21.tick_params(axis='y', labelcolor=line_c2, color=line_c2, which='major', direction='out')
    ax_amaz_tw_22.tick_params(axis='y', labelcolor=line_c3, color=line_c3, which='major', direction='out')
    input_axs2.legend([l1, l2, l3], ['Ku-band VOD', 'C-band VOD', 'C1-band VOD'], fontsize=11, frameon=False)

    input_axs2.set_xlabel(r'Time [yr]', **font_legend)
    input_axs2.set_xlim(1992, 2022)
    set_axis_label_value(input_axs2, [1995,2000,2005,2010,2015,2020], 0, 'x')

    # input_axs1.set_title(r'Amazon Rainforest', **font_legend)

    set_axis_label_value(input_axs1, [1.16,1.18,1.20], 2, 'y')
    set_axis_label_value(ax_amaz_tw_21, [-0.7, -0.6,-0.5], 1, 'y')
    set_axis_label_value(ax_amaz_tw_22, [-0.8,-0.75, -0.7,-0.65], 2, 'y')
    input_axs1.set_ylabel('VOD deseasoned', **font_legend)
    set_axis_label_value(input_axs2, [-1, -0.8, -0.6,-0.4], 1, 'y')
    plt.setp(input_axs1.get_xticklabels(), visible=False)

    ax_amaz_tw_21.spines['right'].set_color(line_c2)
    ax_amaz_tw_22.spines['right'].set_color(line_c3)

    ax_amaz_tw_21.set_ylim(-0.75,-0.45)
    ax_amaz_tw_22.set_ylim(-0.83,-0.63)

    input_axs2.set_ylabel(r'$\lambda$',**font_legend)

def draw_Sams(input_axs1,input_axs2,sams_time, sams1,sams2,sams_la,sams_la2):
    input_axs1.plot(sams_time, sams1, ls='-', lw=2, c=line_c1, alpha=0.8)
    ax_sams_tw_1 = input_axs1.twinx()
    ax_sams_tw_1.plot(sams_time, sams2, c=line_c2)
    ax_sams_tw_1.tick_params(axis='y', labelcolor=line_c2, color=line_c2, which='major', direction='out')

    l1, = input_axs2.plot(sams_time, sams_la, lw=2, ls='-', c=line_c1, alpha=0.8)
    ax_sams_tw_2 = input_axs2.twinx()
    l2,= ax_sams_tw_2.plot(sams_time, sams_la2, lw=2, ls='-', c=line_c2, alpha=0.8)

    linear_fit_image(sams_time, sams_la,input_axs2, 1, 'black', lw=1.8)
    linear_fit_image(sams_time, sams_la2, ax_sams_tw_2, 1, line_c2, lw=1.8)
    ax_sams_tw_2.tick_params(axis='y', labelcolor=line_c2, color=line_c2, which='major', direction='out')

    input_axs2.set_xlabel(r'Time [yr]', **font_legend)
    input_axs2.set_xlim(1978, 2022)
    set_axis_label_value(input_axs2, [1980,1990,2000,2010,2020], 0, 'x')

    # input_axs1.set_title(r'South American Monsoon', **font_legend)
    # ax_sams_tw_2.set_ylabel(r'$\lambda$ deseasoned', color=line_c2, **font_legend)
    # ax_sams_tw_1.set_ylabel('Rainfall deseasoned \n [mm/month]', color=line_c2, **font_legend)
    input_axs1.set_ylabel('Rainfall [mm/hour]', **font_legend)


    set_axis_label_value(input_axs2, [-0.48, -0.46, -0.44, -0.42, -0.4], 2, 'y')
    input_axs2.set_ylim(-0.48,-0.4)
    set_axis_label_value(ax_sams_tw_2, [-1, -0.9, -0.8, -0.7], 1, 'y')
    ax_sams_tw_2.set_ylim(-1.02, -0.65)
    set_axis_label_value(ax_sams_tw_1,[-0.1,0,0.1],1,'y')
    plt.setp(input_axs1.get_xticklabels(), visible=False)
    input_axs2.set_ylabel(r'$\lambda$', **font_legend)

    input_axs2.legend([l1, l2], ['rainfall', 'rainfall deseasoned'], fontsize=11, frameon=False)
    ax_sams_tw_2.spines['right'].set_color(line_c2)
    ax_sams_tw_1.spines['right'].set_color(line_c2)
    # change_axis_scale(input_axs2)