import matplotlib.pyplot as plt
import numpy as np
import tqdm
import scipy.stats as st
import statsmodels.api as sm
from matplotlib.patches import ConnectionPatch
from scipy.ndimage import gaussian_filter
from matplotlib import ticker

legend_size = 15


#function for simulation

def f(x, P_0, P_1=1.0,P_2=1.0):
    return -P_0 + P_1 * x - P_2 * (x**3)

def f_2(x, P_0, P_1=1.5,P_2=0.5):
    return -P_0 + P_1 * x - P_2 * (x**3)

def f_3(x, P_0, P_1=2.0,P_2=0.1):
    return -P_0 + P_1 * x - P_2 * (x**3)

def f_4(x, P_0, P_1=-4,P_2=0):
    return -P_0 + P_1 * x - P_2 * (x**3)

def find_abrupt_point(ts,delay=300):
    return np.where(ts < 0)[0][0] - delay

def de_trend(time,ts,sigma=200):
    de_trend_ts = ts[::10] - gaussian_filter(ts[::10], sigma)
    de_time = np.copy(time[::10])
    return de_time,de_trend_ts

def make_case1(fig,axs,c_list):
    npoints = 80000
    begin = -1
    end = 2.2
    P_0_list = np.linspace(begin, end, npoints)
    P_r = (end - begin) / npoints
    dt = 0.01
    t = np.arange(npoints) * dt
    seed1 = 100
    seed2 = 920
    w = 1500

    xs = np.zeros(npoints)
    xs[0] = 1.2
    ys = np.zeros(npoints)
    ys[0] = 2
    ys_c = np.zeros(npoints)
    ys_c[0] = 2

    np.random.seed(seed1)
    W1 = np.random.normal(scale=np.sqrt(dt), size=xs.size)
    np.random.seed(seed2)
    W2 = np.random.normal(scale=np.sqrt(dt), size=ys.size)

    N_r = 0.2
    coupling_intensity = 1.0

    for i in tqdm.trange(npoints - 1):
        P_0 = P_0_list[i]
        mu_0 = -0.5
        xs[i + 1] = xs[i] + f(xs[i], P_0) * dt + N_r * W1[i]
        ys[i + 1] = ys[i] + f_2(ys[i], mu_0) * dt + N_r * W2[i]

        cc = -mu_0 - coupling_intensity * (xs[i + 1])
        ys_c[i + 1] = ys_c[i] + f_2(ys_c[i], cc) * dt + N_r * W2[i]

    ts_list = [xs, ys, ys_c]

    tp_list = []
    ts_d_list = []
    ar_list = []
    time_list = []
    var_list = []
    lambda_list = []

    tp = find_abrupt_point(xs)
    for ts in ts_list:
        time_d, ts_d = de_trend(t[:tp], ts[:tp])
        ar_1 = np.roll(runac(ts_d, w),w//2)
        var = np.roll(runstd(ts_d, w)**2,w//2)

        lambda_val = np.roll(run_fit_a_ar1(ts_d, w),w//2)

        ar_list.append(ar_1)
        time_list.append(time_d)
        var_list.append(var)
        lambda_list.append(lambda_val)

    axs[0].plot(t, ts_list[0], color=c_list[0])
    axs[1].plot(t, ts_list[1], color=c_list[1], alpha=0.7, label='Independent element')
    axs[1].plot(t, ts_list[2], color=c_list[2], label='Cascading element')

    tpp = len(time_list[0])
    mk_picture(time_list[1], time_list[2], ar_list[1], ar_list[2], axs[2], c_list, tp=tpp)
    mk_picture(time_list[1], time_list[2], var_list[1], var_list[2], axs[3], c_list, tp=tpp)
    mk_picture(time_list[1], time_list[2], lambda_list[1], lambda_list[2], axs[4], c_list, tp=tpp)


    axs[0].set_ylim(-2, 2)
    axs[0].set_xlim(0,500)
    axs[1].set_ylim(-2.6, 2.6)
    axs[2].set_ylim(0.6, 0.8)
    axs[3].set_ylim(0.004, 0.007)
    axs[4].set_ylim(-0.38, -0.18)

    axs[2].set_yticks([0.6,0.7,0.8])
    axs[2].set_yticklabels(['0.6','0.7','0.8'])

    axs[4].set_yticks([-0.4,-0.3,-0.2])
    axs[4].set_yticklabels(['-0.4','-0.3','-0.2'])

    xyA = (find_abrupt_point(xs) * dt, 2)
    xyB = (find_abrupt_point(xs) * dt, -0.4)
    con = ConnectionPatch(xyA=xyA, xyB=xyB, coordsA="data", coordsB="data", axesA=axs[0], axesB=axs[4],
                           color=c_list[0], lw=1.5, alpha=1, ls='--')
    fig.add_artist(con)
    axs[4].legend(fontsize=legend_size, loc='upper left', frameon=False)

def make_case2(fig,axs,c_list):
    npoints = 80000
    begin = -1
    end = 2.2
    w = 1500

    P_0_list = np.linspace(begin, end, npoints)
    P_r = (end - begin) / npoints
    dt = 0.01

    t = np.arange(npoints) * dt

    xs = np.zeros(npoints)
    xs[0] = 1.3
    ys = np.zeros(npoints)
    ys[0] = 0.5
    ys_c = np.zeros(npoints)
    ys_c[0] = 0.5

    np.random.seed(780)
    W1 = np.random.normal(scale=np.sqrt(dt), size=xs.size)
    np.random.seed(310)
    W2 = np.random.normal(scale=np.sqrt(dt), size=ys.size)

    N_r = 0.2
    N_r_2 = 0.2
    coupling_intensity = 2

    for i in tqdm.trange(npoints - 1):
        P_0 = P_0_list[i]
        xs[i + 1] = xs[i] + f(xs[i], P_0) * dt + N_r * W1[i]
        ys[i + 1] = ys[i] + f_4(ys[i], P_0) * dt + N_r_2 * W2[i]

        cc = coupling_intensity * (np.abs(xs[i + 1]) - 1)
        ys_c[i + 1] = ys_c[i] + (f_4(ys_c[i], P_0) + cc) * dt + N_r_2 * W2[i]

    ts_list = [xs, ys, ys_c]
    tp_list = []
    ts_d_list = []
    ar_list = []
    time_list = []
    var_list = []
    lambda_list = []

    tp = find_abrupt_point(xs)
    for ts in ts_list:
        time_d, ts_d = de_trend(t[:tp], ts[:tp])
        ar_1 = np.roll(runac(ts_d, w),w//2)
        var = np.roll(runstd(ts_d, w)**2,w//2)
        lambda_val = np.roll(run_fit_a_ar1(ts_d, w),w//2)
        ar_list.append(ar_1)
        time_list.append(time_d)
        var_list.append(var)
        lambda_list.append(lambda_val)

    axs[0].plot(t, ts_list[0], color=c_list[0])
    axs[1].plot(t, ts_list[1], color=c_list[1], alpha=0.7)
    axs[1].plot(t, ts_list[2], color=c_list[2])

    tpp = len(time_list[0])
    mk_picture(time_list[1], time_list[2], ar_list[1], ar_list[2], axs[2], c_list, tp=tpp)
    mk_picture(time_list[1], time_list[2], var_list[1], var_list[2], axs[3], c_list, tp=tpp)
    mk_picture(time_list[1], time_list[2], lambda_list[1], lambda_list[2], axs[4], c_list, tp=tpp)

    xyA = (find_abrupt_point(xs) * dt, 2)
    xyB = (find_abrupt_point(xs) * dt, -0.4)
    con3 = ConnectionPatch(xyA=xyA, xyB=xyB, coordsA="data", coordsB="data", axesA=axs[0], axesB=axs[4],
                           color=c_list[0], lw=1.5, alpha=1, ls='--')
    fig.add_artist(con3)

    axs[0].set_ylim(-2, 2)
    axs[0].set_xlim(0, 500)
    axs[1].set_ylim(-2.6, 2.6)
    axs[2].set_ylim(0.62, 0.77)
    axs[2].set_yticks([0.65,0.70,0.75])
    axs[2].set_yticklabels(['0.65','0.70','0.75'])

    axs[3].set_ylim(0.0042,0.0078)
    axs[4].set_ylim(-0.4, -0.15)

    axs[4].set_yticks([-0.4,-0.3,-0.2])
    axs[4].set_yticklabels(['-0.4','-0.3','-0.2'])
    axs[4].legend(fontsize=legend_size, loc='upper left', frameon=False)

def make_case3(fig,axs,c_list):
    npoints = 80000
    begin = -1
    end = 2.2
    P_0_list = np.linspace(begin, end, npoints)
    P_r = (end - begin) / npoints
    dt = 0.01
    seed1 = 390  # 100
    seed2 = 82  # 95

    t = np.arange(npoints) * dt

    xs = np.zeros(npoints)
    xs[0] = 1
    ys = np.zeros(npoints)
    ys[0] = 2
    ys_c = np.zeros(npoints)
    ys_c[0] = 2

    np.random.seed(seed1)
    W1 = np.random.normal(scale=np.sqrt(dt), size=xs.size)
    np.random.seed(seed2)
    W2 = np.random.normal(scale=np.sqrt(dt), size=ys.size)

    N_r = 0.2
    coupling_intensity = 0.2

    for i in tqdm.trange(npoints - 1):
        P_0 = P_0_list[i]
        xs[i + 1] = xs[i] + f(xs[i], P_0) * dt + N_r * W1[i]
        ys[i + 1] = ys[i] + f_2(ys[i], P_0) * dt + N_r * W2[i]

        cc = P_0 - coupling_intensity * xs[i + 1] + 0.2
        ys_c[i + 1] = ys_c[i] + f_2(ys_c[i], cc) * dt + N_r * W2[i]

    ts_list = [xs, ys, ys_c]

    tp_list = []
    ts_d_list = []
    ar_list = []
    time_list = []
    var_list = []
    lambda_list = []

    w = 1500
    for ts in ts_list:
        tp = find_abrupt_point(ts)
        time_d, ts_d = de_trend(t[:tp], ts[:tp])
        ar_1 = np.roll(runac(ts_d, w),w//2)
        var = np.roll(runstd(ts_d, w)**2,w//2)
        lambda_val = np.roll(run_fit_a_ar1(ts_d, w),w//2)

        ar_list.append(ar_1)
        time_list.append(time_d)
        var_list.append(var)
        lambda_list.append(lambda_val)

    axs[0].plot(t, ts_list[0], color=c_list[0])
    axs[1].plot(t, ts_list[1], color=c_list[1], alpha=0.7, label='Independent element')
    axs[1].plot(t, ts_list[2], color=c_list[2], label='Cascading element')

    tpp = len(ar_list[0])
    tpp_independ = len(ar_list[1])
    mk_picture(time_list[1], time_list[2], ar_list[1], ar_list[2], axs[2], c_list, tp=tpp,
                  forward=tpp_independ - tpp)
    mk_picture(time_list[1], time_list[2], var_list[1], var_list[2], axs[3], c_list, tp=tpp,
                  forward=tpp_independ - tpp)
    mk_picture(time_list[1], time_list[2], lambda_list[1], lambda_list[2], axs[4], c_list, tp=tpp,
                  forward=tpp_independ - tpp)

    xyA = (find_abrupt_point(xs) * dt, 2)
    xyB = (find_abrupt_point(xs) * dt, -0.4)
    con1 = ConnectionPatch(xyA=xyA, xyB=xyB, coordsA="data", coordsB="data", axesA=axs[0], axesB=axs[4],
                           color=c_list[0], lw=1.5, alpha=1, ls='--')
    fig.add_artist(con1)

    axs[0].set_ylim(-2, 2)
    axs[0].set_xlim(0, 600)
    axs[1].set_ylim(-2.6, 2.6)
    axs[2].set_ylim(0.6, 0.90)
    axs[2].set_yticks([0.6,0.7,0.8,0.9])
    # axs[2].set_yticklabels(['0.6','-0.3','-0.2'])

    axs[3].set_ylim(0.003, 0.018)
    axs[4].set_ylim(-0.4, 0.05)

    axs[4].legend(fontsize=legend_size, loc='upper left', frameon=False)

def make_case4(fig,axs,c_list):
    npoints = 80000
    begin = -1
    end = 2.2
    P_0_list = np.linspace(begin, end, npoints)
    P_r = (end - begin) / npoints
    dt = 0.01
    seed1 = 350  # 100
    seed2 = 990  # 90

    t = np.arange(npoints) * dt

    xs = np.zeros(npoints)
    xs[0] = 1
    ys = np.zeros(npoints)
    ys[0] = 2
    ys_c_p = np.zeros(npoints)
    ys_c_p[0] = 2
    ys_c_n = np.zeros(npoints)
    ys_c_n[0] = 2

    np.random.seed(seed1)
    W1 = np.random.normal(scale=np.sqrt(dt), size=xs.size)
    np.random.seed(seed2)
    W2 = np.random.normal(scale=np.sqrt(dt), size=ys.size)

    N_r = 0.2
    coupling_intensity = 0.3

    for i in tqdm.trange(npoints - 1):
        P_0 = P_0_list[i]
        xs[i + 1] = xs[i] + f(xs[i], P_0) * dt + N_r * W1[i]
        ys[i + 1] = ys[i] + f_2(ys[i], P_0) * dt + N_r * W2[i]

        cc = 1 - coupling_intensity * (1 - xs[i + 1])
        ys_c_p[i + 1] = ys_c_p[i] + f_2(ys_c_p[i], P_0) * dt + cc * N_r * W2[i]

        cc = 1 + coupling_intensity * (1 - xs[i + 1])
        ys_c_n[i + 1] = ys_c_n[i] + f_2(ys_c_n[i], P_0) * dt + cc * N_r * W2[i]

    ts_list = [xs, ys, ys_c_n, ys_c_p]

    tp_list = []
    ts_d_list = []
    ar_list = []
    time_list = []
    var_list = []
    lambda_list = []

    w = 1500
    for ts in ts_list:
        tp = find_abrupt_point(ts)
        time_d, ts_d = de_trend(t[:tp], ts[:tp])
        ar_1 = np.roll(runac(ts_d, w),w//2)
        var = np.roll(runstd(ts_d, w)**2,w//2)
        lambda_val = np.roll(run_fit_a_ar1(ts_d, w),w//2)

        ar_list.append(ar_1)
        time_list.append(time_d)
        var_list.append(var)
        lambda_list.append(lambda_val)

    axs[0].plot(t, ts_list[0], color=c_list[0])
    axs[1].plot(t, ts_list[2], color=c_list[2], label='Cascading system #1')
    axs[1].plot(t, ts_list[1], color=c_list[1], alpha=0.7, label='Independent element')
    axs[1].plot(t, ts_list[3], color=c_list[3], label='Cascading system #2')

    tpp = len(ar_list[0])
    mk_picture_2(time_list[1], time_list[2], time_list[3], ar_list[1], ar_list[2], ar_list[3], axs[2], c_list,
                    tp=tpp)
    mk_picture_2(time_list[1], time_list[2], time_list[3], var_list[1], var_list[2], var_list[3], axs[3], c_list,
                    tp=tpp)
    mk_picture_2(time_list[1], time_list[2], time_list[3], lambda_list[1], lambda_list[2], lambda_list[3], axs[4],
                    c_list, tp=tpp)

    xyA = (find_abrupt_point(xs) * dt, 2)
    xyB = (find_abrupt_point(xs) * dt, -0.4)

    con = ConnectionPatch(xyA=xyA, xyB=xyB, coordsA="data", coordsB="data", axesA=axs[0], axesB=axs[4],
                           color=c_list[0], lw=1.5, alpha=1, ls='--')
    fig.add_artist(con)

    axs[0].set_ylim(-2, 2)
    axs[0].set_xlim(0, 600)
    axs[1].set_ylim(-2.6, 2.6)
    axs[2].set_ylim(0.62, 0.98)
    axs[2].set_yticks([0.7,0.8,0.9])

    axs[3].set_ylim(-0.005,0.035)
    axs[3].set_yticks([0,0.01,0.02,0.03])

    axs[4].set_ylim(-0.4,0.15)
    axs[4].legend(fontsize=legend_size, loc='upper left', frameon=False)


#EWS calculation
def runac(x, w):
    n = x.shape[0]
    xs = np.zeros_like(x)

    for i in range(w // 2):
        xs[i] = np.nan

    for i in range(n - w // 2, n):
        xs[i] = np.nan

    for i in range(w // 2, n - w // 2):
        xw = x[i - w // 2 : i + w // 2 + 1]
        xw = xw - xw.mean()
        if np.std(xw) > 0:
            lg = st.linregress(np.arange(xw.shape[0]), xw)[:]
            p0 = lg[0]
            p1 = lg[1]

            xw = xw - p0 * np.arange(xw.shape[0]) - p1
            xs[i] = np.corrcoef(xw[1:], xw[:-1])[0,1]
        else:
            xs[i] = np.nan

    return xs

def runstd(x, w):
    n = x.shape[0]
    xs = np.zeros_like(x)

    for i in range(w // 2):
        xs[i] = np.nan

    for i in range(n - w // 2, n):
        xs[i] = np.nan

    for i in range(w // 2, n - w // 2):
        xw = x[i - w // 2 : i + w // 2 + 1]
        xw = xw - xw.mean()
        if np.std(xw) > 0:
            lg = st.linregress(np.arange(xw.shape[0]), xw)[:]
            p0 = lg[0]
            p1 = lg[1]
            xw = xw - p0 * np.arange(xw.shape[0]) - p1
            xs[i] = np.std(xw)
        else:
            xs[i] = np.nan

    return xs

def run_fit_a_ar1(x, w):
  n = x.shape[0]
  xs = np.zeros_like(x)

  for i in range(w // 2):
     xs[i] = np.nan

  for i in range(n - w // 2, n):
     xs[i] = np.nan

  for i in range(w // 2, n - w // 2):
     xw = x[i - w // 2 : i + w // 2 + 1]
     xw = xw - xw.mean()

     lg = st.linregress(np.arange(xw.shape[0]), xw)[:]
     p0 = lg[0]
     p1 = lg[1]

     xw = xw - p0 * np.arange(xw.shape[0]) - p1

     dxw = xw[1:] - xw[:-1]
     xw = sm.add_constant(xw)
     model = sm.GLSAR(dxw, xw[:-1], rho=1)
     results = model.iterative_fit(maxiter=10)
     a = results.params[1]

     # xs[i] = np.log(a + 1)
     xs[i] = a
  return xs


# function for plot

def mk_fitting_x(input_x,forward,backward):
    tt = input_x[1] - input_x[0]
    extend_1 = []
    extend_2 = []

    if backward != 0:
        for i in range(1, backward, 1):
            extend_1.append(input_x[0] - i * tt)
    if forward != 0:
        for i in range(1, forward, 1):
            extend_2.append(input_x[-1] + i * tt)

    fitting_x = np.array(extend_1[::-1] + list(input_x) + extend_2)

    return fitting_x

def mk_picture(x1,x2,y,z,axx,c_list,tp=0,forward=200,backward=200):

    fitting_line_width = 2.5

    axx.plot(x1, y, c=c_list[1],lw=2.2,label='Independent element',alpha=0.7)
    axx.plot(x2, z, c=c_list[2],lw=2.2,label='Coupled element')

    if tp != 0:
        start,end = (np.where(np.isnan(y)))[0][-1]+1,tp
        xx = x1[start:end]
        yy = y[start:end]
        k, b = np.polyfit(xx,yy, 1)
        fitting_x = mk_fitting_x(xx,forward=forward,backward=backward)
        axx.plot(fitting_x, k * fitting_x + b, ls='--', c=c_list[1], lw=fitting_line_width, alpha=0.7)

        start,end = (np.where(np.isnan(z)))[0][-1]+1, tp
        xx = x2[start:end]
        yy = z[start:end]
        k, b = np.polyfit(xx,yy, 1)
        fitting_x = mk_fitting_x(xx,forward=forward,backward=backward)
        axx.plot(fitting_x, k*fitting_x+b, ls='--', c=c_list[2],lw=fitting_line_width,alpha=1)

def mk_picture_2(x1,x2,x3,y,z1,z2,axx,c_list,tp=0,forward=200,backward=200):

    fitting_line_width = 2.5

    if tp != 0:
        start, end = tp, len(y)
        k, b = np.polyfit(x1[start:end], y[start:], 1)
        fitting_x = mk_fitting_x(x1[start:end],forward=forward,backward=backward)
        axx.plot(fitting_x, k*fitting_x+b, ls='--', c=c_list[1],lw=fitting_line_width,alpha=0.7)

        start,end = tp, len(z1)
        k, b = np.polyfit(x1[start:end], z1[start:], 1)
        fitting_x = mk_fitting_x(x1[start:end], forward=forward,backward=backward)
        axx.plot(fitting_x, k*fitting_x+b, ls='--', c=c_list[2],lw=fitting_line_width,alpha=1)

        start, end = tp, len(z2)
        k, b = np.polyfit(x3[start:end], z2[start:], 1)
        fitting_x = mk_fitting_x(x3[start:end], forward=forward,backward=backward)
        axx.plot(fitting_x, k*fitting_x+b, ls='--', c=c_list[3],lw=fitting_line_width,alpha=1)

    axx.plot(x1, y, c=c_list[1],lw=2.2,label='Independent element',alpha=0.7)
    axx.plot(x2, z1, c=c_list[2],lw=2.2,label='Noise-increased element')
    axx.plot(x3, z2, c=c_list[3], lw=2.2, label='Noise-reduced element ')


def change_axis_scale(ax):
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter)

def ax_set_homemade(axx,bwith):
    axx.spines['bottom'].set_linewidth(bwith)
    axx.spines['left'].set_linewidth(bwith)
    bwith = 0
    axx.spines['top'].set_linewidth(bwith)
    axx.spines['right'].set_linewidth(bwith)
    axx.tick_params(axis='y', which='major', width=bwith, direction='out')
    axx.tick_params(axis='x', which='major', width=bwith, direction='out')

def make_subplot_label(axs):
    label_list = ['a', 'b', 'c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','s','t','u','v','w','x','y','z']
    font_set = {'fontsize': 25, 'fontweight': 1000, 'verticalalignment': 'top', 'fontfamily': 'Arial'}

    for k in range(0,5):
        for i in range(0, 4, 1):
            ax = axs[k,i]
            ax.text(-.15, 1.1, label_list[k*4+i], transform=ax.transAxes,fontfamily='Arial', fontsize=25, fontweight='bold', va='top', ha='right')
