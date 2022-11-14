import matplotlib.colors as clr
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import scipy.special
from postprocess import bibarch


# plt.rcParams.update({
#     'xtick.labelsize': 15,
#     'ytick.labelsize': 15,
#     'xtick.major.width': 1.5,
#     'xtick.minor.width': 1,
#     'ytick.major.width': 1.5,
#     'ytick.minor.width': 1,
#     'axes.grid': True,
#     'axes.labelsize': 20,
#     'axes.titlesize': 20,
#     'grid.linewidth': 2,
#     'legend.fontsize': 20,
#     'legend.labelspacing': 1,
#     'lines.markeredgewidth': 15,
#     'lines.markersize': 3,
#     'lines.linewidth': 3,
#     'text.latex.preamble': r"\usepackage{amsmath}",
#     'text.usetex': True,
#     'font.size': 15
# })


def annotate_slope(axis, s, base=0.2, dx=0.0, dy=0.0, transpose=False):
    xm, xp, ym, yp = np.inf, -np.inf, np.inf, -np.inf
    for line in axis.get_lines():
        if line.get_xdata().size and line.get_ydata().size:
            xm = min(xm, np.min(line.get_xdata()))
            xp = max(xp, np.max(line.get_xdata()))
            ym = min(ym, np.min(line.get_ydata()))
            yp = max(yp, np.max(line.get_ydata()))

    line_x = np.array([np.power(xm, base) * np.power(xp, 1 - base), xp])
    line_y = np.array([ym, ym * np.power(line_x[1] / line_x[0], s)])
    if dx:
        line_x *= np.power(xm / xp, dx * (1 - base))
    if dy:
        line_y *= np.power(yp / ym, dy * (1 - base * s * np.log(xp / xm) / np.log(yp / ym)))

    axis.plot(line_x, line_y, 'k')
    if not transpose:
        axis.plot([line_x[0], line_x[1], line_x[1]], [line_y[0], line_y[0], line_y[1]], 'k-.')
        axis.annotate(1, xy=(np.sqrt(line_x[0] * line_x[1]), line_y[0]), xytext=(0, -5 - plt.rcParams['font.size']),
                      textcoords='offset pixels')
        axis.annotate(s, xy=(line_x[1], np.sqrt(line_y[0] * line_y[1])), xytext=(10, 0), textcoords='offset pixels',
                      ha='left')
    else:
        axis.plot([line_x[0], line_x[0], line_x[1]], [line_y[0], line_y[1], line_y[1]], 'k-.')
        axis.annotate(1, xy=(np.sqrt(line_x[0] * line_x[1]), line_y[1]), xytext=(0, 10), textcoords='offset pixels')
        axis.annotate(s, xy=(line_x[0], np.sqrt(line_y[0] * line_y[1])), xytext=(-10, 0), textcoords='offset pixels',
                      ha='right')


def fig_rk():
    x_min, x_max = -4, 2
    y_min, y_max = -4, 4
    res_x = 1000
    res_y = 1000

    x, y = np.meshgrid(np.linspace(x_min, x_max, res_x), np.linspace(y_min, y_max, res_y))

    def r(k):
        return np.vectorize(lambda z: sum(np.power(z, range(k + 1)) / scipy.special.factorial(range(k + 1))))

    fig = plt.figure(figsize=(6.28, 8.11))
    ax = fig.add_subplot(111)
    ax.grid(False)
    ax.set_aspect('equal')
    for axis in ('left', 'bottom'):
        ax.spines[axis].set_position('zero')
        ax.spines[axis].set_linewidth(2)
    for axis in ('right', 'top'):
        ax.spines[axis].set_color('none')
    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()

    for (i, c, label) in zip((4, 3, 2, 1),
                             ([.8, .0, .0, 1], [.0, .8, .0, 1], [.0, .0, .8, 1], [.5, .5, .5, 1]),
                             ('4th', '3rd', '2nd', '1st')):
        rk_i = abs(r(i)(x + 1j * y)) < 1
        ax.pcolormesh(x, y, np.ma.masked_array(rk_i, mask=1 - rk_i), cmap=clr.ListedColormap([c]), shading='auto')
        ax.plot([None], '--', c=c, lw=20, label=r'\textrm{{ {0} order}}'.format(label))

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left')
    ax.grid(True)
    plt.subplots_adjust(0.00, 0.02, 1.0, 0.99)
    # plt.show()
    fig.savefig("rk_stab.png", transparent=True)


def fig_bdf():
    plt.rcParams['text.latex.preamble'] = r'\usepackage{sfmath} \boldmath'

    def r(k):
        def sub_r(z):
            coeff = np.zeros(k + 1, dtype=complex)
            for i in range(1, k + 1):
                coeff[0] += 1 / i
                coeff[i] = np.power(-1, i) * scipy.special.binom(k, i) / i
            coeff[0] -= z
            return (np.abs(np.roots(coeff)) < 1).all()

        return np.vectorize(sub_r)

    fig = plt.figure(figsize=(19.20, 10.00))
    c = [.8, .0, .0, 1]
    for (i, x_min, x_max, y_min, y_max) in zip((1, 2, 3, 4, 5, 6),
                                               (-2, -2, -4, -4, -10, -20),
                                               (3, 5, 8, 14, 25, 40),
                                               (-2, -3, -5, -8, -15, -30),
                                               (2, 3, 5, 8, 15, 30),
                                               ):
        res_x = 100
        res_y = 100

        x, y = np.meshgrid(np.linspace(x_min, x_max, res_x), np.linspace(y_min, y_max, res_y))

        ax = fig.add_subplot(2, 3, i)
        ax.grid(False)
        ax.set_aspect('equal')
        for axis in ('left', 'bottom'):
            ax.spines[axis].set_position('zero')
            ax.spines[axis].set_linewidth(2)
        for axis in ('right', 'top'):
            ax.spines[axis].set_color('none')
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        ax.yaxis.tick_left()
        ax.xaxis.tick_bottom()
        rk_i = r(i)(x + 1j * y)
        ax.pcolormesh(x, y, np.ma.masked_array(rk_i, mask=1 - rk_i), cmap=clr.ListedColormap([c]), shading='auto')
        ax.plot([None], '--', c=c, lw=20)
        ax.set_title(r'\textbf{{ BDF{0} }}'.format(i), y=-.1, fontsize=30)
        ax.grid(True)

    plt.subplots_adjust(0.00, 0.05, 1.0, 0.99, 0.05, 0.2)
    # plt.show()
    plt.savefig("bdf_stab.png", transparent=True)


def fig_ab():
    x_min, x_max = -2.5, .5
    y_min, y_max = -1.5, 1.5
    res_x = 1000
    res_y = 1000

    x, y = np.meshgrid(np.linspace(x_min, x_max, res_x), np.linspace(y_min, y_max, res_y))

    def lbd(i, n):
        coeff = np.ones(1)
        for j in range(n):
            if j == i:
                continue
            ai, bi = 1 / (j - i), j / (j - i)
            new_coeff = np.zeros(len(coeff) + 1)
            new_coeff[0] = coeff[0] * ai
            for k in range(1, len(coeff)):
                new_coeff[k] = coeff[k - 1] * bi + coeff[k] * ai
            new_coeff[-1] = coeff[-1] * bi
            coeff = new_coeff
        val = 0
        for j in range(len(coeff)):
            val += coeff[-1 - j] / (j + 1)
        return val

    def r(k):
        def sub_r(z):
            coeff = np.zeros(k + 1, dtype=complex)
            coeff[0] = 1
            for i in range(1, k + 1):
                coeff[i] = -z * lbd(i - 1, k)
            coeff[1] -= 1
            return (np.abs(np.roots(coeff)) <= 1).all()

        return np.vectorize(sub_r)

    fig = plt.figure(figsize=(8.40, 8.20))
    ax = fig.add_subplot(111)
    ax.grid(False)
    ax.set_aspect('equal')
    for axis in ('left', 'bottom'):
        ax.spines[axis].set_position('zero')
        ax.spines[axis].set_linewidth(2)
    for axis in ('right', 'top'):
        ax.spines[axis].set_color('none')
    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()
    for (i, c) in zip((1, 2, 3, 4), ([.5, .5, .5, 1], [.0, .0, .8, 1], [.0, .8, .0, 1], [.8, .0, .0, 1],)):
        rk_i = r(i)(x + 1j * y)
        ax.pcolormesh(x, y, np.ma.masked_array(rk_i, mask=1 - rk_i), cmap=clr.ListedColormap([c]), shading='auto')
        ax.plot([None], '--', c=c, lw=20, label=r'\textrm{{ k = {0} }}'.format(i))
    ax.legend(loc='upper left')
    ax.grid(True)

    plt.subplots_adjust(0.00, 0.02, 1.0, 0.99)
    # plt.show()
    fig.savefig("ab_stab.png", transparent=True)


def fig_preconditioning():
    from scipy.sparse.linalg import gmres as gmres_scipy

    class GMRESCounter:
        def __init__(self, res0=1):
            self.niter = 0
            self.res = [res0]

        def __call__(self, rk):
            self.res.append(rk)
            self.niter += 1

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.set_aspect('equal')
    ax2.set_xlabel(r'$n_{iter}$', fontsize=30)
    ax2.set_ylabel(r'$\frac{\left\| r_n \right\|}{\left\| r_0 \right\|}$', rotation=0, labelpad=30, fontsize=30)

    n = 200
    angle = 2 * np.arange(n) * np.pi / (n - 1)
    a = np.diagflat(2 * np.sin(angle) - 1 + 1j * np.cos(angle)) + np.random.normal(size=(n, n)) / (2 * np.sqrt(n))

    val = scipy.linalg.eigvals(a)
    cond_a = np.linalg.cond(a)
    ax1.plot(np.real(val), np.imag(val), 'x', mew=1, ms=6)

    counter = GMRESCounter()
    gmres_scipy(a, np.ones(n), callback=counter, maxiter=12)
    l1, = ax2.semilogy(counter.res, '+', label=r'$\kappa\left(A\right) = {0:0.2f}$'.format(cond_a))

    jacobi = scipy.sparse.diags(1 / a.diagonal())

    a_prec = a @ jacobi
    val = scipy.linalg.eigvals(a_prec)
    cond_a_pre = np.linalg.cond(a_prec)
    ax1.plot(np.real(val), np.imag(val), '+', mew=1, ms=6)

    counter = GMRESCounter()
    gmres_scipy(a_prec, np.ones(n), callback=counter, maxiter=12)
    l2, = ax2.semilogy(counter.res, 'x', label=r'$\kappa\left(A_{{pre}}\right) = {0:0.2f}$'.format(cond_a_pre))

    ax2.legend(loc='lower center', bbox_to_anchor=(-0.75, -0.075))

    fig.set_size_inches(19.2, 7.23)
    plt.subplots_adjust(0.05, 0.1, 0.99, 0.99)
    # plt.show()
    fig.savefig("preconditioning.png", transparent=True)


def fig_eps():
    class EpsWP2:

        def __str__(self):
            return r'$\varepsilon_\textrm{wp}$ based on $|| \,\cdot\, ||_2$'

        def __call__(self, _x, _v):
            _eps = np.finfo(_x.dtype).eps
            norm_x = np.linalg.norm(_x)
            norm_v = np.linalg.norm(_v)
            return np.sqrt(_eps * (1 + norm_x)) / norm_v

    class EpsWPn:

        def __str__(self):
            return r'$\varepsilon_\textrm{wp}$ based on $|| \,\cdot\, ||_2 / \sqrt{N}$'

        def __call__(self, _x, _v):
            sqrt_n = np.sqrt(_x.size)
            _eps = np.finfo(_x.dtype).eps
            norm_x = np.linalg.norm(_x) / sqrt_n
            norm_v = np.linalg.norm(_v) / sqrt_n
            return np.sqrt(_eps * (1 + norm_x)) / norm_v

    class Burgers:
        @staticmethod
        def riemann(u_l, u_r):
            return np.logical_or(u_l >= 0, u_r <= 0) * np.where((u_l + u_r) > 0, u_l * u_l / 2, u_r * u_r / 2)

        @staticmethod
        def d_riemann(u_l, u_r):
            z = np.zeros_like(u_l)
            return np.logical_or(u_l >= 0, u_r <= 0) * np.where((u_l + u_r) > 0, np.stack((u_l, z)), np.stack((z, u_r)))

        @staticmethod
        def f(_x):
            y = Burgers.riemann(_x[:-1], _x[1:])
            y0 = Burgers.riemann(_x[-1], _x[0])
            return -np.diff(y, prepend=y0, append=y0) * _x.shape[0]

        @staticmethod
        def jac_matvec(_x, _v):
            d_r = Burgers.d_riemann(_x[:-1], _x[1:])
            d_r0 = Burgers.d_riemann(_x[-1], _x[0])
            y = (d_r[0] * _v[:-1] + d_r[1] * _v[1:])
            y0 = (d_r0[0] * _v[-1] + d_r0[1] * _v[0])
            return -np.diff(y, prepend=y0, append=y0) * _x.shape[0]

        @staticmethod
        def x(_n):
            return np.random.random(_n) + 10

        @staticmethod
        def v(_n):
            return 1e-1 * (2 * np.random.random(_n) - 1)

    class Euler:
        gamma = 1.4
        foo_gamma = gamma / (gamma - 1)

        @staticmethod
        def riemann(w_l, w_r):
            return np.column_stack((
                w_l[:, 0] * w_l[:, 1] + w_r[:, 0] * w_r[:, 1],
                w_l[:, 0] * w_l[:, 1] ** 2 + w_l[:, 2] + w_r[:, 0] * w_r[:, 1] ** 2 + w_r[:, 2],
                (w_l[:, 0] * w_l[:, 1] ** 2 / 2 + Euler.foo_gamma * w_l[:, 2]) * w_l[:, 1] +
                (w_r[:, 0] * w_r[:, 1] ** 2 / 2 + Euler.foo_gamma * w_r[:, 2]) * w_r[:, 1]
            )) / 2

        @staticmethod
        def d_riemann(w_l, w_r):
            z = np.zeros((w_l.shape[0],))
            o = np.ones((w_l.shape[0],))
            return np.array([
                [
                    [w_l[:, 1], w_l[:, 0], z],
                    [w_l[:, 1] ** 2, 2 * w_l[:, 0] * w_l[:, 1], o],
                    [w_l[:, 1] ** 3 / 2, 3 * w_l[:, 0] * w_l[:, 1] ** 2 / 2 + Euler.foo_gamma * w_l[:, 2],
                     Euler.foo_gamma * w_l[:, 1]]
                ], [
                    [w_r[:, 1], w_r[:, 0], z],
                    [w_r[:, 1] ** 2, 2 * w_r[:, 0] * w_r[:, 1], o],
                    [w_r[:, 1] ** 3 / 2, 3 * w_r[:, 0] * w_r[:, 1] ** 2 / 2 + Euler.foo_gamma * w_r[:, 2],
                     Euler.foo_gamma * w_r[:, 1]]
                ]]) / 2

        @staticmethod
        def f(_x):
            y = Euler.riemann(_x[:-1], _x[1:])
            y0 = Euler.riemann(_x[None, -1], _x[None, 0])
            return -np.diff(y, prepend=y0, append=y0, axis=0) * _x.shape[0]

        @staticmethod
        def jac_matvec(_x, _v):
            d_r = Euler.d_riemann(_x[:-1], _x[1:])
            d_r0 = Euler.d_riemann(_x[None, -1], _x[None, 0])
            y = np.einsum('ijk,kj->ki', d_r[0], _v[:-1]) + np.einsum('ijk,kj->ki', d_r[1], _v[1:])
            y0 = np.einsum('ijk,j->ki', d_r0[0], _v[-1]) + np.einsum('ijk,j->ki', d_r0[1], _v[0])
            return -np.diff(y, prepend=y0, append=y0, axis=0) * _x.shape[0]

        @staticmethod
        def x(_n):
            _x = np.zeros((_n, 3))
            _x[:, 0] = 1
            _x[:, 1] = 10 * np.sin(np.linspace(0, 2 * np.pi, _n, endpoint=False))
            _x[:, 2] = 1E5
            return _x

        @staticmethod
        def v(_n):
            _v = np.random.random((_n, 3))
            _v[:, 0] *= 1e-3
            _v[:, 1] *= 1e-2
            _v[:, 2] *= 1e+2
            return _v

    epsilons = (EpsWP2(), EpsWPn())

    data = {eps: ([], []) for eps in epsilons}
    list_n = np.logspace(1, 7, 7, dtype=int)
    list_e = np.logspace(-10, -1.5, 500)
    err_0 = err_1 = None
    for n in list_n:
        x = Burgers.x(n)
        v = Burgers.v(n)
        f0 = Burgers.f(x)
        ref = Burgers.jac_matvec(x, v)
        for eps in epsilons:
            e = eps(x, v)
            er = np.linalg.norm(ref - (Burgers.f(x + e * v) - f0) / e) / np.linalg.norm(ref)
            data[eps][0].append(e)
            data[eps][1].append(er)
        if n == list_n[0]:
            err_0 = np.array(
                [np.linalg.norm(ref - (Burgers.f(x + e * v) - f0) / e) for e in list_e]) / np.linalg.norm(ref)
        elif n == list_n[-1]:
            err_1 = np.array(
                [np.linalg.norm(ref - (Burgers.f(x + e * v) - f0) / e) for e in list_e]) / np.linalg.norm(ref)

    data_euler = {eps: ([], []) for eps in epsilons}
    list_n_euler = np.logspace(1, 7, 7, dtype=int)
    list_e_euler = np.logspace(-13, -1, 500)
    x = v = f0 = ref = None
    for n in list_n_euler:
        x = Euler.x(n)
        v = Euler.v(n)
        f0 = Euler.f(x)
        ref = Euler.jac_matvec(x, v)
        for eps in epsilons:
            e = eps(x, v)
            er = np.linalg.norm(ref - (Euler.f(x + e * v) - f0) / e) / np.linalg.norm(ref)
            data_euler[eps][0].append(e)
            data_euler[eps][1].append(er)
    err = np.array([np.linalg.norm(ref - (Euler.f(x + e * v) - f0) / e) for e in list_e_euler]) / np.linalg.norm(ref)

    with plt.rc_context({'figure.figsize': [12, 6], 'lines.markeredgewidth': 1, 'lines.markersize': 15}):
        fig1 = plt.figure()
        ax = fig1.add_subplot()
        ax.loglog(list_e, err_0, 'x', ms=3, color='grey')
        ax.loglog(data[epsilons[0]][0][0], data[epsilons[0]][1][0], 'o', label=str(epsilons[0]))
        ax.legend()
        ax.set_xlabel(r'$\varepsilon$')
        ax.set_ylabel('Relative error in\nJacobian vector product approximation', labelpad=30)

        fig2 = plt.figure()
        gs = gridspec.GridSpec(1, 3)
        ax1 = fig2.add_subplot(gs[0, :2])
        ax2 = fig2.add_subplot(gs[0, 2], sharey=ax1)
        ax1.loglog(list_e, err_1, 'x', ms=3, color='grey')
        for eps in epsilons:
            color = ax1.loglog(*data[eps], '+-', ms=8, mew=2)[0].get_color()
            ax1.loglog(data[eps][0][-1], data[eps][1][-1], 'o', label=str(eps), color=color)
            ax2.loglog(list_n, data[eps][1], '+-', ms=8, mew=2, color=color)
            ax2.loglog(list_n[-1], data[eps][1][-1], 'o', color=color)
        annotate_slope(ax2, 0.25, transpose=True, base=0.4, dy=0.9, dx=0.3)
        ax1.legend()
        ax1.set_xlabel(r'$\varepsilon$')
        ax2.set_xlabel(r'$N$')
        ax1.set_ylabel('Relative error in\nJacobian vector product approximation', labelpad=30)
        fig1.savefig('epsilon_Burgers_10.png')
        fig2.savefig('epsilon_Burgers.png')

        fig = plt.figure()
        gs = gridspec.GridSpec(1, 3)
        ax1 = fig.add_subplot(gs[0, :2])
        ax2 = fig.add_subplot(gs[0, 2], sharey=ax1)
        ax1.loglog(list_e_euler, err, 'x', ms=3, color='grey')
        for eps in epsilons:
            color = ax1.loglog(*data_euler[eps], '+-', ms=8, mew=2)[0].get_color()
            ax1.loglog(data_euler[eps][0][-1], data_euler[eps][1][-1], 'o', label=str(eps), color=color)
            ax2.loglog(list_n_euler, data_euler[eps][1], '+-', ms=8, mew=2, color=color)
            ax2.loglog(list_n_euler[-1], data_euler[eps][1][-1], 'o', color=color)
        annotate_slope(ax2, 0.25, transpose=True, base=0.4, dy=1, dx=0.3)
        ax1.legend()
        ax1.set_xlabel(r'$\varepsilon$')
        ax2.set_xlabel(r'$N$')
        ax1.set_ylabel('Relative error in\nJacobian vector product approximation', labelpad=30)
        fig.savefig('epsilon_Euler.png')

        # plt.show()


def rae_coefficients():
    fig = plt.figure(figsize=[16, 9])
    fig.suptitle(r"Aerodynamic coefficients")

    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)

    ax1.grid(True)
    ax2.grid(True)
    plotter = bibarch.HistoPlotter(ax1, ('LIMITES', ['Extrados', 'Intrados'], "C_lift"), 'ITER',
                                   "/scratchm/pseize/RAE_2822")
    plotter.plot('BASE/INIT', 'k', label='Initialisation')
    plotter.plot('BASE/RUN_1', label='Traditional method')
    plotter.plot('MF/RUN_1', label='JFNK method')
    plotter = bibarch.HistoPlotter(ax2, ('LIMITES', ['Extrados', 'Intrados'], "C_drag"), 'ITER',
                                   "/scratchm/pseize/RAE_2822")
    plotter.plot('BASE/INIT', 'k', label='Initialisation')
    plotter.plot('BASE/RUN_1', label='Traditional method')
    plotter.plot('MF/RUN_1', label='JFNK method')

    ax1.set_title('')
    ax2.set_title('')
    ax1.set_ylabel('Drag coefficient', labelpad=15)
    ax2.set_ylabel('Lift coefficient', labelpad=15)
    ax1.set_xlim(3800, 8000)
    ax1.set_ylim(0.3025197803328823, 0.3025198287464075)
    ax2.set_ylim(0.009251204600522693, 0.009251211374678542)
    ax1.tick_params(labelbottom=False)
    ax1.set_xlabel('')
    ax2.set_xlabel('Iteration number', labelpad=20)
    fig.legend(*ax1.get_legend_handles_labels())
    # plt.show()
    fig.savefig('rae_coefficients.png')


def rae_residuals():
    fig = plt.figure(figsize=[16, 7])
    fig.suptitle(r"$\log_{10}$ of the $L_2$ residual norms", y=0.94)

    ax11 = fig.add_subplot(221)
    ax12 = fig.add_subplot(222)
    ax21 = fig.add_subplot(223, sharex=ax11)
    ax22 = fig.add_subplot(224, sharex=ax12)

    for ax, tag, title in zip((ax11, ax12, ax21, ax22),
                              ('RhoV_x', 'RhoV_y', 'RhoEtot', 'RhoNuTilde'),
                              ("$x$ momentum", "$y$ momentum", "Energy", "Turbulent viscosity")):
        ax.grid(True)
        plotter = bibarch.HistoPlotter(ax, ('RESIDUS', 'MOYENS', tag), 'ITER', "/scratchm/pseize/RAE_2822")
        plotter.plot('BASE/INIT', 'k', label='Initialisation')
        plotter.plot('BASE/RUN_1', label='Traditional method')
        plotter.plot('MF/RUN_1', label='JFNK method')
        ax.set_title('')
        ax.set_ylabel(title, labelpad=15)

    ax11.tick_params(labelbottom=False)
    ax12.tick_params(labelbottom=False)
    ax11.set_xlabel('')
    ax12.set_xlabel('')
    ax21.set_xlabel('Iteration number', labelpad=20)
    ax22.set_xlabel('Iteration number', labelpad=20)
    fig.legend(*ax11.get_legend_handles_labels())
    plt.subplots_adjust(0.08, 0.15, 0.99, 0.85, 0.30, 0.05)
    # plt.show()
    fig.savefig('rae_residuals.png')


def sd_discontinuous():
    class Lagrange2D:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z
            self._buf_x = np.zeros_like(self.x)
            self._buf_y = np.zeros_like(self.y)

        def __call__(self, x, y):
            for i in range(len(self.x)):
                foo_x = np.delete(self.x, i)
                self._buf_x[i] = np.prod((x - foo_x) / (self.x[i] - foo_x))
            for j in range(len(self.y)):
                foo_y = np.delete(self.y, j)
                self._buf_y[j] = np.prod((y - foo_y) / (self.y[j] - foo_y))
            out = 0
            for i in range(len(self.x)):
                for j in range(len(self.y)):
                    out += self.z[j, i] * self._buf_x[i] * self._buf_y[j]
            return out

    def u(x, y):
        return np.sin(2 * x) + np.exp(np.cos(3 * y))

    p = 2
    sol_p = (np.array([-np.cos(np.pi * (2 * i + 1) / (2 * (p + 1))) for i in range(p + 1)]) + 1) / 2

    linspace = np.linspace(0, 1, 100)

    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # ax.plot_surface(*np.meshgrid(linspace, linspace), np.array([[u(x, y) for x in linspace] for y in linspace]),
    #                 antialiased=False)
    # ax.plot_surface(*np.meshgrid(linspace+1, linspace), np.array([[u(x, y) for x in linspace+1] for y in linspace]),
    #                 antialiased=False)

    for dx in [0, 1]:
        for dy in [0, 1]:
            lag = Lagrange2D(sol_p + dx, sol_p + dy, np.array([[u(x, y) for x in sol_p + dx] for y in sol_p + dy]))
            ax1.plot_surface(*np.meshgrid(linspace + dx, linspace + dy),
                             np.array([[u(x, y) for x in linspace + dx] for y in linspace + dy]),
                             antialiased=False, cmap=plt.cm.coolwarm, vmin=-0.3889049438766065, vmax=3.718269521488654)
            ax2.plot_surface(*np.meshgrid(linspace + dx, linspace + dy),
                             np.array([[lag(x, y) for x in linspace + dx] for y in linspace + dy]),
                             # antialiased=False)
                             antialiased=False, cmap=plt.cm.coolwarm, vmin=-0.3889049438766065, vmax=3.718269521488654)

    plt.show()


def sd_scheme():
    fig = plt.figure()
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)
    ax1.grid(False)
    ax2.grid(False)
    ax3.grid(False)
    ax4.grid(False)
    ax5.grid(False)
    ax6.grid(False)

    p = 2
    sol_p = (np.array([-np.cos(np.pi * (2 * i + 1) / (2 * (p + 1))) for i in range(p + 1)]) + 1) / 2
    flux_p = (np.append(-1, np.append(np.polynomial.legendre.legroots(p * [0] + [1]), 1)) + 1) / 2
    linspace = np.linspace(0, 1, 100)

    def u(x):
        return 7 * x * x - 8 * x + 1

    def f(y):
        return y * y - 0.25

    ax1.plot(linspace, u(linspace), 'r')
    ax2.plot(linspace, u(linspace), 'r')
    ax3.plot(linspace, u(linspace), 'r')
    ax4.plot(linspace, u(linspace), 'r')
    ax5.plot(linspace, u(linspace), 'r')
    ax6.plot(linspace, u(linspace), 'r')
    ax1.plot(sol_p, u(sol_p), 'ro')
    ax2.plot(sol_p, u(sol_p), 'ro')
    ax3.plot(sol_p, u(sol_p), 'ro')
    ax4.plot(sol_p, u(sol_p), 'ro')
    ax5.plot(sol_p, u(sol_p), 'ro')
    ax6.plot(sol_p, u(sol_p), 'ro')

    ax2.plot(flux_p, u(flux_p), 'bo')
    ax3.plot(flux_p, u(flux_p), 'bo')
    ax4.plot(flux_p, u(flux_p), 'bo')
    ax5.plot(flux_p, u(flux_p), 'bo')
    ax6.plot(flux_p, u(flux_p), 'bo')

    ax3.plot(flux_p, f(u(flux_p)), 'bs')
    ax4.plot(flux_p, [0.25, *f(u(flux_p[1:-1])), -0.25], 'bs')
    ax5.plot(flux_p, [0.25, *f(u(flux_p[1:-1])), -0.25], 'bs')
    ax6.plot(flux_p, [0.25, *f(u(flux_p[1:-1])), -0.25], 'bs')

    class Lagrange:
        def __init__(self, x, z):
            self.x = x
            self.z = z
            self._buf_x = np.zeros_like(self.x)

        def __call__(self, x):
            out = 0
            for i in range(len(self.x)):
                foo_x = np.delete(self.x, i)
                out += self.z[i] * np.prod((x - foo_x) / (self.x[i] - foo_x))
            return out

        def derivative(self, x):
            out = 0
            for i in range(len(self.x)):
                val = 0
                foo = np.delete(self.x, i)
                for k in range(len(foo)):
                    val += np.prod(x - np.delete(foo, k))
                out += self.z[i] * val / np.prod(self.x[i] - foo)
            return out

    l = Lagrange(flux_p, [0.25, *f(u(flux_p[1:-1])), -0.25])
    ax5.plot(linspace, [l(x) for x in linspace], 'b')
    ax6.plot(linspace, [l(x) for x in linspace], 'b')

    ax6.plot(linspace, [0.1*l.derivative(x) for x in linspace], 'g')

    plt.show()


if __name__ == '__main__':
    # fig_rk()
    # fig_bdf()
    # fig_ab()
    # fig_preconditioning()
    # fig_eps()
    # rae_coefficients()
    # rae_residuals()
    # sd_discontinuous()
    sd_scheme()
    pass
