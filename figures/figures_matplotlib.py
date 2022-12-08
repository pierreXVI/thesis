import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import scipy.special
from postprocess import bibarch, utils, jaguar_tools, pvlib

plt.rcParams.update({
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 200,
    'savefig.transparent': True,
    'text.latex.preamble': r"\usepackage{amsmath}",
    'text.usetex': True,
    'font.size': 10,
    'mathtext.fontset': 'cm',
    'font.family': 'STIXGeneral'
})


def rk_stab():
    x_min, x_max = -4, 2
    y_min, y_max = -3.5, 3.5
    res_x = 1000
    res_y = 1000

    x, y = np.meshgrid(np.linspace(x_min, x_max, res_x), np.linspace(y_min, y_max, res_y))

    def r(k):
        return np.vectorize(lambda z: sum(np.power(z, range(k + 1)) / scipy.special.factorial(range(k + 1))))

    fig = plt.figure(figsize=(3, 3.5))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    for axis in ('left', 'bottom'):
        ax.spines[axis].set_position('zero')
    for axis in ('right', 'top'):
        ax.spines[axis].set_color('none')

    handles = []
    for (i, c, label) in zip((4, 3, 2, 1),
                             ([.8, .0, .0, 1], [.0, .8, .0, 1], [.0, .0, .8, 1], [.5, .5, .5, 1]),
                             ('4th', '3rd', '2nd', '1st')):
        rk_i = abs(r(i)(x + 1j * y)) < 1
        ax.pcolormesh(x, y, np.ma.masked_array(rk_i, mask=1 - rk_i), cmap=matplotlib.colors.ListedColormap([c]),
                      shading='auto')
        handles.append(matplotlib.patches.Patch(color=c, label=r'\textrm{{ {0} order}}'.format(label)))

    ax.legend(handles=handles[::-1], loc='upper left')
    ax.grid(True)
    fig.subplots_adjust(0.00, 0.02, 1.0, 0.99)
    fig.savefig("rk_stab.png")
    # plt.show()


def ab_stab():
    x_min, x_max = -2.5, .5
    y_min, y_max = -1.25, 1.25
    res_x = 1000
    res_y = 1000

    x, y = np.meshgrid(np.linspace(x_min, x_max, res_x), np.linspace(y_min, y_max, res_y))

    def lbd(_i, n):
        coeff = np.ones(1)
        for j in range(n):
            if j == _i:
                continue
            ai, bi = 1 / (j - _i), j / (j - _i)
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
            for _i in range(1, k + 1):
                coeff[_i] = -z * lbd(_i - 1, k)
            coeff[1] -= 1
            return (np.abs(np.roots(coeff)) <= 1).all()

        return np.vectorize(sub_r)

    fig = plt.figure(figsize=(3.5, 3))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    for axis in ('left', 'bottom'):
        ax.spines[axis].set_position('zero')
    for axis in ('right', 'top'):
        ax.spines[axis].set_color('none')

    handles = []
    for (i, c) in zip((1, 2, 3, 4), ([.5, .5, .5, 1], [.0, .0, .8, 1], [.0, .8, .0, 1], [.8, .0, .0, 1],)):
        rk_i = r(i)(x + 1j * y)
        ax.pcolormesh(x, y, np.ma.masked_array(rk_i, mask=1 - rk_i), cmap=matplotlib.colors.ListedColormap([c]),
                      shading='auto')
        handles.append(matplotlib.patches.Patch(color=c, label=r'\textrm{{ k = {0} }}'.format(i)))

    ax.legend(handles=handles, loc='upper left')
    ax.grid(True)

    fig.subplots_adjust(0.00, 0.02, 1.0, 0.99)
    fig.savefig("ab_stab.png")
    # plt.show()


def bdf_stab():
    def r(k):
        def sub_r(z):
            coeff = np.zeros(k + 1, dtype=complex)
            for _i in range(1, k + 1):
                coeff[0] += 1 / _i
                coeff[_i] = np.power(-1, _i) * scipy.special.binom(k, _i) / _i
            coeff[0] -= z
            return (np.abs(np.roots(coeff)) < 1).all()

        return np.vectorize(sub_r)

    fig = plt.figure(figsize=(5.78851, 3.61782))
    for (i, y_max) in zip((1, 2, 3, 4, 5, 6), (2.5, 4, 5, 10, 15, 22)):
        x_min = -y_max / 2
        x_max = 2 * y_max

        res_x = 500
        res_y = 500
        x, y = np.meshgrid(np.linspace(x_min, x_max, res_x), np.linspace(-y_max, y_max, res_y))

        ax = fig.add_subplot(2, 3, i)
        ax.set_aspect('equal')
        for axis in ('left', 'bottom'):
            ax.spines[axis].set_position('zero')
        for axis in ('right', 'top'):
            ax.spines[axis].set_color('none')
        rk_i = r(i)(x + 1j * y)
        ax.pcolormesh(x, y, np.ma.masked_array(rk_i, mask=1 - rk_i),
                      cmap=matplotlib.colors.ListedColormap([.8, .0, .0, 1]), shading='auto')
        ax.set_title(r'\textrm{{ BDF{0} }}'.format(i), y=-.2)
        ax.grid(True)

    fig.subplots_adjust(0.00, 0.075, 1.0, 0.99, 0.05, 0.25)
    plt.savefig("bdf_stab.png")
    # plt.show()


def preconditioning():
    from scipy.sparse.linalg import gmres as gmres_scipy

    rng = np.random.RandomState(161616)

    class GMRESCounter:
        def __init__(self, res0=1):
            self.niter = 0
            self.res = [res0]

        def __call__(self, rk):
            self.res.append(rk)
            self.niter += 1

    fig = plt.figure(figsize=(5.78851, 2.5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.grid(True)
    ax2.grid(True)
    ax1.set_aspect('equal')
    ax2.set_xlabel(r'$n_{iter}$')
    ax2.set_ylabel(r'$\frac{\left\| r_n \right\|}{\left\| r_0 \right\|}$', rotation=0, labelpad=10)

    n = 200
    angle = 2 * np.arange(n) * np.pi / (n - 1)
    a = np.diagflat(2 * np.sin(angle) - 1 + 1j * np.cos(angle)) + rng.normal(size=(n, n)) / (2 * np.sqrt(n))

    val = scipy.linalg.eigvals(a)
    cond_a = np.linalg.cond(a)
    ax1.plot(np.real(val), np.imag(val), 'x', ms=3)
    counter = GMRESCounter()
    gmres_scipy(a, np.ones(n), callback=counter, maxiter=12)
    ax2.semilogy(counter.res, 'x', label=r'$\kappa\left(A\right) = {0:0.2f}$'.format(cond_a))

    a_prec = a @ scipy.sparse.diags(1 / a.diagonal())
    val = scipy.linalg.eigvals(a_prec)
    cond_a_pre = np.linalg.cond(a_prec)
    ax1.plot(np.real(val), np.imag(val), '+', ms=3)
    counter = GMRESCounter()
    gmres_scipy(a_prec, np.ones(n), callback=counter, maxiter=12)
    ax2.semilogy(counter.res, '+', label=r'$\kappa\left(A_{{pre}}\right) = {0:0.2f}$'.format(cond_a_pre))

    ax1.legend(*ax2.get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(0.5, -0.2))

    fig.subplots_adjust(0.06, 0.1, 0.99, 0.99, 0.3, 0)
    fig.savefig("preconditioning.png")
    # plt.show()


def fig_eps():
    rng = np.random.RandomState(12)

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
            return rng.random(_n) + 10

        @staticmethod
        def v(_n):
            return 1e-1 * (2 * rng.random(_n) - 1)

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
            _v = rng.random((_n, 3))
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

    with plt.rc_context({'figure.figsize': [5.78851, 2.894255], 'axes.grid': True}):
        fig1 = plt.figure()
        ax = fig1.add_subplot()
        ax.loglog(list_e, err_0, 'x', color='grey', ms=3, mew=0.5)
        ax.loglog(data[epsilons[0]][0][0], data[epsilons[0]][1][0], 'o', label=str(epsilons[0]))
        ax.legend()
        ax.set_xlabel(r'$\varepsilon$')
        ax.set_ylabel('Relative error in\nJacobian vector product approximation', labelpad=10)
        fig1.subplots_adjust(0.15, 0.15, 0.99, 0.99)

        fig2 = plt.figure()
        gs = matplotlib.gridspec.GridSpec(1, 3)
        ax1 = fig2.add_subplot(gs[0, :2])
        ax2 = fig2.add_subplot(gs[0, 2], sharey=ax1)
        ax1.loglog(list_e, err_1, 'x', color='grey', ms=3, mew=0.5)
        for eps in epsilons:
            color = ax1.loglog(*data[eps], '+-')[0].get_color()
            ax1.loglog(data[eps][0][-1], data[eps][1][-1], 'o', label=str(eps), color=color)
            ax2.loglog(list_n, data[eps][1], '+-', color=color)
            ax2.loglog(list_n[-1], data[eps][1][-1], 'o', color=color)
        with plt.rc_context({'lines.linewidth': 1}):
            utils.annotate_slope(ax2, 0.25, transpose=True, base=0.4, dy=0.9, dx=0.3)
        ax1.legend()
        ax1.set_xlabel(r'$\varepsilon$')
        ax2.set_xlabel(r'$N$')
        ax1.set_ylabel('Relative error in\nJacobian vector product approximation', labelpad=10)
        ax2.tick_params(labelleft=False)
        fig2.subplots_adjust(0.15, 0.15, 0.99, 0.99, 0.05, 0)

        fig = plt.figure()
        gs = matplotlib.gridspec.GridSpec(1, 3)
        ax1 = fig.add_subplot(gs[0, :2])
        ax2 = fig.add_subplot(gs[0, 2], sharey=ax1)
        ax1.loglog(list_e_euler, err, 'x', color='grey', ms=3, mew=0.5)
        for eps in epsilons:
            color = ax1.loglog(*data_euler[eps], '+-')[0].get_color()
            ax1.loglog(data_euler[eps][0][-1], data_euler[eps][1][-1], 'o', label=str(eps), color=color)
            ax2.loglog(list_n_euler, data_euler[eps][1], '+-', color=color)
            ax2.loglog(list_n_euler[-1], data_euler[eps][1][-1], 'o', color=color)
        with plt.rc_context({'lines.linewidth': 1}):
            utils.annotate_slope(ax2, 0.25, transpose=True, base=0.4, dy=1, dx=0.3)
        ax1.legend()
        ax1.set_xlabel(r'$\varepsilon$')
        ax2.set_xlabel(r'$N$')
        ax1.set_ylabel('Relative error in\nJacobian vector product approximation', labelpad=10)
        ax2.tick_params(labelleft=False)
        fig.subplots_adjust(0.15, 0.15, 0.99, 0.99, 0.05, 0)

        fig1.savefig('epsilon_Burgers_10.png')
        fig2.savefig('epsilon_Burgers.png')
        fig.savefig('epsilon_Euler.png')
        # plt.show()


def rae_cp():
    fig = plt.figure(figsize=[4.6, 3.5])
    ax = fig.add_subplot(111)
    ax.grid(True)
    ax.set_xlabel('$x/c$')
    ax.set_title(r'$C_p = \frac{{ P - P_\infty }}{{ \frac{{1}}{{2}} P_\infty \gamma \operatorname{{Ma}}^2 }} $')

    plotter = pvlib.Plotter()

    def cp(pressure):
        return ((pressure / 26500) - 1) * 2 / (1.4 * 0.75 ** 2)

    reader, _, _ = plotter.load_data("/scratchm/pseize/RAE_2822/BASE/INIT/ENSIGHT/archive_CHARME.surf.ins.case", ['P'])
    coord, p = pvlib.get_point_data(reader, 'P', plotter.find_blocks(reader, ['Intrados', 'Extrados']))
    ax.plot(coord[:, 0], cp(p), 'x', label='Traditional method')

    reader, _, _ = plotter.load_data("/scratchm/pseize/RAE_2822/MF/RUN_1/ENSIGHT/archive_CHARME.surf.ins.case", ['P'])
    coord, p = pvlib.get_point_data(reader, 'P', plotter.find_blocks(reader, ['Intrados', 'Extrados']))
    ax.plot(coord[:, 0], cp(p), '+', label='JFNK method')
    ax.invert_yaxis()
    ax.legend()

    fig.subplots_adjust(0.1, 0.12, 0.99, 0.88)
    fig.savefig('rae_cp.png')
    # plt.show()


def rae_coefficients():
    fig = plt.figure(figsize=[5.78851, 3.8])
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
    ax1.set_ylabel('Lift coefficient', labelpad=10)
    ax2.set_ylabel('Drag coefficient', labelpad=10)
    ax1.set_xlim(3800, 8000)
    ax1.set_ylim(0.3025197803328823, 0.3025198287464075)
    ax2.set_ylim(0.009251204600522693, 0.009251211374678542)
    ax1.tick_params(labelbottom=False)
    ax1.set_xlabel('')
    ax2.set_xlabel('Iteration number', labelpad=10)
    fig.legend(*ax1.get_legend_handles_labels())

    fig.subplots_adjust(0.1, 0.15, 0.97, 0.85, 0, 0.2)
    fig.savefig('rae_coefficients.png')
    # plt.show()


def rae_residuals():
    with plt.rc_context({'font.size': 8, 'xtick.labelsize': 6, 'ytick.labelsize': 6}):
        fig = plt.figure(figsize=[5.78851, 3])
        fig.suptitle(r"$\log_{10}$ of the $L_2$ residual norms", y=0.95)

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
            ax.set_ylabel(title)

        ax11.tick_params(labelbottom=False)
        ax12.tick_params(labelbottom=False)
        ax11.set_xlabel('')
        ax12.set_xlabel('')
        ax21.set_xlabel('Iteration number', labelpad=10)
        ax22.set_xlabel('Iteration number', labelpad=10)
        fig.legend(*ax11.get_legend_handles_labels())
        fig.align_ylabels([ax11, ax21])
        fig.align_ylabels([ax12, ax22])

        fig.subplots_adjust(0.1, 0.15, 0.99, 0.85, 0.30, 0.1)
        fig.savefig('rae_residuals.png')
        # plt.show()


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
        return -np.cos(5 * x) * np.tanh(-5 * (y - 1))

    p = 2
    sol_p = (np.array([-np.cos(np.pi * (2 * i + 1) / (2 * (p + 1))) for i in range(p + 1)]) + 1) / 2
    mesh = np.linspace(0, 1, 100)

    fig = plt.figure(figsize=[5.78851, 3.25])
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    ax1.set_xlabel('$x$', labelpad=-10)
    ax1.set_ylabel('$y$', labelpad=-10)
    ax2.set_xlabel('$x$', labelpad=-10)
    ax2.set_ylabel('$y$', labelpad=-10)
    ax1.set_xticks([0, 1, 2])
    ax1.set_yticks([0, 1, 2])
    ax2.set_xticks([0, 1, 2])
    ax2.set_yticks([0, 1, 2])
    ax1.xaxis.set_tick_params(pad=-5)
    ax2.xaxis.set_tick_params(pad=-5)
    ax1.yaxis.set_tick_params(pad=-5)
    ax2.yaxis.set_tick_params(pad=-5)
    ax1.zaxis.set_tick_params(label1On=False)
    ax2.zaxis.set_tick_params(label1On=False)
    ax1.set_title(r'$u\left(x, y\right)$', y=1, pad=15)
    ax2.set_title(r'$u_\textrm{SD}\left(x, y\right)$', y=1, pad=15)

    for dx in [0, 1]:
        for dy in [0, 1]:
            lag = Lagrange2D(sol_p + dx, sol_p + dy, np.array([[u(x, y) for x in sol_p + dx] for y in sol_p + dy]))
            ax1.plot_surface(*np.meshgrid(mesh + dx, mesh + dy),
                             np.array([[u(x, y) for x in mesh + dx] for y in mesh + dy]),
                             antialiased=False, cmap=plt.cm.coolwarm, vmin=-1, vmax=1)
            ax2.plot_surface(*np.meshgrid(mesh + dx, mesh + dy),
                             np.array([[lag(x, y) for x in mesh + dx] for y in mesh + dy]),
                             antialiased=False)

    fig.subplots_adjust(0, 0, 1, 1, 0, 0)
    fig.savefig('sd_discontinuous.png')
    # plt.show()


def sd_scheme():
    class Lagrange:
        def __init__(self, x, z):
            self.x = x
            self.z = z

        def __call__(self, x):
            if isinstance(x, (list, tuple, np.ndarray)):
                return np.array([self(_x) for _x in x])
            out = 0
            for i in range(len(self.x)):
                foo_x = np.delete(self.x, i)
                out += self.z[i] * np.prod((x - foo_x) / (self.x[i] - foo_x))
            return out

        def derivative(self, x):
            if isinstance(x, (list, tuple, np.ndarray)):
                return np.array([self.derivative(_x) for _x in x])
            out = 0
            for i in range(len(self.x)):
                val = 0
                foo = np.delete(self.x, i)
                for k in range(len(foo)):
                    val += np.prod(x - np.delete(foo, k))
                out += self.z[i] * val / np.prod(self.x[i] - foo)
            return out

    p = 2
    sol_p = (np.array([-np.cos(np.pi * (2 * i + 1) / (2 * (p + 1))) for i in range(p + 1)]) + 1) / 2
    flux_p = (np.append(-1, np.append(np.polynomial.legendre.legroots(p * [0] + [1]), 1)) + 1) / 2
    u = Lagrange(sol_p, [0.5, -1.25, -0.4])
    f = Lagrange(flux_p, [1.5, 0.1, -1.2, -0.5])
    f_l, f_r = 0, 1
    f_c = Lagrange(flux_p, [0.4, 0.1, -1.2, 0.2])

    mesh = np.linspace(0, 1, 100)

    with plt.rc_context({'font.size': 5, 'lines.markersize': 3}):
        fig = plt.figure(figsize=[5.78851, 3])
        ax0 = fig.add_subplot(231)
        ax1 = fig.add_subplot(232)
        ax2 = fig.add_subplot(233)
        ax3 = fig.add_subplot(234)
        ax4 = fig.add_subplot(235)
        ax5 = fig.add_subplot(236)

        for ax in (ax0, ax1, ax2, ax3, ax4, ax5):
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-3, 1.65)
            ax.grid(False)

        ax0.set_title("Step 0: $p$-order interpolation of the\nsolution on $p+1$ solution points", x=0.5, y=0)
        for ax in (ax0, ax1, ax2, ax3, ax4, ax5):
            ax.plot(mesh, u(mesh), 'r')
            ax.plot(sol_p, u(sol_p), 'ro')
            ax.plot([-0.1, 0], [1, 0.5], 'r')
            ax.plot([1, 1.1], [0.5, 0], 'r')

        ax1.set_title("Step 1: extrapolation of the\nsolution at $p+2$ flux points", x=0.5, y=0)
        for ax in (ax1, ax2, ax3, ax4, ax5):
            ax.plot(flux_p, u(flux_p), 'bo')

        ax2.set_title("Step 2: evaluation of the\nflux density at flux points", x=0.5, y=0)
        ax2.plot(flux_p, f(flux_p), 'bs')
        ax2.plot(0, f_l, 'ks')
        ax2.plot(1, f_r, 'ks', fillstyle='none')

        ax3.set_title("Step 3: computation of a unique\nflux density at segment end points", x=0.5, y=0)
        for ax in (ax3, ax4, ax5):
            ax.plot(flux_p, f_c(flux_p), 'bs')

        ax4.set_title("Step 4: $p\\!+\\!1$-order interpolation\nof the flux on flux points,\n"
                      "continuous between segments", x=0.5, y=0)
        for ax in (ax4, ax5):
            ax.plot(mesh, f_c(mesh), 'b')
            ax.plot([-0.1, 0], [0.7, f_c(0)], 'b')
            ax.plot([1, 1.1], [f_c(1), -0.1], 'b')

        ax5.set_title("Step 5: computation of the flux\ndivergence as a $p$-order\npolynomial at solution points",
                      x=0.5, y=0)
        ax5.plot(mesh, 0.2 * f_c.derivative(mesh) - 1, 'g')
        ax5.plot(sol_p, 0.2 * f_c.derivative(sol_p) - 1, 'go')

        fig.subplots_adjust(0.01, 0.01, 0.99, 0.99, 0, 0)
        fig.savefig('sd_scheme.png')
        # plt.show()


def sd_points():
    fig = plt.figure(figsize=[5.78851, 1.8089093749999998])
    ax = fig.add_subplot(111)
    ax.grid(False)
    ax.set_ylabel('$p$', rotation=0, labelpad=10)
    ax.yaxis.set_ticks([0, 2, 4, 6, 8, 10])

    sp = fp = None
    for p in range(12):
        sol_p = (np.array([-np.cos(np.pi * (2 * i + 1) / (2 * (p + 1))) for i in range(p + 1)]) + 1) / 2
        flux_p = (np.append(-1, np.append(np.polynomial.legendre.legroots(p * [0] + [1]), 1)) + 1) / 2
        ax.plot([0, 1], [p, p], 'k', lw=1)
        fp, = ax.plot(flux_p, 0 * flux_p + p, 'bs', ms=3)
        sp, = ax.plot(sol_p, 0 * sol_p + p, 'ro', ms=3)
    ax.set_ylim(np.flip(ax.get_ylim()))
    ax.legend([sp, fp], ['Solution points', 'Flux points'], ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1))

    fig.subplots_adjust(0.1, 0.1, 0.99, 0.8)
    fig.savefig('sd_points.png')
    # plt.show()


def covo_rk2():
    fig = plt.figure(figsize=[5.78851, 2.894255])
    ax = fig.add_subplot(111)

    for mesh in [16, 32, 64]:
        data_x, data_y = [], []
        for case in ['CFL_0.01', 'CFL_0.0129', 'CFL_0.0167', 'CFL_0.0215', 'CFL_0.0278', 'CFL_0.0359', 'CFL_0.0464',
                     'CFL_0.0599', 'CFL_0.0774', 'CFL_0.1', 'CFL_0.129', 'CFL_0.167', 'CFL_0.215', 'CFL_0.278',
                     'CFL_0.359', 'CFL_0.464', 'CFL_0.599', 'CFL_0.774', 'CFL_1']:
            data = jaguar_tools.read_error(
                os.path.join("/visu/pseize/COVO_JAGUAR/JOB_CFL_RK2/4x{0}".format(mesh), case))

            cfl = float(case.split('_')[1])
            error = data['Integral L2 error']['p'][-1]
            if np.isnan(error) or (
                    data_x and error / data_y[-1] > np.power(cfl / data_x[-1], 10)):
                break
            data_x.append(cfl)
            data_y.append(error)
        ax.loglog(data_x, data_y, 'x-', label="$N = {0}$".format(mesh))

    ax.grid(True)
    ax.set_xlabel(r'$\mathcal{N}_\textrm{CFL}$')
    ax.set_title('Pressure $L_2$ error, RK2, $p = 4$')
    with plt.rc_context({'lines.linewidth': 1}):
        utils.annotate_slope(ax, 2, dx=0.1, dy=0.5)
    ax.legend()

    fig.subplots_adjust(0.08, 0.15, 0.99, 0.9)
    fig.savefig('covo_rk2.png')
    # plt.show()


def covo_rk2_rk4():
    fig = plt.figure(figsize=[5.78851, 2.894255])
    ax = fig.add_subplot(111)

    lines_rk2 = []
    for mesh in [16, 32, 64]:
        data_x, data_y = [], []
        for case in ['CFL_0.01', 'CFL_0.0129', 'CFL_0.0167', 'CFL_0.0215', 'CFL_0.0278', 'CFL_0.0359', 'CFL_0.0464',
                     'CFL_0.0599', 'CFL_0.0774', 'CFL_0.1', 'CFL_0.129', 'CFL_0.167', 'CFL_0.215', 'CFL_0.278',
                     'CFL_0.359', 'CFL_0.464', 'CFL_0.599', 'CFL_0.774', 'CFL_1']:
            data = jaguar_tools.read_error(
                os.path.join("/visu/pseize/COVO_JAGUAR/JOB_CFL_RK2/4x{0}".format(mesh), case))

            cfl = float(case.split('_')[1])
            error = data['Integral L2 error']['p'][-1]
            if np.isnan(error) or (
                    data_x and error / data_y[-1] > np.power(cfl / data_x[-1], 10)):
                break
            data_x.append(cfl)
            data_y.append(error)
        lines_rk2.append(ax.loglog(data_x, data_y, 'x-', label="RK2, N = {0}".format(mesh))[0])

    lines_rk4 = []
    for mesh in [16, 32, 64]:
        data_x, data_y = [], []
        for case in ['CFL_0.1', 'CFL_0.129', 'CFL_0.167', 'CFL_0.215', 'CFL_0.278', 'CFL_0.359', 'CFL_0.464',
                     'CFL_0.599', 'CFL_0.774', 'CFL_1']:
            data = jaguar_tools.read_error(
                os.path.join("/visu/pseize/COVO_JAGUAR/JOB_CFL_RK4/4x{0}".format(mesh), case))

            cfl = float(case.split('_')[1])
            error = data['Integral L2 error']['p'][-1]
            if np.isnan(error) or (
                    data_x and error / data_y[-1] > np.power(cfl / data_x[-1], 10)):
                break

            data_x.append(cfl)
            data_y.append(error)
        lines_rk4.append(ax.loglog(data_x, data_y, 'x-', label="RK4, N = {0}".format(mesh))[0])
        ax.loglog([0.01, 0.1], [data_y[0], data_y[0]], 'k--')

    ax.legend([lines_rk2, lines_rk4], ['RK2', 'RK4'], loc='lower center', bbox_to_anchor=(0.8, 0.1),
              handler_map={list: matplotlib.legend_handler.HandlerTuple(ndivide=None)}, handlelength=6)
    ax.text(0.1, 0.9, r'$16 \times 16$', transform=ax.transAxes,
            bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8, 'edgecolor': '0.8'})
    ax.text(0.1, 0.5, r'$32 \times 32$', transform=ax.transAxes,
            bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8, 'edgecolor': '0.8'})
    ax.text(0.1, 0.175, r'$64 \times 64$', transform=ax.transAxes,
            bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8, 'edgecolor': '0.8'})

    ax.grid(True)
    ax.set_xlabel(r'$\mathcal{N}_\textrm{CFL}$')
    ax.set_title('Pressure $L_2$ error, $p = 4$')

    fig.subplots_adjust(0.08, 0.15, 0.99, 0.9)
    fig.savefig('covo_rk2_rk4.png')
    # plt.show()


def covo_rk():
    fig = plt.figure(figsize=[5.78851, 3.859])
    ax = fig.add_subplot(111)

    data_x, data_y = [], []
    for case in ['CFL_0.1', 'CFL_0.129', 'CFL_0.167', 'CFL_0.215', 'CFL_0.278', 'CFL_0.359', 'CFL_0.464',
                 'CFL_0.599', 'CFL_0.774', 'CFL_1']:
        data = jaguar_tools.read_error(
            os.path.join("/visu/pseize/COVO_JAGUAR/JOB_CFL_RK4/8x32", case))

        cfl = float(case.split('_')[1])
        error = data['Integral L2 error']['p'][-1]
        if np.isnan(error) or (data_x and error / data_y[-1] > np.power(cfl / data_x[-1], 10)):
            break
        data_x.append(cfl)
        data_y.append(error)
    ax.loglog(data_x, data_y, 'x-', lw=2, ms=8, label="RK4, $p = 8$, $N = 32$")
    axins = ax.inset_axes([0.65, 0.02, 0.3, 0.5])
    axins.loglog(data_x, data_y, 'x-')
    axins.set_xticklabels([], minor=True)
    axins.set_yticklabels([], minor=True)
    axins.set_xlim(0.15, 0.3)
    axins.set_ylim(4e-07, 3e-06)

    data_x, data_y = [], []
    for case in ['CFL_0.1', 'CFL_0.129', 'CFL_0.167', 'CFL_0.215', 'CFL_0.278', 'CFL_0.359', 'CFL_0.464',
                 'CFL_0.599', 'CFL_0.774', 'CFL_1']:
        data = jaguar_tools.read_error(
            os.path.join("/visu/pseize/COVO_JAGUAR/JOB_CFL_RK6LDLD/4x64", case))

        cfl = float(case.split('_')[1])
        error = data['Integral L2 error']['p'][-1]
        if np.isnan(error) or (data_x and error / data_y[-1] > np.power(cfl / data_x[-1], 10)):
            break
        data_x.append(cfl)
        data_y.append(error)
    ax.loglog(data_x, data_y, 'x-', lw=2, ms=8, label="RKo6s, $p = 4$, $N = 64$")

    data_x, data_y = [], []
    for case in ['CFL_0.1', 'CFL_0.129', 'CFL_0.167', 'CFL_0.215', 'CFL_0.278', 'CFL_0.359', 'CFL_0.464',
                 'CFL_0.599', 'CFL_0.774', 'CFL_1']:
        data = jaguar_tools.read_error(
            os.path.join("/visu/pseize/COVO_JAGUAR/JOB_CFL_RK3SHU/6x64", case))

        cfl = float(case.split('_')[1])
        error = data['Integral L2 error']['p'][-1]
        if np.isnan(error) or (
                data_x and error / data_y[-1] > np.power(cfl / data_x[-1], 10)):
            break
        data_x.append(cfl)
        data_y.append(error)
    ax.loglog(data_x, data_y, 'x-', lw=2, ms=8, label="TVDRK(3, 3), $p = 6$, $N = 64$")

    ax.indicate_inset_zoom(axins, edgecolor="black")
    utils.annotate_slope(axins, 4, dx=0.01, dy=0.8)
    utils.annotate_slope(ax, 3, dx=0.6, dy=0.6)
    utils.annotate_slope(ax, 2, dy=0.95)

    ax.grid(True)
    ax.set_xlabel(r'$\mathcal{N}_\textrm{CFL}$')
    ax.set_title('Pressure $L_2$ error', pad=25)
    legend = ax.legend(loc='upper left', bbox_to_anchor=(0, 1.1))
    shift = max([t.get_window_extent().width for t in legend.get_texts()])
    for t in legend.get_texts():
        temp_shift = shift - t.get_window_extent().width
        t.set_position((temp_shift, 0))

    fig.subplots_adjust(0.08, 0.1, 0.99, 0.85)
    fig.savefig('covo_rk.png')
    # plt.show()


def covo_exp():
    fig = plt.figure(figsize=[5.78851, 2.894255])
    ax = fig.add_subplot(111)

    data_x, data_y = [], []
    for case in ['CFL_0.1', 'CFL_0.129', 'CFL_0.167', 'CFL_0.215', 'CFL_0.278', 'CFL_0.359', 'CFL_0.464',
                 'CFL_0.599', 'CFL_0.774', 'CFL_1', 'CFL_1.29', 'CFL_1.67', 'CFL_2.15', 'CFL_2.78', 'CFL_3.59']:
        data = jaguar_tools.read_error(
            os.path.join("/visu/pseize/COVO_JAGUAR/JOB_CFL_EXPEULER_20/6x32", case))

        cfl = float(case.split('_')[1])
        error = data['Integral L2 error']['p'][-1]
        if np.isnan(error) or (data_x and error / data_y[-1] > np.power(cfl / data_x[-1], 10)):
            break
        data_x.append(cfl)
        data_y.append(error)
    ax.loglog(data_x, data_y, 'x-', label="exponential Rosenbrock--Euler")

    data_x, data_y = [], []
    for case in ['CFL_0.1', 'CFL_0.129', 'CFL_0.167', 'CFL_0.215', 'CFL_0.278', 'CFL_0.359', 'CFL_0.464',
                 'CFL_0.599', 'CFL_0.774', 'CFL_1', 'CFL_1.29', 'CFL_1.67', 'CFL_2.15', 'CFL_2.78', 'CFL_3.59']:
        data = jaguar_tools.read_error(
            os.path.join("/visu/pseize/COVO_JAGUAR/JOB_CFL_EXPRB32_20.5/6x32", case))

        cfl = float(case.split('_')[1])
        error = data['Integral L2 error']['p'][-1]
        if np.isnan(error) or (data_x and error / data_y[-1] > np.power(cfl / data_x[-1], 10)):
            break
        data_x.append(cfl)
        data_y.append(error)
    ax.loglog(data_x, data_y, 'x-', label="ExpRB32")

    data_x, data_y = [], []
    for case in ['CFL_0.1', 'CFL_0.129', 'CFL_0.167', 'CFL_0.215', 'CFL_0.278', 'CFL_0.359', 'CFL_0.464',
                 'CFL_0.599', 'CFL_0.774', 'CFL_1', 'CFL_1.29', 'CFL_1.67', 'CFL_2.15', 'CFL_2.78', 'CFL_3.59']:
        data = jaguar_tools.read_error(
            os.path.join("/visu/pseize/COVO_JAGUAR/JOB_CFL_EXPRB42_20.5/6x32", case))

        cfl = float(case.split('_')[1])
        error = data['Integral L2 error']['p'][-1]
        if np.isnan(error) or (
                data_x and error / data_y[-1] > np.power(cfl / data_x[-1], 10)):
            break
        data_x.append(cfl)
        data_y.append(error)
    ax.loglog(data_x, data_y, 'x-', label="ExpRB42")

    utils.annotate_slope(ax, 4, base=0.1, dx=0.1, dy=0.15)
    utils.annotate_slope(ax, 3, base=0.2, dx=0.15, dy=0.22, transpose=True)
    utils.annotate_slope(ax, 2, dy=0.95)

    ax.grid(True)
    ax.set_xlabel(r'$\mathcal{N}_\textrm{CFL}$')
    ax.set_title('Pressure $L_2$ error, $p = 6$, $N = 32$')
    ax.legend()

    fig.subplots_adjust(0.08, 0.15, 0.99, 0.9)
    fig.savefig('covo_exp.png')
    # plt.show()


def tgv_curves():
    fig = plt.figure(figsize=[5.78851, 5])
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)
    ax1.grid(True)
    ax2.grid(True)
    ax1.set_ylabel('Kinetic energy')
    ax2.set_ylabel('Enstrophy')
    ax1.tick_params(labelbottom=False)
    ax2.set_xlabel('$t / t_c$')
    fig.align_ylabels([ax1, ax2])
    axins = ax2.inset_axes([0.025, 0.5, 0.3, 1])
    axins.set_xlim(9.3, 9.6)
    axins.set_ylim(9.2, 9.6)
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    ax2.indicate_inset_zoom(axins, edgecolor="black")

    tabular = []

    ref = jaguar_tools.ref_taylor_green_vortex()
    ax1.plot(ref['Time'], ref['kinetic_energy'], 'k', lw=3)
    ax2.plot(ref['Time'], ref['enstrophy'], 'k', lw=3, label='Reference')
    axins.plot(ref['Time'], ref['enstrophy'], 'k')
    tabular.append(r"{0:12s} & {1:26s} & {2:23s} & {3:20s} & {4:21s} \\"
                   .format('Method', r'$\mathcal{N}_\textrm{CFL}$', r'$N_\textrm{iterations}$',
                           'Time / iteration (s)', 'Total time (hh:mm:ss)'))

    for case_path, label in zip((
            '/visu/pseize/TGV/RK6LDLD/0.28',
            '/visu/pseize/TGV/RK3SHU/0.14',
            '/tmp_user/sator/pseize/TGV/RK4SHU/0.28',
            '/visu/pseize/TGV/EXPEULER/1.4_FAST',
            '/visu/pseize/TGV/EXPEULER/2.8',
            '/visu/pseize/TGV/EXPEULER/5.6_SLOW',
            '/visu/pseize/TGV/EXPEULER/11.2_SSLOW',
            '/visu/pseize/TGV/EXPRB32/2.8',
            '/tmp_user/sator/pseize/TGV/EXPRB42/2.8',
    ), (
            'RKo6s', 'TVDRK(3, 3)', 'SSPRK(5, 4)', 'ExpEuler(10)', 'ExpEuler(20)', 'ExpEuler(40)', 'ExpEuler(80)',
            'ExpRB32', 'ExpRB42'
    )):
        data = jaguar_tools.read_integral_quantities(os.path.join(case_path, 'RUN_*'))
        ax1.plot(data['Time'], data['kinetic_energy'], lw=1)
        ax2.plot(data['Time'], data['enstrophy'], lw=1, label=label)
        axins.plot(data['Time'], data['enstrophy'], lw=1)

        with open(utils.fetch_file(os.path.join(case_path, 'RUN_1', 'input.txt'))[0]) as input_file:
            for line in input_file:
                if 'Last iteration:' in line:
                    n_iter = int(line.split(':')[1])
                    break
        t_ite = utils.fetch_slurm_stats(utils.fetch_file(os.path.join(case_path, 'TIMER', 'slurm.*.out'))[0])[0] / 1000
        t = t_ite * n_iter
        hh = t // 3600
        mm = (t - 3600 * hh) // 60
        ss = t - 3600 * hh - 60 * mm
        tabular.append(r"{0:12s} & {1:26s} & {2:23d} & {3:20.3f} & {4:>21s} \\"
                       .format(label, case_path.split('/')[-1].split('_')[0], n_iter, t_ite,
                               '{0:02.0f}:{1:02.0f}:{2:02.0f} '.format(hh, mm, ss)),
                       )

    blank = plt.Line2D([], [], c='none')
    ax2.legend(handles=[*ax2.lines[1:4], blank, *ax2.lines[4:], blank, ax2.lines[0]],
               ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.25))

    fig.subplots_adjust(0.1, 0.3, 0.99, 0.99, 0, 0.05)
    fig.savefig('tgv_curves.png')
    # print('\n'.join([tabular[0] + r' \hline', *tabular[1:3], tabular[3] + r' \hline', *tabular[4:]]))
    # plt.show()


if __name__ == '__main__':
    # rk_stab()
    # ab_stab()
    # bdf_stab()
    # preconditioning()
    # fig_eps()
    # rae_cp()
    # rae_coefficients()
    # rae_residuals()
    # sd_discontinuous()
    # sd_scheme()
    # sd_points()
    # covo_rk2()
    # covo_rk2_rk4()
    # covo_rk()
    # covo_exp()
    # tgv_curves()
    pass
