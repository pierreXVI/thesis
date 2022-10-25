import matplotlib.colors as clr
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import scipy.special
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'xtick.major.width': 1.5,
    'xtick.minor.width': 1,
    'ytick.major.width': 1.5,
    'ytick.minor.width': 1,
    'axes.grid': True,
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'grid.linewidth': 2,
    'legend.fontsize': 20,
    'legend.labelspacing': 1,
    'lines.markeredgewidth': 15,
    'lines.markersize': 3,
    'lines.linewidth': 3,
    'text.latex.preamble': r"\usepackage{amsmath}",
    'text.usetex': True
})


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
        def x(n):
            return np.random.random(n) + 10

        @staticmethod
        def v(n):
            return 1e-1 * (2 * np.random.random(n) - 1)

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
        ax1.legend()
        ax1.set_xlabel(r'$\varepsilon$')
        ax2.set_xlabel(r'$N$')
        ax1.set_ylabel('Relative error in\nJacobian vector product approximation', labelpad=30)

        # plt.show()
        fig1.savefig('Burgers_10.png')
        fig2.savefig('Burgers_10000000.png')


if __name__ == '__main__':
    # fig_rk()
    # fig_bdf()
    # fig_ab()
    # fig_preconditioning()
    # fig_eps()
    pass
