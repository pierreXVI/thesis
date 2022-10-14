import matplotlib
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import scipy.special

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

    fig = plt.figure()
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
    fig.set_size_inches(6.28, 8.11)
    plt.subplots_adjust(0.00, 0.02, 1.0, 0.99)
    # plt.show()
    fig.savefig("rk_stab.png", transparent=True)


def fig_bdf():
    def r(k):
        def sub_r(z):
            coeff = np.zeros(k + 1, dtype=complex)
            for i in range(1, k + 1):
                coeff[0] += 1 / i
                coeff[i] = np.power(-1, i) * scipy.special.binom(k, i) / i
            coeff[0] -= z
            return (np.abs(np.roots(coeff)) < 1).all()

        return np.vectorize(sub_r)

    fig = plt.figure()
    c = [.8, .0, .0, 1]
    for (i, x_min, x_max, y_min, y_max) in zip((1, 2, 3, 4, 5, 6),
                                               (-2, -2, -4, -4, -10, -20),
                                               (3, 5, 8, 14, 25, 40),
                                               (-2, -3, -5, -8, -15, -30),
                                               (2, 3, 5, 8, 15, 30),
                                               ):
        res_x = 700
        res_y = 700

        x, y = np.meshgrid(np.linspace(x_min, x_max, res_x), np.linspace(y_min, y_max, res_y))

        ax = fig.add_subplot(2, 3, i)
        ax.set_aspect('equal')
        for axis in ('left', 'bottom'):
            ax.spines[axis].set_position('zero')
            ax.spines[axis].set_linewidth(2)
        for axis in ('right', 'top'):
            ax.spines[axis].set_color('none')
        ax.yaxis.tick_left()
        ax.xaxis.tick_bottom()
        rk_i = r(i)(x + 1j * y)
        ax.pcolormesh(x, y, np.ma.masked_array(rk_i, mask=1 - rk_i), cmap=clr.ListedColormap([c]), shading='auto')
        ax.plot([None], '--', c=c, lw=20)
        ax.set_title('BDF{0}'.format(i), y=-.1)
        ax.grid()

    fig.set_size_inches(19.20, 10.00)
    plt.subplots_adjust(0.00, 0.05, 1.0, 0.99, 0.05, 0.2)
    # plt.show()
    plt.savefig("images/bdf_stab.png", transparent=True)


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

    fig = plt.figure()
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

    fig.set_size_inches(8.40, 8.20)
    plt.subplots_adjust(0.00, 0.02, 1.0, 0.99)
    # plt.show()
    fig.savefig("ab_stab.png", transparent=True)


def fig_precond():
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
    ax2.set_xlabel(r'$n_{iter}$')
    ax2.set_ylabel(r'$\frac{||r_n||}{||r_0||}$', rotation=0)

    n = 200
    angle = 2 * np.arange(n) * np.pi / (n - 1)
    a = np.diagflat(2 * np.sin(angle) - 1 + 1j * np.cos(angle)) + np.random.normal(size=(n, n)) / (2 * np.sqrt(n))

    val, vec = scipy.linalg.eig(a)
    cond_a = np.linalg.cond(a)
    cond_v = np.linalg.cond(vec)
    ax1.plot(np.real(val), np.imag(val), 'x', mew=1, ms=6)

    counter = GMRESCounter()
    gmres_scipy(a, np.ones(n), callback=counter, maxiter=12)
    l1, = ax2.semilogy(counter.res, '+')

    jacobi = scipy.sparse.diags(1 / a.diagonal())

    a_prec = a @ jacobi
    val, vec = scipy.linalg.eig(a_prec)
    cond_a_pre = np.linalg.cond(a_prec)
    cond_v_pre = np.linalg.cond(vec)
    ax1.plot(np.real(val), np.imag(val), '+', mew=1, ms=6)

    counter = GMRESCounter()
    gmres_scipy(a_prec, np.ones(n), callback=counter, maxiter=12)
    l2, = ax2.semilogy(counter.res, 'x')

    leg = ax2.legend([l1, l2], ['', ''], handletextpad=16, loc='lower center', bbox_to_anchor=(-0.75, 0.0))

    ax2.text(-1, 0.035,
             r'$\begin{{aligned}}'
             r'\kappa\left(A\right) &= {0:0.2f} & \kappa\left(V\right) &= {1:0.2f} \\[10pt]'
             r'\kappa\left(A_{{pre}}\right) &= {2:0.2f} \quad& \kappa\left(V_{{pre}}\right) &= {3:0.2f} \\'
             r'\end{{aligned}}$'.format(cond_a, cond_v, cond_a_pre, cond_v_pre),
             transform=ax2.transAxes,
             zorder=leg.zorder + 1,
             fontsize=matplotlib.rcParams['legend.fontsize']
             )

    plt.show()


if __name__ == '__main__':
    # fig_rk()
    # fig_bdf()
    fig_ab()
    pass
