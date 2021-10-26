import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import scipy.stats as sta


# Лабораторная работа №1
# Клионский Алексей 853501


class DiscreteSVGenerator:
    def __init__(self, matrix, x_values, y_values):
        self.x_values = x_values
        self.y_values = y_values

        assert abs(np.sum(matrix) - 1.) < 1e-9

        self.row_partial_sums = np.sum(matrix, axis=1)

        self.row_sums_cumsum = np.cumsum(self.row_partial_sums)
        self.row_cumsums = np.cumsum(matrix, axis=1) / self.row_partial_sums.reshape(-1, 1)

    def get_item(self, num):
        row = np.searchsorted(self.row_sums_cumsum, num[0])
        column = np.searchsorted(self.row_cumsums[row], num[1])
        return row, column

    def __iter__(self):
        return self

    def __next__(self):
        num = np.random.uniform(size=2)
        return self.get_item(num)


#  Эмпирическая матрица распределения

def empiric_probability_matrix(matrix, x_values, y_values, n=1000):
    gen = DiscreteSVGenerator(matrix, x_values, y_values)
    empiric_matrix = np.zeros(matrix.shape)
    for i in range(n):
        empiric_matrix[next(gen)] += 1
    empiric_matrix /= np.sum(empiric_matrix)
    return empiric_matrix


matrix = np.array([
    [0.021983033985187334, 0.14709638987880597, 0.17484991745242057, 0.07118442113724227],
    [0.0368898846050644, 0.33786296642833963, 0.011532421452695715, 0.059387608341158686],
    [0.034094261530760894, 0.054284823798560575, 0.04244115008519647, 0.008393121304567371],
])
x_values = np.array([4., 5., 9.])
y_values = np.array([2., 4., 6., 8.])


def plot_theoretical_and_empirical_probability_matrices(theoretical, empirical):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey='all')
    ax1.set_title('Теоретическая матрица распределения')
    ax2.set_title('Эмпирическая матрица распределения')
    sn.heatmap(theoretical, ax=ax1)
    sn.heatmap(empirical, ax=ax2)
    plt.show()


empiric_matrix = empiric_probability_matrix(matrix, x_values, y_values)

plot_theoretical_and_empirical_probability_matrices(matrix, empiric_matrix)


# Гистограммы


def plot_histograms(matrix, x_values, y_values, amount=1000):
    gen = DiscreteSVGenerator(matrix, x_values, y_values)
    samples = [next(gen) for _ in range(amount)]

    x_samples = [x_values[sample[0]] for sample in samples]
    x_values, x_counts = np.unique(x_samples, return_counts=True)

    y_samples = [y_values[sample[1]] for sample in samples]
    y_values, y_counts = np.unique(y_samples, return_counts=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))

    ax1.set_title('Эмпирическая гистограмма вектора X')
    ax1.bar(x_values, x_counts / len(x_samples), tick_label=x_values)

    ax2.set_title('Эмпирическая гистограмма вектора Y')
    ax2.bar(y_values, y_counts / len(y_samples), tick_label=y_values)

    plt.show()


plot_histograms(matrix, x_values, y_values)


# Точечные и интервальные оценки математического ожидания

def expected_values(matrix, x_values, y_values):
    return x_values @ np.sum(matrix, axis=1), y_values @ np.sum(matrix, axis=0)


def calculate_theoretical_and_empirical_expected_values(matrix, x_values, y_values, amount=1000):
    empirical_matrix = empiric_probability_matrix(matrix, x_values, y_values, amount)
    theoretical, empirical = expected_values(matrix, x_values, y_values), expected_values(empirical_matrix, x_values, y_values)
    print(f'Теоретическое:\nM[x] = {theoretical[0]}, M[y] = {theoretical[1]}')
    print(f'Эмпирическое:\nM[x] = {empirical[0]}, M[y] = {empirical[1]}')


# calculate_theoretical_and_empirical_expected_values(matrix, x_values, y_values)


def calculate_intervals_expectation(values, n=1000, confidence_level=0.95):
    normal_quantile = sta.norm.ppf((1 + confidence_level) / 2)

    sample_mean = np.mean(values)
    sample_var = np.var(values, ddof=1)

    return (sample_mean - np.sqrt(sample_var / n) * normal_quantile, sample_mean + np.sqrt(sample_var / n) * normal_quantile)


def interval_for_x_and_y(matrix, x_values, y_values, amount=1000):
    gen = DiscreteSVGenerator(matrix, x_values, y_values)
    samples = np.array([next(gen) for _ in range(amount)])

    x_samples = [x_values[sample[0]] for sample in samples]
    y_samples = [y_values[sample[1]] for sample in samples]

    interval_for_x = calculate_intervals_expectation(x_samples, amount)
    print('Доверительный интервал для оценки математического ожидания X: ', interval_for_x)

    interval_for_y = calculate_intervals_expectation(y_samples, amount)
    print('Доверительный интервал для оценки математического ожидания Y: ', interval_for_y)


interval_for_x_and_y(matrix, x_values, y_values)


# Точечные и интервальные оценки дисперсии

def variance(matrix, x_values, y_values):
    m_x, m_y = expected_values(matrix, x_values, y_values)
    return np.square(x_values) @ np.sum(matrix, axis=1) - m_x ** 2, np.square(y_values) @ np.sum(matrix, axis=0) - m_y ** 2


def calculate_theoretical_and_empirical_variance(matrix, x_values, y_values, amount=1000):
    empirical_matrix = empiric_probability_matrix(matrix, x_values, y_values, amount)
    theoretical, empirical = variance(matrix, x_values, y_values), variance(empirical_matrix, x_values, y_values)

    print(f'Теоретическое:\nD[x] = {theoretical[0]}, D[y] = {theoretical[1]}')
    print(f'Эмпирическое:\nD[x] = {empirical[0]}, D[y] = {empirical[1]}')


calculate_theoretical_and_empirical_variance(matrix, x_values, y_values)


def calculate_interval_variance(values, n, confidence_level=0.95):
    sv_var = np.var(values, ddof=1)
    chi_mass = sta.chi2(n - 1)
    array = chi_mass.rvs(100000)
    temp = sta.mstats.mquantiles(array, prob=[(1 - confidence_level) / 2, (1 + confidence_level) / 2])
    xi_minus = temp[1]
    xi_plus = temp[0]
    return ((n - 1) * sv_var / xi_minus, (n - 1) * sv_var / xi_plus)


def interval_for_x_and_y_variance(matrix, x_values, y_values, amount=1000):
    gen = DiscreteSVGenerator(matrix, x_values, y_values)
    samples = np.array([next(gen) for _ in range(amount)])
    x_samples = [x_values[sample[0]] for sample in samples]
    y_samples = [y_values[sample[1]] for sample in samples]
    dov_interval_for_x = calculate_interval_variance(x_samples, amount)
    print('Доверительный интервал для оценки дисперсии X: ', dov_interval_for_x)

    dov_interval_for_y = calculate_interval_variance(y_samples, amount)
    print('Доверительный интервал для оценки дисперсии Y: ', dov_interval_for_y)


interval_for_x_and_y_variance(matrix, x_values, y_values)


# Коэффициент корреляции


def covariation(matrix, x_values, y_values):
    return x_values @ matrix @ y_values - np.prod(expected_values(matrix, x_values, y_values))


def correlation_coeff(matrix, x_values, y_values, amount=1000):
    gen = DiscreteSVGenerator(matrix, x_values, y_values)
    samples = np.array([next(gen) for _ in range(amount)])
    d1 = np.var([x[0] for x in samples], ddof=1)
    d2 = np.var([x[1] for x in samples], ddof=1)
    return covariation(matrix, x_values, y_values) / np.sqrt(np.prod((d1, d2)))


def calculate_empiric_and_theoretic(matrix, x, y, amount=1000):
    empirical_matrix = empiric_probability_matrix(matrix, x_values, y_values, amount)
    print('Теоретическая ковариация = ', covariation(matrix, x, y))
    print('Эмпиричекая ковариация = ', covariation(empirical_matrix, x, y))
    print('Теоретический коэффициент корреляция = ', correlation_coeff(matrix, x_values, y_values))
    print('Эмпирический коэффициент корреляция = ', correlation_coeff(empirical_matrix, x_values, y_values))


calculate_empiric_and_theoretic(matrix, x_values, y_values)
