import numpy as np


class Theoretical:
    def __init__(self, n, m, l, ny, v):
        self.n = n
        self.m = m
        self.l = l
        self.v = v
        self.ny = ny

    def get_probabilities(self):
        """
        ρ=λ/μ
        β=v/μ
        p0=(1+ ρ/1! + ρ2/2! +...+ρn/n! + ρn/n!∗∑mi=1ρi∏il=1(n+lβ) −1
        p(k) = ρk / k! ∗ p0,(k=1,n¯)
        p(n+i) = p(n) ∗ ρi / ∏l=1(n+lβ)..pi, (i=1,m¯)
        """

        ro = self.l / self.ny
        beta = self.v / self.ny
        p_0 = (
            np.sum([(ro ** i) / np.math.factorial(i) for i in range(self.n + 1)])
            + ((ro ** self.n) / np.math.factorial(self.n))
            * (
                np.sum(
                    [
                        (ro ** i)
                        / (np.prod([self.n + l * beta for l in range(1, i + 1)]))
                        for i in range(1, self.m + 1)
                    ]
                )
            )
        ) ** -1
        probabilities = [p_0] + [
            ((ro ** k) / np.math.factorial(k)) * p_0 for k in range(1, self.n + 1)
        ]
        p_n = probabilities[-1]
        probabilities += [
            ((ro ** i) / (np.prod([self.n + l * beta for l in range(1, i + 1)]))) * p_n
            for i in range(1, self.m + 1)
        ]
        return probabilities

    @property
    def final_probabilities(self):
        return self.get_probabilities()

    # Вероятность отказа
    def get_reject(self):
        """p(r) = p(n+m)"""
        return self.final_probabilities[self.n + self.m]

    # Вероятность образования очереди
    def get_queue_probability(self):
        """p(q) = ∑i=0..m=1 p(n+i)"""
        return np.sum([self.final_probabilities[self.n + i] for i in range(self.m)])

    # Относитальная пропускная способность
    def get_relative_bandwidth(self):
        """Q = 1 - p(q)"""
        return 1 - self.get_reject()

    # Абсолютная пропускная способность
    def get_absolute_bandwidth(self):
        """A = Q * λ"""
        return self.get_relative_bandwidth() * self.l

    # Среднее число заявок в очереди
    def get_average_number_of_elements_in_queue(self):
        """L(q) = ∑i=1..m i ∗ p(n+i)"""
        return np.sum([i * self.final_probabilities[self.n + i] for i in range(1, self.m + 1)])

    # Среднее число заявок в СМО
    def get_average_number_of_elements_in_smo(self):
        """L(qs) = ∑k=1..n k ∗ p(k) + ∑i=1..m (n+i) ∗ p(n+i)"""
        return np.sum([k * self.final_probabilities[k] for k in range(1, self.n + 1)]) + np.sum(
            [(self.n + i) * self.final_probabilities[self.n + i] for i in range(1, self.m + 1)]
        )

    # Среднее время пребывания заявки в очереди
    def get_average_time_an_item_in_queue(self):
        """T(q) = L(q) / λ"""
        return self.get_reject() / self.l

    # Среднее время пребывания заявки в СМО
    def get_average_time_an_item_in_smo(self):
        """T(qs) = L(qs) / λ"""
        return self.get_average_number_of_elements_in_smo() / self.l

    # Среднее число занятых каналов
    def get_average_number_of_busy_channels(self):
        """k = Q * p"""
        ro = self.l / self.ny
        return self.get_relative_bandwidth() * ro

    def get_summary(self):
        """
        финальные вероятности состояний
        абсолютная пропускная способность
        относительная пропускная способность
        вероятность отказа
        вероятность образования очереди
        среднее число заявок в СМО
        среднее число заявок в очереди
        среднее время пребывания заявки в СМО
        среднее время пребывания заявки в очереди
        среднее число занятых каналов
        """

        return (
            self.final_probabilities,
            self.get_absolute_bandwidth(),
            self.get_relative_bandwidth(),
            self.get_reject(),
            self.get_queue_probability(),
            self.get_average_number_of_elements_in_smo(),
            self.get_average_number_of_elements_in_queue(),
            self.get_average_number_of_elements_in_smo(),
            self.get_average_time_an_item_in_queue(),
            self.get_average_number_of_busy_channels(),
        )
