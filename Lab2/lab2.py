import simpy
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from scipy.stats import chisquare

from Lab2.empirical import Empirical
from Lab2.theoretical import Theoretical

# Лабораторная работа №2
# Клионский Алексей 853501

"""
n = число каналов СМО
m = количество мест в очереди
λ = интенсивность поступления заявок в СМО
μ = интенсивность обслуживания заявок в СМО
T = Время пребывания заявки в очереди

В данной лабораторной работе, исходя из условия мы используем многоканальную СМО с ограниченной очередью
и ограниченным временем ожидания в очереди
"""


class SystemMassService:
    def __init__(self, env, n, m, ny, v, l):
        self.n = n
        self.m = m
        self.env = env
        self.ny = ny
        self.v = v
        self.l = l
        self.loader = simpy.Resource(env, n)

        self.sum_list_queue_len_and_count_active_channels = []
        self.times_smo = []
        self.queue_list_lens = []
        self.queue_times = []

        self.reject_list = []
        self.serve_list = []

    def start(self, iter):
        while True:
            yield self.env.timeout(np.random.exponential(1 / self.l))
            self.env.process(iter(self))

    def waiting(self):
        yield self.env.timeout(np.random.exponential(1.0 / self.v))

    def serve(self):
        yield self.env.timeout(np.random.exponential(1.0 / self.ny))


def servicing(system: SystemMassService):
    queue_len = len(system.loader.queue)
    count_active_channels = system.loader.count
    with system.loader.request() as request:
        current_queue_len = len(system.loader.queue)
        current_count_active_channels = system.loader.count
        system.queue_list_lens.append(queue_len)
        system.sum_list_queue_len_and_count_active_channels.append(queue_len + count_active_channels)
        if current_queue_len <= system.m:
            arrival_time = system.env.now
            response = yield request | system.env.process(system.waiting())
            system.queue_times.append(system.env.now - arrival_time)
            if request in response:
                yield system.env.process(system.serve())
                system.serve_list.append(current_queue_len + current_count_active_channels)
            else:
                system.reject_list.append(current_queue_len + current_count_active_channels)
            system.times_smo.append(system.env.now - arrival_time)
        else:
            system.reject_list.append(system.n + system.m + 1)
            system.times_smo.append(0)
            system.queue_times.append(0)


class Test:
    def __init__(self, n, m, ny, v, l, time):
        self.n = n
        self.m = m
        self.ny = ny
        self.v = v
        self.l = l
        self.time = time

    def start_system(self):
        env = simpy.Environment()
        system_mass_service = SystemMassService(
            env=env, n=self.n, m=self.m, ny=self.ny, v=self.v, l=self.l
        )
        env.process(system_mass_service.start(servicing))
        env.run(until=self.time)

        theoretical = Theoretical(n=self.n, m=self.m, ny=self.ny, v=self.v, l=self.l)
        empirical = Empirical(system_mass_service=system_mass_service)

        theoretical_summary = theoretical.get_summary()
        empirical_summary = empirical.get_summary()

        names = [
            "финальные вероятности состояний",
            "абсолютная пропускная способность",
            "относительная пропускная способность",
            "вероятность отказа",
            "вероятность образования очереди",
            "среднее число заявок в СМО",
            "среднее число заявок в очереди",
            "среднее время пребывания заявки в СМО",
            "среднее время пребывания заявки в очереди",
            "среднее число занятых каналов",
        ]
        for index, value in enumerate(names):
            print("==================================")
            print(value, "\n", "Теоретическое:", theoretical_summary[index], "\n", "Эмпирическое:", empirical_summary[index])
        return theoretical_summary[0], empirical_summary[0]

    def plot(self, theoretical_probabilities, experimental_probabilities):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        ax1.set_title("Теоретические финальные вероятности")
        ax1.bar(
            range(len(theoretical_probabilities)), theoretical_probabilities
        )
        ax2.set_title("Эмпирические финальные вероятности")
        ax2.bar(
            range(len(experimental_probabilities)),
            experimental_probabilities, color='red'
        )
        plt.show()


if __name__ == "__main__":
    # test1 = Test(n=2, m=10, l=10, ny=5, v=1, time=3000)
    # p_theoretical, p_empirical = test1.start_system()
    # test1.plot(p_theoretical, p_empirical)

    test2 = Test(n=4, m=13, l=6, ny=7, v=1, time=3000)
    p_theoretical2, p_empirical2 = test2.start_system()
    test2.plot(p_theoretical2, p_empirical2)
    #
    # test3 = Test(n=1, m=15, l=30, ny=3, v=1, time=3000)
    # p_theoretical3, p_empirical3 = test3.start_system()
    # test3.plot(p_theoretical3, p_empirical3)
