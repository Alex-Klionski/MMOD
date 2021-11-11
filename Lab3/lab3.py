import simpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare

# Лабораторная работа №3 (вариант 12)
# Клионский Алексей 853501

'''
Подсчитать характеристики эффективности для простейшей одноканальной СМО с
тремя местами в очереди (m = 3) 
при условиях: Х = 4 заявки/ч; to6c = 1 / (х = 0,5.2). 
ВЫЯСНИТЬ, как эти характеристики изменяются, если увеличить число мест в очереди до m = 4. 
'''

from Lab3.empirical import Empirical
from Lab3.theoretical import Theoretical

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

        theoretical = Theoretical(m=self.m, ny=self.ny, l=self.l)
        empirical = Empirical(system_mass_service=system_mass_service)

        theoretical_summary = theoretical.get_summary()
        empirical_summary = empirical.get_summary()

        names = [
            "финальные вероятности состояний",
            "абсолютная пропускная способность",
            "относительная пропускная способность",
            "вероятность отказа",
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
    # ny = 2
    # При m = 3
    # test1 = Test(n=1, m=3, l=4, ny=2, v=1, time=1000)
    # p_theoretical, p_empirical = test1.start_system()
    # test1.plot(p_theoretical, p_empirical)

    # При m =4
    test2 = Test(n=1, m=4, l=4, ny=2, v=1, time=1000)
    p_theoretical, p_empirical = test2.start_system()
    test2.plot(p_theoretical, p_empirical)

    '''
    При увеличении числа мест с 3 до 4 (m) приводит к незначительному
    увеличению абсолютной (A) и относительной (Q) пропускной способности,
    сопровождаясь увеличением среднего числа заявок в очереди и в системе,
    и времен.
    Это норм. , т.к. некоторые заявки, получающие отказ в первом варианте,
    становятся в очередь во втором.
    '''
