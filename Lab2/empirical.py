import numpy as np


class Empirical:
    def __init__(self, system_mass_service: 'SystemMassService'):
        self.system_mass_service = system_mass_service

    # Финальные вероятности
    def get_probabilities(self):
        reject_and_serve_array = np.array(
            self.system_mass_service.reject_list + self.system_mass_service.serve_list
        )
        return [
            (len(reject_and_serve_array[reject_and_serve_array == i]) / len(reject_and_serve_array))
            for i in range(
                1, self.system_mass_service.n + self.system_mass_service.m + 2
            )
        ]

    # Вероятность образования очереди
    def get_queue_probability(self):
        reject_and_serve_array = np.array(
            self.system_mass_service.reject_list + self.system_mass_service.serve_list
        )
        return np.sum(
            [
                (len(reject_and_serve_array[reject_and_serve_array == i]) / len(reject_and_serve_array))
                for i in range(
                    1, self.system_mass_service.n + self.system_mass_service.m + 2
                )
                if i > self.system_mass_service.n
                and i < self.system_mass_service.n + self.system_mass_service.m + 1
            ]
        )

    # Вероятность отказа
    def get_reject(self):
        reject_array = np.array(self.system_mass_service.reject_list)
        return len(
            reject_array[reject_array == self.system_mass_service.n + self.system_mass_service.m + 1]
        ) / len(reject_array)

    # Относительная пропускная способность
    def get_relative_bandwidth(self):
        """Q = p(обс) = 1 - p(откз)"""
        return 1 - self.get_reject()

    # Абсолютная пропускная способность
    def get_absolute_bandwidth(self):
        """A = Q * λ"""
        return self.get_relative_bandwidth() * self.system_mass_service.l

    # Среднее число заявок в очереди
    def get_average_number_of_elements_in_queue(self):
        return np.array(self.system_mass_service.queue_list_lens).mean()

    # Среднее число заявок в СМО
    def get_average_number_of_elements_in_smo(self):
        return np.array(self.system_mass_service.sum_list_queue_len_and_count_active_channels).mean()

    # Среднее время пребывания заявки в СМО
    def get_average_time_an_item_in_smo(self):
        return np.array(self.system_mass_service.times_smo).mean()

    # Среднее время пребывания заявки в очереди
    def get_average_time_an_item_in_queue(self):
        return np.array(self.system_mass_service.queue_times).mean()

    # Среднее число активных каналов
    def get_average_number_of_busy_channels(self):
        return (
            self.get_relative_bandwidth()
            * self.system_mass_service.l
            / self.system_mass_service.ny
        )

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
            self.get_probabilities(),
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
