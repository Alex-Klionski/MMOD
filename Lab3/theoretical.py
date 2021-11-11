class Theoretical:
    def __init__(self, m, l, ny):
        self.m = m
        self.l = l
        self.ny = ny

    def get_probabilities(self):
        ro = self.l / self.ny
        p_0 = (1 - ro) / (1 - ro ** (self.m+2))

        probabilities = [(ro ** i) * p_0 for i in range(1, self.m + 1)]
        return probabilities

    @property
    def final_probabilities(self):
        return self.get_probabilities()

    # Вероятность отказа
    def get_reject(self):
        ro = self.l / self.ny
        p_0 = (1 - ro) / (1 - ro ** (self.m + 2))
        return ro ** (self.m + 1) * p_0

    # Относитальная пропускная способность
    def get_relative_bandwidth(self):
        """Q = 1 - p(отк)"""
        return 1 - self.get_reject()

    # Абсолютная пропускная способность
    def get_absolute_bandwidth(self):
        """A = λ * Q"""
        return self.l * self.get_relative_bandwidth()

    # Среднее число заявок в очереди
    def get_average_number_of_elements_in_queue(self):
        ro = self.l / self.ny
        p_0 = (1 - ro) / (1 - ro ** (self.m + 2))

        return ro ** 2 * (1 - ro ** self.m * (self.m * (1 - ro) + 1)) / ((1 - ro) ** 2) * p_0

    # Среднее число заявок в СМО
    def get_average_number_of_elements_in_smo(self):
        ro = self.l / self.ny
        p_0 = (1 - ro) / (1 - ro ** (self.m + 2))
        return ro * (1 + ro * ((1 - ro ** self.m * (self.m * (1 - ro) + 1)) / ((1 - ro) ** 2))) * p_0

    # Среднее время пребывания заявки в очереди
    def get_average_time_an_item_in_queue(self):
        return self.get_average_number_of_elements_in_queue() / self.l

    # Среднее время пребывания заявки в СМО
    def get_average_time_an_item_in_smo(self):
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
            self.get_average_number_of_elements_in_smo(),
            self.get_average_number_of_elements_in_queue(),
            self.get_average_number_of_elements_in_smo(),
            self.get_average_time_an_item_in_queue(),
            self.get_average_number_of_busy_channels(),
        )
