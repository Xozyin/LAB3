import torch
import numpy as np
import pandas as pd

# Задание 1
# 1. Cоздайте тензор x целочисленного типа, хранящий случайное значение
from random import randint
minSize = 5;
maxSize = 9;
# Создаем тензор со случайными целыми числами (от 0 до 99)
x = torch.randint(0, 99, size=(randint(minSize, maxSize), randint(minSize, maxSize)))
print(x)
print(x.dtype)  #Тип torch.int64

# 2. Преобразуйте тензор к типу float32; 
x = x.to(dtype=torch.float32)
print(x)

# 3. Операции с тензором
n = 2
# Возведение в степень n
x = x ** n
print("\n3.1. Тензор после возведения в степень", n, ":\n", x)
# Умножение на случайное число от 1 до 10
rand_val = randint(1, 10)
x = x * rand_val
print("\n3.2. После умножения на", rand_val, ":\n", x)
# Взятие экспоненты
x = torch.exp(x)
print("\n3.3. После взятия экспоненты:\n", x)

# 4. Вычислите и выведите на экран значение производной для полученного в пункте 3 значения по x
# Создаем копию исходного тензора перед возведением в степень и другими операциями
# Для этого нужно перестроить вычисления с отслеживанием градиента

# Сначала создадим новый тензор с теми же значениями, но с requires_grad=True
x_with_grad = x.detach().clone().requires_grad_(True)

# Повторяем операции пункта 3, но с отслеживанием градиента
y = x_with_grad ** n
y = y * rand_val
z = torch.exp(y)

# Вычисляем градиент
z.backward(torch.ones_like(z))  # передаем единичный тензор той же формы
print("\n4. Производная dz/dx:\n", x_with_grad.grad)

# Задание 2
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import matplotlib.pyplot as plt  # matplotlib для построения графиков
import torch.nn as nn  # Модуль для создания нейронных сетей
import torch.optim as optim  # Модуль для оптимизации

# Считываем данные
df = pd.read_csv('data.csv')

# три столбца - это признаки, четвертый - целевая переменная (то, что мы хотим предсказывать)

# выделим целевую переменную в отдельную переменную
y = df.iloc[:, 4].values

# так как ответы у нас строки - нужно перейти к численным значениям
y = np.where(y == "Iris-setosa", 1, -1)

# возьмем три признака для обучения
X = df.iloc[:, [0, 1, 2]].values  # теперь используем три признака

# Преобразуем данные в тензоры PyTorch
X_tensor = torch.tensor(X, dtype=torch.float32)  # Признаки
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # Целевые значения (преобразуем в размер [n, 1])

# Определим модель (линейный слой)
class Neuron(nn.Module):
    def __init__(self):
        super(Neuron, self).__init__()
        self.linear = nn.Linear(3, 1)  # Линейный слой с 3 входами и 1 выходом

    def forward(self, x):
        return self.linear(x)

# Создаем модель
model = Neuron()

# Определяем функцию потерь и оптимизатор
criterion = nn.MSELoss()  # Среднеквадратичная ошибка
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Стохастический градиентный спуск

# Список для сохранения весов
weights_history = []

# Обучение модели
n = 100
for i in range(n):
    # Прямой проход
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)

    # Обратный проход и оптимизация
    optimizer.zero_grad()  # Обнуляем градиенты
    loss.backward()  # Вычисляем градиенты
    optimizer.step()  # Обновляем веса

    # Сохраняем веса на каждой 10-й итерации
    if (i + 1) % 10 == 0:
        weights_history.append([model.linear.weight.data.numpy().copy(), model.linear.bias.data.numpy().copy()])

    # Выводим данные каждые 10 итераций
    if (i + 1) % 10 == 0:
        print(f'Итерация [{i + 1}/{n}], Ошибка: {loss.item():.4f}')
        print(f'Веса: {model.linear.weight.data.numpy()}')
        print(f'Смещение: {model.linear.bias.data.numpy()}\n')