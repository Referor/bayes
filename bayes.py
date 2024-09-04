import streamlit as st
import numpy as np

# Функция для вычисления вероятности по теореме Байеса
def bayes_update(prior, likelihood, false_positive_rate):
    denominator = (likelihood * prior) + (false_positive_rate * (1 - prior))
    posterior = (likelihood * prior) / denominator
    return posterior

# Функция для симуляции Монте-Карло
def monte_carlo_simulation(p, sample_size, population_size, n_simulations=10000):
    # Симуляция вероятности наличия проблемы у выборки sample_size
    simulated_counts = np.random.binomial(n=sample_size, p=p, size=n_simulations)
    # Перевод в пропорции (доли) от выборки
    proportions = simulated_counts / sample_size
    # Масштабирование до размеров генеральной совокупности
    projected_counts = proportions * population_size
    return projected_counts

# Инициализация данных в Streamlit
st.title("Анализ вероятности проблемы с использованием Байесовской теоремы и симуляции Монте-Карло")

# Инициализация переменных в session_state
if "prior" not in st.session_state:
    st.session_state.prior = 0.5
    st.session_state.likelihood = 0.8
    st.session_state.false_positive_rate = 0.3
    st.session_state.history = []
    st.session_state.interview_count = 0

# Ввод начальных данных
st.session_state.prior = st.number_input("Начальная вероятность того, что проблема есть у клиентов, P(A) (например, 0.5 для 50%)", min_value=0.0, max_value=1.0, value=st.session_state.prior, step=0.01, disabled=st.session_state.interview_count > 0)
st.session_state.likelihood = st.number_input("P(B|A) - Вероятность подтверждения проблемы, с помощью интервью, если она существует (например, 0.8 для 80%)", min_value=0.0, max_value=1.0, value=st.session_state.likelihood, step=0.01, disabled=st.session_state.interview_count > 0)
st.session_state.false_positive_rate = st.number_input("P(B|¬A) - Вероятность подтверждения проблемы, если она не существует (например, 0.3 для 30%)", min_value=0.0, max_value=1.0, value=st.session_state.false_positive_rate, step=0.01, disabled=st.session_state.interview_count > 0)

# Кнопка для подтверждения результата интервью
if st.button("Интервью подтвердило проблему"):
    st.session_state.interview_count += 1
    st.session_state.prior = bayes_update(st.session_state.prior, st.session_state.likelihood, st.session_state.false_positive_rate)
    st.session_state.history.append(f"Интервью {st.session_state.interview_count}: Вероятность после подтверждения = {st.session_state.prior:.4f}")

# Кнопка для опровержения результата интервью
if st.button("Интервью не подтвердило проблему"):
    st.session_state.interview_count += 1
    st.session_state.prior = bayes_update(st.session_state.prior, 1 - st.session_state.likelihood, 1 - st.session_state.false_positive_rate)
    st.session_state.history.append(f"Интервью {st.session_state.interview_count}: Вероятность после опровержения = {st.session_state.prior:.4f}")

# Показ истории обновлений
st.subheader("История изменений вероятности")
for entry in st.session_state.history:
    st.write(entry)

# Показ итоговой информации
st.subheader("Итоговая информация")
st.write(f"Всего проведено интервью: {st.session_state.interview_count}")
st.write(f"Финальная вероятность после всех интервью: {st.session_state.prior:.4f}")

# Ввод данных для симуляции Монте-Карло
st.subheader("Симуляция Монте-Карло для оценки числа компаний с проблемой")
population_size = st.number_input("Размер генеральной совокупности (например, 1000 компаний)", min_value=1, value=1000)
n_simulations = st.number_input("Количество симуляций Монте-Карло", min_value=100, value=1000, step=100)

# Кнопка для запуска симуляции
if st.button("Запустить симуляцию Монте-Карло"):
    projected_counts = monte_carlo_simulation(st.session_state.prior, st.session_state.interview_count, population_size, n_simulations)

    # Расчет доверительного интервала
    lower_bound = np.percentile(projected_counts, 2.5)
    upper_bound = np.percentile(projected_counts, 97.5)

    # Вывод результатов симуляции
    st.subheader("Результаты симуляции Монте-Карло")
    st.write(f"95% доверительный интервал для числа компаний с проблемой: от {int(lower_bound)} до {int(upper_bound)}")

    # Вывод процентного диапазона доверительного интервала
    lower_percentage = (lower_bound / population_size) * 100
    upper_percentage = (upper_bound / population_size) * 100

    st.write(f"Процент компаний, которые могут иметь проблему: от {lower_percentage:.2f}% до {upper_percentage:.2f}%")

    # Вывод интерпретации результатов
    st.write(f"На основе {n_simulations} симуляций, модель прогнозирует, что в генеральной совокупности из {population_size} компаний, примерно от {int(lower_bound)} до {int(upper_bound)} компаний (или от {lower_percentage:.2f}% до {upper_percentage:.2f}%) могут иметь проблему.")
    

# Пояснение результатов:
st.subheader("Пояснения к методам")

st.write("""
### Что такое доверительный интервал?
Доверительный интервал — это диапазон значений, в котором с определенной степенью уверенности (в нашем случае — 95%) можно ожидать, что будет находиться истинное значение. Например, если мы получили доверительный интервал для числа компаний с проблемой от 200 до 300, это означает, что с 95%-ной вероятностью количество компаний с проблемой будет в этом диапазоне.

### Что такое симуляция Монте-Карло?
Симуляция Монте-Карло — это метод, который использует случайные числа и вероятности для моделирования большого числа возможных сценариев. В нашем случае мы моделируем, сколько компаний может иметь проблему на основе текущей вероятности, проведя тысячи симуляций (например, 10,000).

### Что такое теорема Байеса?
Теорема Байеса — это метод вычисления вероятности события на основе новых данных. В контексте нашего анализа мы обновляем вероятность существования проблемы (вероятность A) каждый раз, когда получаем новый результат интервью — подтвердилось наличие проблемы или нет. Этот метод позволяет постепенно улучшать наши предсказания на основе поступающих данных.
""")
