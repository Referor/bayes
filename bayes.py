import streamlit as st
from math import sqrt

# Функция для вычисления вероятности по теореме Байеса
def bayes_update(prior, likelihood, false_positive_rate):
    denominator = (likelihood * prior) + (false_positive_rate * (1 - prior))
    posterior = (likelihood * prior) / denominator
    # Проверка границ вероятности
    posterior = max(0, min(1, posterior))
    return posterior

# Функция для расчета доверительного интервала с учетом генеральной совокупности
def confidence_interval(p, n_interviews, n_population, z=1.96):
    # Finite population correction factor
    fpc = sqrt((n_population - n_interviews) / (n_population - 1))
    margin_of_error = z * sqrt((p * (1 - p)) / n_interviews) * fpc
    lower_bound = p - margin_of_error
    upper_bound = p + margin_of_error
    # Проверка границ вероятности
    lower_bound = max(0, lower_bound)
    upper_bound = min(1, upper_bound)
    return lower_bound, upper_bound

# Заголовок приложения
st.title("Делаем выводы из интервью на малых данных, используя теорему Байеса")
st.text("by Андрей Батрименко, Вадим Глазков и GPT")

# Инициализация переменных в session_state
if "prior" not in st.session_state:
    st.session_state.prior = 0.5
    st.session_state.likelihood = 0.8
    st.session_state.false_positive_rate = 0.3
    st.session_state.history = []
    st.session_state.interview_count = 0

# Поля ввода всегда видимы, но становятся неактивными после первого ввода
st.session_state.prior = st.number_input("Начальная вероятность того, что проблема есть у клиентов, P(A) (например, 0.5 для 50%)", min_value=0.0, max_value=1.0, value=st.session_state.prior, step=0.01, disabled=st.session_state.interview_count > 0)
st.session_state.likelihood = st.number_input("P(B|A) - Вероятность подтверждения проблемы, с помощью интервью, если она существует (например, 0.8 для 80%)", min_value=0.0, max_value=1.0, value=st.session_state.likelihood, step=0.01, disabled=st.session_state.interview_count > 0)
st.session_state.false_positive_rate = st.number_input("P(B|¬A) - Вероятность подтверждения проблемы, с помощью интервью, если она не существует (например, 0.3 для 30%)", min_value=0.0, max_value=1.0, value=st.session_state.false_positive_rate, step=0.01, disabled=st.session_state.interview_count > 0)

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

# Кнопка для сброса данных и начала заново
#if st.button("Сбросить и начать заново"):
#    st.session_state.clear()  # Очищаем все данные в session_state
    # Используем JavaScript для перезагрузки страницы
#    st.experimental_rerun()

# Отображение истории обновлений
st.subheader("История изменений вероятности")
for entry in st.session_state.history:
    st.write(entry)

# Показ итоговой информации
st.subheader("Итоговая информация")
st.write(f"Всего проведено интервью: {st.session_state.interview_count}")
st.write(f"Финальная вероятность после всех интервью: {st.session_state.prior:.4f}")

# Итоговый вывод
if st.session_state.prior > 0.5:
    st.write("Итог: Проблема, скорее всего, существует.")
else:
    st.write("Итог: Проблема, скорее всего, не существует.")

# Ввод данных для расчета доверительного интервала
st.subheader("Расчет доверительного интервала")
sample_size = st.number_input("Размер генеральной совокупности", min_value=1, value=100)

# Кнопка для расчета доверительного интервала
if st.button("Рассчитать доверительный интервал"):
    lower_bound, upper_bound = confidence_interval(st.session_state.prior, st.session_state.interview_count, sample_size)
    lower_count = int(lower_bound * sample_size)
    upper_count = int(upper_bound * sample_size)
    st.write(f"Доверительный интервал: [{lower_bound:.4f}, {upper_bound:.4f}]")
    st.write(f"Ожидаемое количество клиентов, имеющих проблему: от {lower_count} до {upper_count}")

# Пояснение выводов
st.subheader("Пояснение результатов")
st.write("""
    - Вероятность после каждого интервью показывает, насколько уверены мы в существовании проблемы у клиентов.
    - Доверительный интервал показывает диапазон возможного количества клиентов, которые могут иметь проблему, исходя из вашей выборки и размера генеральной совокупности.
    - Если финальная вероятность после всех интервью превышает 50%, это значит, что проблема скорее всего существует.
""")
