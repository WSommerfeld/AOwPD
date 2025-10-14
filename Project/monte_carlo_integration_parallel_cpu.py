import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

from multiprocessing import Pool



def monte_carlo_integration_thread(f, a, b, n, y_min, y_max):
    """Pojedynczy wątek"""

    #Gambling
    n_hit = 0

    #GŁÓWNA PĘTLA
    for i in range(n):
        #Losowy punkt (x, y)
        random_x = a + np.random.random() * (b - a)
        random_y = y_min + np.random.random() * (y_max - y_min)

        #Wartość funkcji w random_x
        y_func = f(random_x)

        #Sprawdzenie trafienia
        if y_min >= 0:
            #Funkcja dodatnia
            if random_y <= y_func:
                n_hit += 1
        else:
            #Funkcja ujemna
            if y_func >= 0 and 0 <= random_y <= y_func:
                n_hit += 1
            elif y_func < 0 and y_func <= random_y <= 0:
                n_hit -= 1
    #GŁÓWNA PĘTLA

    return n_hit


def monte_carlo_integration(f, a, b, n, n_threads=None):
    """Całkowanie numeryczne metodą Monte carlo - wersja wielowątkowa"""

    #Liczba rdzeni
    if n_threads is None:
        n_threads = multiprocessing.cpu_count()

    #Wyznaczenie ograniczeń
    x_sample = np.linspace(a, b, 1000)
    y_sample = f(x_sample)
    y_min = np.min(y_sample)
    y_max = np.max(y_sample)

    #Podział próbek na wątki
    samples_per_thread = n // n_threads
    remainder = n % n_threads

    #Zadania dla wątków
    tasks = []
    for i in range(n_threads):
        n_samples = samples_per_thread + (1 if i < remainder else 0)
        tasks.append((f, a, b, n_samples, y_min, y_max))

    #Uruchomienie wątków
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        results = executor.map(lambda args: monte_carlo_integration_thread(*args), tasks)

    #Suma trafień ze wszystkich wątków
    total_hits = sum(results)

    #Obliczenie całki
    rectangle = (b - a) * (y_max - y_min)
    estimated_integral = rectangle * (total_hits / n)

    return estimated_integral


#wizualizacja
def monte_carlo_thread_visualization(f, a, b, n, y_min, y_max):
    """Pojedynczy wątek + dane do wizualizacji"""

    x_random = []
    y_random = []
    hit = []
    n_hit = 0

    for i in range(n):
        random_x = a + np.random.random() * (b - a)
        random_y = y_min + np.random.random() * (y_max - y_min)
        y_func = f(random_x)

        x_random.append(random_x)
        y_random.append(random_y)

        if y_min >= 0:
            if random_y <= y_func:
                hit.append(True)
                n_hit += 1
            else:
                hit.append(False)
        else:
            if y_func >= 0 and 0 <= random_y <= y_func:
                hit.append(True)
                n_hit += 1
            elif y_func < 0 and y_func <= random_y <= 0:
                hit.append(True)
                n_hit += 1
            else:
                hit.append(False)

    return x_random, y_random, hit, n_hit


def visualize_monte_carlo(f, a, b, n=1000, n_threads=None):
    """Wizualizacja"""

    # Liczba wątków na podstawie liczby rdzeni
    if n_threads is None:
        n_threads = multiprocessing.cpu_count()

    # Wyznaczenie ograniczeń
    x_sample = np.linspace(a, b, 1000)
    y_sample = f(x_sample)
    y_min = np.min(y_sample)
    y_max = np.max(y_sample)

    # Podział próbek na wątki
    samples_per_thread = n // n_threads
    remainder = n % n_threads

    # Zadania dla wątków
    tasks = []
    for i in range(n_threads):
        n_samples = samples_per_thread + (1 if i < remainder else 0)
        tasks.append((f, a, b, n_samples, y_min, y_max))

    # Uruchomienie wątków
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        results = executor.map(lambda args: monte_carlo_thread_visualization(*args), tasks)

    # Łączenie wyników z wszystkich wątków
    x_random = []
    y_random = []
    hit = []
    n_hit = 0

    for x_r, y_r, h, n_h in results:
        x_random.extend(x_r)
        y_random.extend(y_r)
        hit.extend(h)
        n_hit += n_h

    # Konwersja do numpy
    x_random = np.array(x_random)
    y_random = np.array(y_random)
    hit = np.array(hit)

    # Rysowanie
    plt.figure(figsize=(12, 7))

    rect_x = [a, b, b, a, a]
    rect_y = [y_min, y_min, y_max, y_max, y_min]
    plt.plot(rect_x, rect_y, 'k--', linewidth=1.5, label='Rectangle', zorder=1)

    x_plot = np.linspace(a, b, 1000)
    y_plot = f(x_plot)
    plt.plot(x_plot, y_plot, 'b-', linewidth=3, label='f(x)', zorder=3)
    plt.fill_between(x_plot, 0, y_plot, alpha=0.15, color='blue')

    # Punkty
    plt.scatter(x_random[hit], y_random[hit],
                c='green', s=5, alpha=0.7, label=f'HIT ({n_hit})', zorder=2)
    plt.scatter(x_random[~hit], y_random[~hit],
                c='red', s=5, alpha=0.7, label=f'MISS ({n - n_hit})', zorder=2)

    # Pole
    rectangle = (b - a) * (y_max - y_min)
    estimated_integral = rectangle * (n_hit / n)

    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title(f'Metoda Monte Carlo - parallel ({n_threads} threads)\n' +
              f'Rectangle field = {rectangle:.4f}, ' +
              f'Ratio = {n_hit}/{n} = {n_hit / n:.4f}\n' +
              f'Integral ≈ {estimated_integral:.4f}',
              fontsize=13)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def monte_carlo_integration_multiprocess(f, a, b, n, n_processes=None):
    """Multiprocessing (GIL ogranicza wątki)"""

    if n_processes is None:
        n_processes = multiprocessing.cpu_count()

    #Wyznaczenie ograniczeń
    x_sample = np.linspace(a, b, 1000)
    y_sample = f(x_sample)
    y_min = np.min(y_sample)
    y_max = np.max(y_sample)

    #Podział próbek
    samples_per_process = n // n_processes
    remainder = n % n_processes

    tasks = []
    for i in range(n_processes):
        n_samples = samples_per_process + (1 if i < remainder else 0)
        tasks.append((f, a, b, n_samples, y_min, y_max))

    #Procesy zamiast wątków
    with Pool(processes=n_processes) as pool:
        results = pool.starmap(monte_carlo_integration_thread, tasks)

    total_hits = sum(results)
    rectangle = (b - a) * (y_max - y_min)
    estimated_integral = rectangle * (total_hits / n)

    return estimated_integral
