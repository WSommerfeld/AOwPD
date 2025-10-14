import numpy as np
import matplotlib.pyplot as plt


def monte_carlo_integration(f, a, b, n):
    """Całkowanie numeryczne metodą Monte carlo - wersja sekwencyjna"""

    #Wyznaczenie ohraniczeń
    x_sample = np.linspace(a, b, 1000)
    y_sample = f(x_sample)
    y_min = np.min(y_sample)
    y_max = np.max(y_sample)

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

    #Wartość całki
    rectangle = (b - a) * (y_max - y_min)
    estimated_integral = rectangle * (n_hit / n)

    return estimated_integral


def visualize_monte_carlo(f, a, b, n=1000):

    "Wizualizacja"

    x_sample = np.linspace(a, b, 1000)
    y_sample = f(x_sample)
    y_min = np.min(y_sample)
    y_max = np.max(y_sample)


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

    x_random = np.array(x_random)
    y_random = np.array(y_random)
    hit = np.array(hit)


    plt.figure(figsize=(12, 7))


    rect_x = [a, b, b, a, a]
    rect_y = [y_min, y_min, y_max, y_max, y_min]
    plt.plot(rect_x, rect_y, 'k--', linewidth=1.5, label='Rectangle', zorder=1)


    x_plot = np.linspace(a, b, 1000)
    y_plot = f(x_plot)
    plt.plot(x_plot, y_plot, 'b-', linewidth=3, label='f(x)', zorder=3)
    plt.fill_between(x_plot, 0, y_plot, alpha=0.15, color='blue')

    #Punkty
    plt.scatter(x_random[hit], y_random[hit],
                c='green', s=5, alpha=0.7, label=f'HIT ({n_hit})', zorder=2)
    plt.scatter(x_random[~hit], y_random[~hit],
                c='red', s=5, alpha=0.7, label=f'MISS ({n - n_hit})', zorder=2)

    #Pole
    rectangle = (b - a) * (y_max - y_min)
    estimated_integral = rectangle * (n_hit / n)

    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title(f'Metoda Monte Carlo\n' +
              f'Rectangle field = {rectangle:.4f}, ' +
              f'Ratio = {n_hit}/{n} = {n_hit / n:.4f}\n' +
              f'Integral ≈ {estimated_integral:.4f}',
              fontsize=13)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



