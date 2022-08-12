import matplotlib.pyplot as plt
import numpy as np
import math

def representar_regresion_logistica(b0, b1):
    ''' representacion grafica de linea y sigmoide'''
    
    # generacion de datos
    x = np.linspace(-6, 6)
    x_to_classify = [-3, -1.8, -2, -1, 0, 1.5 , 2, 3, -0.12, 0.34]
    y_to_classify = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    y_sig = generate_sigmoid_points(b0, b1, x)
    y = b1*x + b0

    # representacion grafica
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15,5))

    ax1.set_title('Función Sigmoide')
    ax1.scatter(x_to_classify, y_to_classify)
    ax1.plot(x, y_sig, c='g')
    ax1.set_xlim(-3.5, 3.5)

    ax2.set_title(f'Funcion Lineal, b0:{ b0}; b1:{ b1}')
    ax2.plot(x, y)
    ax2.set_ylim(-30, 30)
    ax2.set_xlim(-5, 5)
    return plt.show()
    
    
def sigmoid(arr, c, scale=1):
    ''' sigmoid '''
    arr = np.asarray(arr)
    return 1/(1 + np.exp(-arr*scale+c))


def calculate_logs(res, y_ground_truth):
    ''' Calcular log likelihood - if 1 results is log(predic) else log(1 -predic)'''
    log_like = 0
    for idx, y_i in enumerate(y_ground_truth):
        if y_i == 1 or y_i:
            log_like += math.log(res[idx])
        else:
            log_like += math.log(1 - res[idx])   
    return log_like
    
    
def get_log_likelihood(x, y, model_results=False, b0=False, b1=False):
    ''' estimacion por maxima verosimilitud'''
    # w from model
    if model_results:
        w = np.array(model_results.params)
    
    # manual weights    
    else:
        w = np.array([b0, b1])
        
    print(f"Coeficientes: ")
    print(f"    b0: {w[0]}")
    print(f"    b1: {w[1]}")

    # prediction
    res_m = np.dot(x, w)
    res_ms = sigmoid(res_m, 0)
    
    # calculate logs
    log_like = calculate_logs(res_ms, y)
    return log_like