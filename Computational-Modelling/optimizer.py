from scipy import optimize
import numpy as np
import ws2_19InClass as sol

def error(v_0):
    aim = np.array([8.1678,2.0543])
    actual = sol.get_sol(v_0[0])
    error = np.linalg.norm(aim-actual)
    print(error, v_0)
    return error

solution = optimize.fmin(error,[2], maxiter=1000)
