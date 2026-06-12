def armijo(f, grad_f, x0, alpha=1e-4, beta=0.5, gamma=1):
    '''
    Armijo line search over f and its gradient

    f - funtion taking a vector
    grad_f - gradient of f, taking and returning a vector of the same dimension
    x0 - scan location
    alpha - sufficient decrease constant
    beta - constant governing granularity of search
    gamma - initial step size
    '''

    f_x = f(x0)
    grad_f_x = grad_f(x0)

    z = -1 * grad_f_x

    while f(x0 + gamma * z) > f_x + alpha * gamma * grad_f_x.T @ z :
        gamma = beta * gamma

    return x0 + gamma * z, gamma