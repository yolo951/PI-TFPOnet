import sympy as sp


mu0, h, b0, p0, q0, eps, x, y = sp.symbols('mu0 h b0 p0 q0 eps x y')


mu0_expr = 1/eps*sp.sqrt(b0 + (p0**2 + q0**2)/4/eps**2)


sqrt2 = sp.sqrt(2)


A = 1/4/sp.sinh(mu0_expr*h/sqrt2)**2 * sp.Matrix([
    [sp.exp(-mu0_expr * h / sqrt2), -1, sp.exp(mu0_expr * h / sqrt2), -1],
    [sp.exp(mu0_expr * h / sqrt2), -1, sp.exp(-mu0_expr * h / sqrt2), -1],
    [-1, sp.exp(mu0_expr * h / sqrt2), -1, sp.exp(-mu0_expr * h / sqrt2)],
    [-1, sp.exp(-mu0_expr * h / sqrt2), -1, sp.exp(mu0_expr * h / sqrt2)]
])


A_prime = sp.Matrix([
    [
        sp.exp((p0*x + q0*y) / 2 / eps**2 + (p0 + q0) * h / 4 / eps**2) * sp.exp(mu0_expr * (x + y) / sqrt2),
        sp.exp((p0*x + q0*y) / 2 / eps**2 + (p0 + q0) * h / 4 / eps**2) * sp.exp(-mu0_expr * (x + y) / sqrt2),
        sp.exp((p0*x + q0*y) / 2 / eps**2 + (p0 + q0) * h / 4 / eps**2) * sp.exp(mu0_expr * (x - y) / sqrt2),
        sp.exp((p0*x + q0*y) / 2 / eps**2 + (p0 + q0) * h / 4 / eps**2) * sp.exp(mu0_expr * (y - x) / sqrt2)
    ],
    [
        sp.exp((p0*(x+h) + q0*y) / 2 / eps**2 + (q0 - p0) * h / 4 / eps**2) * sp.exp(mu0_expr * (x + y + h) / sqrt2),
        sp.exp((p0*(x+h) + q0*y) / 2 / eps**2 + (q0 - p0) * h / 4 / eps**2) * sp.exp(-mu0_expr * (x + y + h) / sqrt2),
        sp.exp((p0*(x+h) + q0*y) / 2 / eps**2 + (q0 - p0) * h / 4 / eps**2) * sp.exp(mu0_expr * (x + h - y) / sqrt2),
        sp.exp((p0*(x+h) + q0*y) / 2 / eps**2 + (q0 - p0) * h / 4 / eps**2) * sp.exp(mu0_expr * (y - x - h) / sqrt2)
    ],
    [
        sp.exp((p0*(x+h) + q0*(y+h)) / 2 / eps**2 - (p0 + q0) * h / 4 / eps**2) * sp.exp(mu0_expr * (x + y + 2*h) / sqrt2),
        sp.exp((p0*(x+h) + q0*(y+h)) / 2 / eps**2 - (p0 + q0) * h / 4 / eps**2) * sp.exp(-mu0_expr * (x + y + 2*h) / sqrt2),
        sp.exp((p0*(x+h) + q0*(y+h)) / 2 / eps**2 - (p0 + q0) * h / 4 / eps**2) * sp.exp(mu0_expr * (x - y) / sqrt2),
        sp.exp((p0*(x+h) + q0*(y+h)) / 2 / eps**2 - (p0 + q0) * h / 4 / eps**2) * sp.exp(mu0_expr * (y - x) / sqrt2)
    ],
    [
        sp.exp((p0*x + q0*(y+h)) / 2 / eps**2 + (p0 - q0) * h / 4 / eps**2) * sp.exp(mu0_expr * (x + y + h) / sqrt2),
        sp.exp((p0*x + q0*(y+h)) / 2 / eps**2 + (p0 - q0) * h / 4 / eps**2) * sp.exp(-mu0_expr * (x + y + h) / sqrt2),
        sp.exp((p0*x + q0*(y+h)) / 2 / eps**2 + (p0 - q0) * h / 4 / eps**2) * sp.exp(mu0_expr * (x - y - h) / sqrt2),
        sp.exp((p0*x + q0*(y+h)) / 2 / eps**2 + (p0 - q0) * h / 4 / eps**2) * sp.exp(mu0_expr * (y + h - x) / sqrt2)
    ]
])


product_A_A_prime = A * A_prime
product_A_prime_A = A_prime * A


simplified_product_A_A_prime = sp.simplify(product_A_A_prime)
simplified_product_A_prime_A = sp.simplify(product_A_prime_A)


print("A * A' =")
sp.pprint(simplified_product_A_A_prime)

print("\nA' * A =")
sp.pprint(simplified_product_A_prime_A)

is_identity_A_A_prime = simplified_product_A_A_prime == sp.eye(4)
is_identity_A_prime_A = simplified_product_A_prime_A == sp.eye(4)

print("\nA * A' is identity:", is_identity_A_A_prime)
print("A' * A is identity:", is_identity_A_prime_A)
