import numpy as np
from sage.coding.goppa_code import GoppaCode
from sage.rings.finite_rings.finite_field_constructor import GF
from sage.rings.integer import Integer
from sage.all import vector

np.random.seed(1)


class Bob:

    def __init__(self, m, t):
        self.g = None
        self.G = None
        self.k, self.n = None, None
        self.S = None
        self.P = None
        self.G_prime = None
        self.C = None
        self.m = m
        self.t = t

    def gen_keys(self):
        self.g, self.G = self.gen_G()

        self.k = self.G.shape[0]
        self.n = self.G.shape[1]
        self.S = self.gen_S()
        self.P = self.gen_P()

        self.G_prime = np.matmul(np.matmul(self.S, self.G), self.P)

    def gen_G(self):

        F = GF(2**self.m)
        R = F['x']
        (x,) = R._first_ngens(1)

        g = np.random.randint(0, 2, 1)[0]*x**Integer(self.t)
        for i in range(self.t):
            g += np.random.randint(0, 2, 1)[0]*x**Integer(i)
        print(f"\n\ng(z): {g}")

        L = [a for a in F.list() if g(a) != Integer(0)]
        self.C = GoppaCode(g, L)
        G = self.C.generator_matrix().numpy().astype('int')
        print(f"G: {G}")
        return g, G

    def gen_S(self):

        S = np.random.randint(0, 2, (self.k, self.k))

        while np.isclose(np.linalg.det(S), 0):

            S = np.random.randint(0, 2, (self.k, self.k))

        return S

    def gen_P(self):

        P = np.zeros((self.n, self.n))

        rows = np.arange(0, self.n)
        cols = np.arange(0, self.n)

        while rows.shape[0] > 0:
            row = np.random.randint(rows.shape[0])
            col = np.random.randint(cols.shape[0])

            P[rows[row]][cols[col]] = 1
            rows = np.delete(rows, row)
            cols = np.delete(cols, col)

        return P

    def send_keys(self):
        self.gen_keys()
        return self.G_prime, self.t

    def decrypt(self, y):
        y_prime = np.matmul(y, np.linalg.inv(self.P))
        vec = vector(y_prime.astype('int'))
        m_prime = self.C.decode_to_code(vec)
        np_m_prime = m_prime.numpy()[:self.S.shape[0]]
        return np.matmul(np_m_prime, np.linalg.inv(self.S)).astype('int')


class Alice:

    def __init__(self):
        self.unencrypted_m = None

    def gen_message(self, k):
        return np.random.randint(0, 2, k)

    def encrypt(self, G_prime, t):
        self.unencrypted_m = self.gen_message(G_prime.shape[0])

        e = self.generate_e(G_prime.shape[1], t)
        mat_prod = np.matmul(self.unencrypted_m, G_prime)
        return (mat_prod + e).astype('int')

    def generate_e(self, n, t):
        num_ones = np.random.randint(0, t+1, 1)[0]

        e = np.zeros(n)
        e[:num_ones] = np.ones(num_ones)
        return np.random.permutation(e)


m = 4
t = 2
print(f"m: {m}")
print(f"t: {t}")

bob = Bob(m, t)
g_prime, t = bob.send_keys()
print(f"k: {g_prime.shape[0]}")
print(f"n: {g_prime.shape[1]}")
print(f"Public Key size: {g_prime.nbytes} bytes\n\n")

alice = Alice()

y = alice.encrypt(g_prime, t)
print(f"Alice's message: {alice.unencrypted_m}")
m_decoded = bob.decrypt(y)

print(f"Bob's decrypted message: {m_decoded}")

message_equals = np.array_equal(m_decoded, alice.unencrypted_m)
print(f"Bob's decrypted message == Alice's message: {message_equals}")
