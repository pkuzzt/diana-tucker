#include "algorithm.hpp"
#include "logger.hpp"

template <typename T>
void push_in_matrix_J(Tensor<T> &J, vint shape, int x, int y,
                      const Tensor<T> &mat) {
    assert(J.shape()[0] == J.shape()[1]);
    assert(x < shape.size());
    assert(y < shape.size());
    assert(x != y); /* Diagonal matrices of J is I. */
    int N_J = J.shape()[0];
    T *data_J = J.data();
    T *data_mat = mat.data();
    int x_begin = 0;
    int y_begin = 0;
    int N = shape[x];
    int M = shape[y];
    for (int i = 0; i < shape.size(); i++) {
        if (i < x) {
            x_begin += shape[i];
        }
        if (i < y) {
            y_begin += shape[i];
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            int i_now = i + x_begin;
            int j_now = j + y_begin;
            T dat_now = data_mat[i * M + j];
            data_J[i_now * N_J + j_now] = dat_now;
            data_J[j_now * N_J + i_now] = dat_now;
        }
    }
}
template <typename T>
void push_in_vector_B(Tensor<T> &B, vint shape, int x, const Tensor<T> &vec) {
    assert(x < shape.size());
    int N_B = B.shape()[0];
    T *data_B = B.data();
    T *data_vec = vec.data();
    int x_begin = 0;
    int N = shape[x];
    for (int i = 0; i < shape.size(); i++) {
        if (i < x) {
            x_begin += shape[i];
        }
    }
    for (int i = 0; i < N; i++) {
        int i_now = i + x_begin;
        data_B[i_now] = data_vec[i];
    }
}

template <typename T>
void push_vector_into_U(const std::vector<Tensor<T>> &U,
                        const Tensor<T> &u_now) {
    int N = U.size();
    int pos = 0;
    T *data_u_now = u_now.data();
    for (int i = 0; i < N; i++) {
        int delta = U[i].size();
        Util::memcpy(U[i].data(), data_u_now + pos, delta * sizeof(T));
        pos += delta;
    }
}

template <typename T>
std::tuple<Tensor<T>, Tensor<T>>
generate_J_and_b(const Tensor<T> &A, const std::vector<Tensor<T>> &U) {
    int N = A.ndim();
    vint shape = A.shape();
    int N_J = 0;
    for (auto i : shape) {
        N_J += i;
    }
    Tensor<T> J(vint{N_J, N_J});
    Tensor<T> b(vint{N_J}, false);
    T *data_J = J.data();
    T *data_b = b.data();
    T *data_A = A.data();
    // Generate diagonal.
    for (int i = 0; i < N_J; i++) {
        data_J[i * N_J + i] = 1;
    }
    // Generate other blocks.
    Tensor<T> K[N];
    K[N - 1] = U[N - 1];
    for (int i = N - 2; i >= 2; i--) {
        K[i] = kr(U[i], K[i + 1]);
    }
    Tensor<T> B = A.copy();
    for (int i = 0; i < N - 1; i++) {
        Tensor<T> C = B.copy();
        for (int j = i + 1; j < N; j++) {
            if (j != N - 1) {
                vint C_shape = C.shape();
                C.reshape(vint{C_shape[0], C_shape[1], K[j + 1].size()});
                push_in_matrix_J(J, shape, i, j, ttv(C, K[j + 1], 2));
                C.reshape(C_shape);
                C = ttv(C, U[j], 1);
            } else {
                push_in_matrix_J(J, shape, i, j, C);
                push_in_vector_B(b, shape, i, ttv(C, U[j], 1));
            }
        }
        B = ttv(B, U[i], 0);
    }
    push_in_vector_B(b, shape, N - 1, B);
    return std::make_tuple(J, b);
}

template <typename T>
std::tuple<std::vector<Tensor<T>>, double>
Algorithm::GRQI::decompose(const Tensor<T> &A, const std::vector<Tensor<T>> &U,
                           double tolerance) {
    /* Do assertions. */
    int N = A.ndim();
    vint shape = A.shape();
    assert(tolerance > 0);
    assert(A.ndim() == U.size());
    for (int i = 0; i < A.ndim(); i++) {
        assert(U[i].ndim() == 1);
        assert(shape[i] == U[i].shape()[0]);
    }
    /* Calculate ||A||_F first. */
    double AF = A.fnorm();
    /* Sort U by it's size and permute A. */
    vint perm(N);
    for (int i = 0; i < N; i++) {
        perm[i] = i;
    }
    std::sort(perm.begin(), perm.end(),
              [&](const int &i, const int &j) -> bool {
                  return shape[i] > shape[j];
              });
    auto A_p = permute(A, perm);
    std::vector<Tensor<T>> U_p(N);
    for (int i = 0; i < N; i++) {
        U_p[i] = U[perm[i]].copy();
    }
    /* Start generalized Rayleigh quotient iteration. */
    double residual = 0;
    double residual_last = 2 * tolerance;
    double lambda;
    int k = 0;
    info("||A||_F = " + std::to_string(AF) + ".");
    while (std::abs(residual_last - residual) > tolerance) {
        auto [J, b] = generate_J_and_b(A_p, U_p);
        auto u = solve(J, b);
        u.nmul(N - 2);
        push_vector_into_U(U_p, u);
        for (int i = 0; i < N; i++) {
            U_p[i].nmul(1 / U_p[i].fnorm());
        }
        lambda = ttvc(A_p, U_p)[0];
        residual_last = residual;
        residual = std::sqrt(1 - (lambda * lambda) / (AF * AF));
        k++;
        info("Iteration " + std::to_string(k) +
             ": lambda = " + std::to_string(lambda) +
             "; residual = " + std::to_string(residual) + "; error_delta = " +
             std::to_string(std::abs(residual_last - residual)) + ".");
    }
    /* Sort U back. */
    std::vector<Tensor<T>> U_ret(N);
    for (int i = 0; i < N; i++) {
        U_ret[perm[i]] = U_p[i].copy();
    }
    return std::make_tuple(U_ret, lambda);
}