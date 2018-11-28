# Code For Submission
import csv
import pandas as pd
import numpy as np
import re
import tensorflow as tf
import edward as ed
from metric_learn import LMNN
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale as varscale
from sklearn.model_selection import LeaveOneOut
from scipy.linalg import sqrtm
from edward.models import MultivariateNormalTriL, Normal, Poisson


def rbf_fun(X, X2=None, lengthscale=1.0, variance=1.0, noise=0):
    lengthscale = tf.constant(lengthscale, dtype=tf.float32)
    variance = tf.constant(variance, dtype=tf.float32)
    noise_sq = tf.constant(noise, dtype=tf.float32)
    noise = tf.constant(noise, dtype=tf.float32)
    X = tf.convert_to_tensor(X, tf.float32)
    Xs = tf.reduce_sum(tf.square(X), 1)
    if X2 is None:
        X2 = X
        X2s = Xs
    else:
        X2 = tf.convert_to_tensor(X2, tf.float32)
        X2s = tf.reduce_sum(tf.square(X2), 1)
    square = tf.reshape(Xs, [-1, 1]) + tf.reshape(X2s, [1, -1]) - \
        tf.multiply(2.0, tf.matmul(X, X2, transpose_b=True))
    output = variance * tf.exp(-tf.cast(square, dtype=tf.float32) /
                               tf.multiply(tf.constant(2.0, dtype=tf.float32),
                                           tf.square(lengthscale)))
    output = output + (tf.multiply(tf.eye(int(X.shape[0])), noise_sq))
    grad_lengthscale = tf.multiply(output, (-2.0)*square/(lengthscale**3.0))
    grad_variance = tf.multiply(2.0*tf.sqrt(variance),
                                tf.exp(-tf.cast(square, dtype=tf.float32) /
                                       tf.multiply(2.0,
                                                   tf.square(lengthscale))))
    grad_noise = 2 * noise
    return output, grad_lengthscale, grad_variance, grad_noise


def matern_fun(X, X2=None, lengthscale_in=1.0, gamma_in=3/2):
    lengthscale = tf.constant(lengthscale_in, dtype='float32')
    X = tf.convert_to_tensor(X, tf.float32)
    Xs = tf.reduce_sum(tf.square(X), 1)
    if X2 is None:
        X2 = X
        X2s = Xs
    else:
        X2 = tf.convert_to_tensor(X2, tf.float32)
        X2s = tf.reduce_sum(tf.square(X2), 1)
    square = tf.reshape(Xs, [-1, 1]) + tf.reshape(X2s, [1, -1]) - \
        2 * tf.matmul(X, X2, transpose_b=True)
    temp_sess = tf.Session()
    abs_distance = sqrtm(temp_sess.run(square))
    cube_distance = tf.convert_to_tensor(np.power(abs_distance, 3), tf.float32)
    if gamma_in == 3/2:
        output = (1.0 + (tf.sqrt(3.0)*tf.sqrt(square))/lengthscale) * \
            tf.exp(-(tf.sqrt(3.0)*tf.sqrt(square))/lengthscale)
        grad = tf.multiply(3.0*square/lengthscale**2,
                           tf.exp(-tf.sqrt(3.0)*tf.sqrt(square)/lengthscale))
        return output, grad
    if gamma_in == 5/2:
        output = (1.0 + tf.sqrt(5.0)*tf.sqrt(square)/lengthscale +
                  5.0*square/(3.0*lengthscale**2)) * \
                  tf.exp(-(tf.sqrt(5.0)*tf.sqrt(square)/lengthscale))
        grad = ((5.0*square/lengthscale**3) +
                ((5.0*tf.sqrt(5.0) * cube_distance) -
                 10.0*square))/(3*lengthscale**4)
        grad = grad * tf.exp(-tf.sqrt(5.0)*tf.sqrt(square)/lengthscale)
        return output, grad


def rat_quadratic_fun(X, X2=None, magnitude=1.0, lengthscale=1.0,
                      diffuseness=1.0):
    magnitude = tf.constant(magnitude, tf.float32)
    lengthscale = tf.constant(lengthscale, tf.float32)
    diffuseness = tf.constant(diffuseness, tf.float32)
    X = tf.convert_to_tensor(X, tf.float32)
    Xs = tf.reduce_sum(tf.square(X), 1)
    if X2 is None:
        X2 = X
        X2s = Xs
    else:
        X2 = tf.convert_to_tensor(X2, tf.float32)
        X2s = tf.reduce_sum(tf.square(X2), 1)
    square = tf.reshape(Xs, [-1, 1]) + tf.reshape(X2s, [1, -1]) - \
        2 * tf.matmul(X, X2, transpose_b=True)
    parenthesis = 1 + square/(2*(lengthscale**2)*diffuseness)
    output = (magnitude**2)*tf.pow(parenthesis, -diffuseness)
    grad_magnitude = (2*magnitude)*tf.pow(parenthesis, -diffuseness)
    grad_lengthscale = tf.multiply(tf.multiply((magnitude**2)/lengthscale**3,
                                   square),
                                   tf.pow(parenthesis, - (diffuseness+1)))
    grad_diffuseness = tf.multiply(-output,
                                   (tf.divide(square,
                                              ((2 * (diffuseness**2) *
                                                (lengthscale**2)) +
                                               (diffuseness * square))) -
                                    2*tf.log(magnitude) - tf.log(parenthesis)))
    return output, grad_magnitude, grad_lengthscale, grad_diffuseness


def diff_loocv_each(K_mat, y_vec, diff_k_theta_in):
    y_vec = tf.reshape(tf.cast(y_vec, tf.float32), [-1, 1])
    K_mat_inv = tf.linalg.inv(K_mat)
    alpha = tf.matmul(K_mat_inv, y_vec)
    z_j = tf.matmul(K_mat_inv, diff_k_theta_in)
    first_diff_l_theta = -tf.matrix_diag_part(K_mat_inv)
    second_diff_l_theta = tf.multiply(alpha, tf.matmul(z_j, alpha)) - \
        (1.0/2.0)*tf.multiply(1.0 + tf.divide(tf.square(alpha),
                                              tf.matrix_diag_part(K_mat_inv)),
                              tf.matrix_diag_part(tf.matmul(z_j, K_mat_inv)))
    diff_l_theta = tf.multiply(first_diff_l_theta, second_diff_l_theta)
    diff_out = tf.reduce_sum(diff_l_theta)
    temp_sess = tf.Session()
    output_grad = temp_sess.run(diff_out)
    return output_grad


def pred_f_and_cov(X_input, y_input, X_star_input, fun_form, sigma_sq_n,
                   param_list):
    X_input = tf.cast(X_input, tf.float32)
    y_input = tf.reshape(tf.cast(y_input, tf.float32), [-1, 1])
    X_star_input = tf.reshape(tf.cast(X_star_input, tf.float32), [1, -1])
    if fun_form == 'rbf':
        lengthscale_in = param_list[0]
        variance_in = param_list[1]
        f_new_1 = rbf_fun(X_star_input, X_input,
                          lengthscale_in, variance_in)[0]
        f_new_2 = tf.linalg.inv(rbf_fun(X=X_input, lengthscale=lengthscale_in,
                                        variance=variance_in)[0] +
                                tf.multiply(sigma_sq_n,
                                            tf.eye(int(X_input.shape[0]))))
        f_new = tf.matmul(f_new_1, f_new_2)
        f_new = tf.matmul(f_new, y_input)
        cov_new_1 = rbf_fun(X_star_input, lengthscale=lengthscale_in,
                            variance=variance_in)[0]
        cov_new_2_1 = rbf_fun(X_star_input, X_input,
                              lengthscale_in, variance_in)[0]
        cov_new_2_2_1 = rbf_fun(X_input, lengthscale=lengthscale_in,
                                variance=variance_in)[0] + \
            tf.multiply(sigma_sq_n, tf.eye(int(X_input.shape[0])))
        cov_new_2_2_1 = tf.linalg.inv(cov_new_2_2_1)
        cov_new_2_2_2 = rbf_fun(X_input, X_star_input,
                                lengthscale_in, variance_in)[0]
        cov_new_2_2 = tf.matmul(cov_new_2_2_1, cov_new_2_2_2)
        cov_new_2 = tf.matmul(cov_new_2_1, cov_new_2_2)
        cov_new = cov_new_1 - cov_new_2
        return f_new, cov_new
    elif fun_form == 'matern':
        lengthscale_in = param_list[0]
        gamma_in = param_list[1]
        f_new_1 = matern_fun(X_star_input, X_input,
                             lengthscale_in=lengthscale_in,
                             gamma_in=gamma_in)[0]
        f_new_2 = tf.linalg.inv(matern_fun(X=X_input,
                                           lengthscale_in=lengthscale_in,
                                           gamma_in=gamma_in)[0] +
                                tf.multiply(sigma_sq_n,
                                            tf.eye(int(X_input.shape[0]))))
        f_new = tf.matmul(f_new_1, f_new_2)
        f_new = tf.matmul(f_new, y_input)
        cov_new_1 = matern_fun(X_star_input, lengthscale_in=lengthscale_in,
                               gamma_in=gamma_in)[0]
        cov_new_2_1 = matern_fun(X_star_input, X_input,
                                 lengthscale_in=lengthscale_in,
                                 gamma_in=gamma_in)[0]
        cov_new_2_2_1 = matern_fun(X_input, lengthscale_in=lengthscale_in,
                                   gamma_in=gamma_in)[0] + \
            tf.multiply(sigma_sq_n, tf.eye(int(X_input.shape[0])))
        cov_new_2_2_1 = tf.linalg.inv(cov_new_2_2_1)
        cov_new_2_2_2 = matern_fun(X_input, X_star_input,
                                   lengthscale_in, variance_in)[0]
        cov_new_2_2 = tf.matmul(cov_new_2_2_1, cov_new_2_2_2)
        cov_new_2 = tf.matmul(cov_new_2_1, cov_new_2_2)
        return f_new, cov_new
    elif fun_form == 'rat_quad':
        magnitude_in = param_list[0]
        lengthscale_in = param_list[1]
        diffuseness_in = param_list[2]
        f_new_1 = rat_quadratic_fun(X_star_input, X_input,
                                    magnitude=magnitude_in,
                                    lengthscale=lengthscale_in,
                                    diffuseness=diffuseness_in)[0]
        f_new_2 = tf.linalg.inv(rat_quadratic_fun(X_input, None,
                                                  magnitude_in,
                                                  lengthscale_in,
                                                  diffuseness_in)[0] +
                                tf.multiply(sigma_sq_n,
                                            tf.eye(int(X_input.shape[0]))))
        f_new = tf.matmul(f_new_1, f_new_2)
        f_new = tf.matmul(f_new, y_input)
        cov_new_1 = rat_quadratic_fun(X_star_input,
                                      magnitude=magnitude_in,
                                      lengthscale=lengthscale_in,
                                      diffuseness=diffuseness_in)[0]
        cov_new_2_1 = rat_quadratic_fun(X_star_input, X_input,
                                        magnitude=magnitude_in,
                                        lengthscale=lengthscale_in,
                                        diffuseness=diffuseness_in)[0]
        cov_new_2_2_1 = rat_quadratic_fun(X_input,
                                          magnitude=magnitude_in,
                                          lengthscale=lengthscale_in,
                                          diffuseness=diffuseness_in)[0] +\
            tf.multiply(sigma_sq_n, tf.eye(int(X_input.shape[0])))
        cov_new_2_2_1 = tf.linalg.inv(cov_new_2_2_1)
        cov_new_2_2_2 = rat_quadratic_fun(X_input, X_star_input,
                                          magnitude_in,
                                          lengthscale_in,
                                          diffuseness_in)[0]
        cov_new_2_2 = tf.matmul(cov_new_2_2_1, cov_new_2_2_2)
        cov_new_2 = tf.matmul(cov_new_2_1, cov_new_2_2)
        return f_new, cov_new
    else:
        # rbf part
        lengthscale_in_rbf = param_list[0]
        variance_in_rbf = param_list[1]
        # matern part
        lengthscale_in_matern = param_list[2]
        gamma_in_matern = param_list[3]
        # rational quadratic part
        magnitude_rq = param_list[4]
        lengthscale_rq = param_list[5]
        diffuseness_rq = param_list[6]
        x_star_x_part = rbf_fun(X_star_input,
                                X_input,
                                lengthscale=lengthscale_in_rbf,
                                variance=variance_in_rbf)[0]
        x_star_x_part = tf.add(x_star_x_part,
                               matern_fun(X_star_input, X_input,
                                          lengthscale_in=lengthscale_in_matern,
                                          gamma_in=gamma_in_matern)[0])
        x_star_x_part = tf.add(x_star_x_part,
                               rat_quadratic_fun(X_star_input,
                                                 X_input,
                                                 magnitude_rq,
                                                 lengthscale_rq,
                                                 diffuseness_rq)[0])
        x_only_part = rbf_fun(X_input, lengthscale=lengthscale_in_rbf,
                              variance=variance_in_rbf)[0]
        x_only_part = tf.add(x_only_part,
                             matern_fun(X_input,
                                        lengthscale_in=lengthscale_in_matern,
                                        gamma_in=gamma_in_matern)[0])
        x_only_part = tf.add(x_only_part,
                             rat_quadratic_fun(X_input,
                                               magnitude=magnitude_rq,
                                               lengthscale=lengthscale_rq,
                                               diffuseness=diffuseness_rq)[0])
        x_only_part = tf.add(x_only_part,
                             tf.multiply(sigma_sq_n,
                                         tf.eye(int(X_input.shape[0]))))
        x_only_part_inv = tf.linalg.inv(x_only_part)
        x_x_star_part = rbf_fun(X_input, X_star_input,
                                lengthscale=lengthscale_in_rbf,
                                variance=variance_in_rbf)[0]
        x_x_star_part = tf.add(x_x_star_part,
                               matern_fun(X_input,
                                          X_star_input,
                                          lengthscale_in=lengthscale_in_matern,
                                          gamma_in=gamma_in_matern)[0])
        x_x_star_part = tf.add(x_x_star_part,
                               rat_quadratic_fun(X_input,
                                                 X_star_input,
                                                 magnitude=magnitude_rq,
                                                 lengthscale=lengthscale_rq,
                                                 diffuseness=diffuseness_rq)[0]
                               )
        x_star_part = rbf_fun(X_star_input,
                              lengthscale=lengthscale_in_rbf,
                              variance=variance_in_rbf)[0]
        x_star_part = tf.add(x_star_part,
                             matern_fun(X_star_input,
                                        lengthscale_in=lengthscale_in_matern,
                                        gamma_in=gamma_in_matern)[0])
        x_star_part = tf.add(x_star_part,
                             rat_quadratic_fun(X_star_input,
                                               magnitude=magnitude_rq,
                                               lengthscale=lengthscale_rq,
                                               diffuseness=diffuseness_rq)[0])
        matmul_part = tf.matmul(tf.matmul(x_star_x_part,
                                          x_only_part_inv), x_x_star_part)
        f_new = tf.matmul(matmul_part, y_input)
        cov_new = x_star_part + matmul_part
    return f_new, cov_new


def val_loocv(X_input, y_input, param_in, sigma_sq_in,
              max_VI_iter, qf_in, mean_prior=0):
    f_pred_all = np.zeros(X_input.shape[0])
    loo = LeaveOneOut()
    temp_sess = tf.Session()
    N = int(X_input.shape[0])
    D = int(X_input.shape[1])
    for train_index, test_index in loo.split(X_input):
        X_star_input = X_input[test_index, :].reshape(1, -1)
        X_other_input = X_input[train_index, :]
        y_other_input = y_input[train_index].reshape(-1, 1)
        k_star = rbf_fun(X_other_input, X_star_input,
                         lengthscale=param_in[0], variance=param_in[1])[0]
        k_star_1 = matern_fun(X_other_input, X_star_input,
                              lengthscale_in=param_in[2],
                              gamma_in=param_in[3])[0]
        k_star_2 = rat_quadratic_fun(X_other_input,
                                     X_star_input,
                                     magnitude=param_in[4],
                                     lengthscale=param_in[5],
                                     diffuseness=param_in[6])[0]
        k_star_all = tf.add(tf.add(k_star, k_star_1), k_star_2)
        x_only_part = rbf_fun(X_other_input,
                              lengthscale=param_in[0],
                              variance=param_in[1])[0]
        x_only_part = tf.add(x_only_part,
                             matern_fun(X_other_input,
                                        lengthscale_in=param_in[2],
                                        gamma_in=param_in[3])[0])
        x_only_part = tf.add(x_only_part,
                             rat_quadratic_fun(X_other_input,
                                               magnitude=param_in[4],
                                               lengthscale=param_in[5],
                                               diffuseness=param_in[6])[0])
        x_only_part = tf.add(x_only_part,
                             tf.multiply(sigma_sq_in,
                                         tf.eye(int(X_input.shape[0]))))
        x_only_part_inv = tf.linalg.inv(x_only_part)
        # Inference from Edward Part
        X = tf.placeholder(tf.float32, [N-1, D])
        f = MultivariateNormalTriL(loc=tf.zeros(N-1),
                                   scale_tril=tf.cholesky(x_only_part))
        y = Poisson(rate=tf.nn.softplus(f))
        w_mat = tf.matmul(x_only_part_inv, k_star_all)
        y_other_input = tf.reshape(y_other_input, [-1])
        y_other_input = tf.cast(y_other_input, dtype=tf.float32)
        inference_vi = ed.KLqp({f: qf_in},
                               data={X: X_other_input, y: y_other_input})
        inference_vi.run(n_iter=max_VI_iter)
        y_post = ed.copy(y, {f: qf_in})
        m_mat = y_post.eval()
        f_star_each = mean_prior + \
            tf.matmul(tf.transpose(w_mat),
                      (tf.reshape(y_other_input, [-1, 1]) - m_mat))
        f_pred_all[test_index] = temp_sess.run(f_star_each)
    sum_sq_err = np.sum(np.square(y_input - f_pred_all))
    return f_pred_all, sum_sq_err


def grad_descent(X_input, y_input, fun_form,
                 param_list, learning_rate, max_iter):
    if fun_form == 'rbf':
        for n in range(max_iter):
            temp_rbf = rbf_fun(X_input, lengthscale=param_list[0],
                               variance=param_list[1])
            K_mat = temp_rbf[0]
            grad_length = temp_rbf[1]
            grad_var = temp_rbf[2]
            diff_param_length = diff_loocv_each(K_mat, y_input, grad_length)
            diff_param_var = diff_loocv_each(K_mat, y_input, grad_var)
            param_list[0] = param_list[0] - (diff_param_length * learning_rate)
            param_list[1] = param_list[1] - (diff_param_var * learning_rate)
            if abs(diff_param_length) <= 0.001 or abs(diff_param_var) <= 0.001:
                break
        return param_list

    elif fun_form == 'rat_quadratic':
        for n in range(max_iter):
            temp_quad_fun = rat_quadratic_fun(X_input, magnitude=param_list[0],
                                              lengthscale=param_list[1],
                                              diffuseness=param_list[2])
            K_mat = temp_quad_fun[0]
            g_m = temp_quad_fun[1]
            g_l = temp_quad_fun[2]
            g_d = temp_quad_fun[3]
            diff_param_m = diff_loocv_each(K_mat, y_input, g_m)
            diff_param_l = diff_loocv_each(K_mat, y_input, g_l)
            diff_param_d = diff_loocv_each(K_mat, y_input, g_d)
            param_list[0] = param_list[0] - (diff_param_m * learning_rate)
            param_list[1] = param_list[1] - (diff_param_l * learning_rate)
            param_list[2] = param_list[2] - (diff_param_d * learning_rate)
            if ((abs(diff_param_m) <= 0.001 or
                 abs(diff_param_l) <= 0.001 or
                 abs(diff_param_d) <= 0.001)):
                break
        return param_list
    else:
        for n in range(max_iter):
            temp_matern = matern_fun(X_input,
                                     lengthscale_in=param_list[0])
            K_mat = temp_matern[0]
            grad = temp_matern[1]
            diff_param_len = diff_loocv_each(K_mat, y_input, grad)
            param_list[0] = param_list[0] - (diff_param_len*learning_rate)
            if abs(diff_param_len) <= 0.001:
                break
        return param_list


def classify_y(y_input):
    empty_list = list()
    for each_val in y_input:
        if each_val == 0:
            empty_list.append(0)
        elif each_val > 0 and each_val <= 3:
            empty_list.append(1)
        else:
            empty_list.append(2)
    return np.array(empty_list)


def lmnn_apply(x_input, y_input, k_input):
    y_classified = classify_y(y_input)
    lmnn = LMNN(k=k_input, learn_rate=1e-6)
    x_output = lmnn.fit_transform(x_input, y_classified)
    return x_output


def pca_apply(x_input, k_input):
    pca_mod = PCA(n_components=k_input)
    x_out = pca_mod.fit_transform(x_input)
    return x_out


# Data importation
year_vec = np.array(range(1856, 2019))
dict_all_data = dict()
iter_year = 0
dat_each_year = list()

# This part assumes that the data file and the code are in the same directory
# If not, please edit the directory below

with open('ssta_dat.tsv') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    for row in reader:
        indicator = row[0].strip()
        if not re.match('May', indicator):
            each_dat = row[1:len(row)]
            for i in range(len(each_dat)):
                each_dat[i] = float(each_dat[i].strip())
            dat_each_year.extend(each_dat)
        else:
            if len(dat_each_year) != 0:
                dict_all_data[year_vec[iter_year]] = np.array(dat_each_year)
                dat_each_year = list()
                iter_year += 1
    dict_all_data[year_vec[iter_year]] = np.array(dat_each_year)

d_all_dat = dict_all_data
dat_out = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d_all_dat.items()]))
dat_out_final = dat_out.T
temp = dat_out_final.columns[dat_out_final.apply(lambda col: col.mean() > 100)]
dat_out_final = dat_out_final.drop(temp, axis=1)

y_val_dict = dict()
filepath = 'newlandfreq.txt'
with open(filepath) as fp:
    each_line = fp.readline()
    while each_line:
        temp_split = each_line.split()
        for i in range(len(temp_split)):
            temp_split[i] = float(temp_split[i])
        y_val_dict[temp_split[0]] = np.array(temp_split[1:len(temp_split)])
        each_line = fp.readline()

y_out = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in y_val_dict.items()]))
y_out_final = y_out.T
y_out_final = y_out_final.loc[1856:, ].reset_index()
dat_out_final = dat_out_final.loc[:2016, ].reset_index()
y_train_full = y_out_final.iloc[:, 1]
y_train_full = y_train_full.to_frame().reset_index()

dat_out_final = dat_out_final.values
y_train_full = y_train_full.iloc[:, 1].values

N = dat_out_final.shape[0]
D = dat_out_final.shape[1]

y_train_scaled = y_train_full
x_train_scaled = varscale(dat_out_final)

param_list_1_x_full = grad_descent(x_train_scaled,
                                   y_train_scaled, 'rbf',
                                   [1.0, 1.0], 0.000035, 1000)
param_list_2_x_full = grad_descent(x_train_scaled,
                                   y_train_scaled, 'matern',
                                   [1.0], 0.000035, 1000)
param_list_3_x_full = grad_descent(x_train_scaled, y_train_scaled,
                                   'rat_quadratic', [1.0, 1.3, 1.6],
                                   0.000013, 1000)
param_list_x_full_all = param_list_1_x_full.extend(param_list_2_x_full)
param_list_x_full_all = param_list_x_full_all.extend(param_list_3_x_full)

qf_test = Normal(loc=tf.get_variable('qf/loc3', [N-1]),
                 scale=tf.nn.softplus(tf.get_variable('qf/scale3', N-1)))

log_loo, mu_out, cov_out = val_loocv(X_input=x_train_scaled,
                                     y_input=y_train_scaled,
                                     param_in=param_list_x_full_all,
                                     sigma_sq_in=1e-32, max_VI_iter=1000,
                                     qf_in=qf_test)

error_list_lmnn = list()
error_list_pca = list()

for k in range(3, 11):
    x_lmnn = lmnn_apply(x_train_scaled, y_train_scaled, k)
    param_list_1_x_lmnn = grad_descent(x_lmnn, y_train_scaled,
                                       'rbf', [1.0, 1.0, 1.0],
                                       0.000035, 1000)
    param_list_2_x_lmnn = grad_descent(x_lmnn, y_train_scaled,
                                       'matern', [1.0, 1.0, 1.0],
                                       0.000035, 1000)
    param_list_3_x_lmnn = grad_descent(x_lmnn, y_train_scaled,
                                       'rat_quadratic', [1.0, 1.3, 1.6],
                                       0.000013, 1000)
    param_list_x_lmnn_all = param_list_1_x_lmnn.extend(param_list_2_x_lmnn)
    param_list_x_lmnn_all = param_list_x_lmnn_all.extend(param_list_3_x_lmnn)
    log_loo, mu_out, cov_out = val_loocv(X_input=x_lmnn,
                                         y_input=y_train_scaled,
                                         param_in=param_list_x_lmnn_all,
                                         sigma_sq_in=1e-32,
                                         max_VI_iter=1000, qf_in=qf_test)
    error_list_lmnn.append(np.sqrt(np.sum(np.square(y_train_scaled - mu_out))))

for k in range(3, 11):
    x_pca = pca_apply(x_train_scaled, y_train_scaled, k)
    param_list_1_x_pca = grad_descent(x_pca, y_train_scaled,
                                      'rbf', [1.0, 1.0, 1.0], 0.000035, 1000)
    param_list_2_x_pca = grad_descent(x_pca, y_train_scaled,
                                      'matern', [1.0, 1.0, 1.0],
                                      0.000035, 1000)
    param_list_3_x_pca = grad_descent(x_pca, y_train_scaled,
                                      'rat_quadratic', [1.0, 1.3, 1.6],
                                      0.000013, 1000)
    param_list_x_pca_all = param_list_1_x_pca.extend(param_list_2_x_pca)
    param_list_x_pca_all = param_list_x_pca_all.extend(param_list_3_x_pca)
    log_loo, mu_out, cov_out = val_loocv(X_input=x_pca,
                                         y_input=y_train_scaled,
                                         param_in=param_list_x_pca_all,
                                         sigma_sq_in=1e-32, max_VI_iter=1000,
                                         qf_in=qf_test)
    error_list_pca.append(np.sqrt(np.sum(np.square(y_train_scaled - mu_out))))
