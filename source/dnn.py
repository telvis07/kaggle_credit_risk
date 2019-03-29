"""
author: Telvis Calhoun

Train a DNN using Tensorflow using specified parameters
"""
import os
import shutil
import scipy
import json

import tensorflow as tf
import numpy as np

from sklearn.metrics import (precision_recall_curve, average_precision_score,
                             precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, accuracy_score)

from preprocess import load_train_data_train_test_split, load_train_and_kaggle_submission
from utils import plot_roc_curve_interp
import matplotlib.pyplot as plt

PARAM_ELU = "elu"
PARAM_RELU = "relu"


def leaky_relu(z, name=None):
    """
    Leaky Relu example from "Nonsaturating Activation Functions"
    from the book: "Hands-On Machine Learning with Scikit-Learn and TensorFlow"

    :param z: output from hidden layer
    :param name: layer name
    :return: output of leaky relu function
    """
    return tf.maximum(0.01 * z, z, name=name)


def api_dnn(train_fn, results_output_json_fn, test_fn=None,
            model_dir="./model/credit_risk_dnn", npzdir="./npzdir",
            b_show_plot=False, output_png=None, model_config=None,
            kaggle_submission_output_csv="submission.csv"):
    """
    Function to build DNN with parameters and generate kaggle submission.

    :param train_fn: training file with all features merged into one file
    :param results_output_json_fn: file path to store JSON with experiment parameters and perf results
    :param test_fn: testing file containing Kaggle submission entries
    :param model_dir: Directory path to store the model
    :param npzdir: Directory path to store predict() and predict_proba() output
    :param b_show_plot: Boolean to show the plot (if running from jupyter notebook)
    :param output_png: File path to write ROC curve PNG
    :param model_config: DNN configuration
    :param kaggle_submission_output_csv: File path to write Kaggle submission output
    :return: Python dictionary containing model experiment performance results
    """

    for _dir in [npzdir, model_dir]:
        if os.path.isdir(_dir):
            shutil.rmtree(_dir)

        os.makedirs(_dir)
        print("mkdir",_dir)

    X_train, X_test, y_train, y_test, scaler = load_train_data_train_test_split(train_fn=train_fn)
    X_ktest, submission_df = load_train_and_kaggle_submission(test_fn=test_fn, scaler=scaler)


    model_config["n_inputs"] = X_train.shape[1]
    n_epochs = model_config["n_epoch"]
    batch_size = model_config["batch_size"]

    print("DNN Config", json.dumps(model_config, indent=2))

    n_inputs = model_config["n_inputs"]
    n_hidden1 = model_config["n_hidden1"]
    n_hidden2 = model_config["n_hidden2"]
    n_outputs = model_config["n_outputs"]
    nn_type = model_config["nn_type"]
    learning_rate = model_config["learning_rate"]
    do_gradient_clipping = model_config["do_gradient_clipping"]
    do_metric_type = model_config["do_metric_type"]

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int64, shape=(None), name="y")
    pos_weight_val = 0

    # training placeholder: set to True when training
    # This will be used to tell the tf.layers.batch_normalization() function whether it
    # (True) use the current mini-batch mean and standard deviation (during training)
    # or
    # (False) use the whole training set's mean and standard deviation (during testing)
    training = tf.placeholder_with_default(False, shape=(), name="training")
    pos_weight = tf.placeholder(tf.float32, shape=(None), name="pos_weight")

    with tf.name_scope("dnn"):
        if nn_type == "batch_norm":
            he_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

            # Layer (1) : full connected, batch norm, ELU (exponential linear unit)
            hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1",
                                      kernel_initializer=he_init)
            bn1 = tf.layers.batch_normalization(hidden1, training=training, momentum=0.9)
            bn1_act = tf.nn.elu(bn1)

            # Layer (2)
            hidden2 = tf.layers.dense(bn1_act, n_hidden2, name="hidden2",
                                      kernel_initializer=he_init)
            bn2 = tf.layers.batch_normalization(hidden2, training=training, momentum=0.9)
            bn2_act = tf.nn.elu(bn2)

            # outputs
            logits_before_bn = tf.layers.dense(bn2_act, n_outputs, name="outputs")
            logits = tf.layers.batch_normalization(logits_before_bn, training=training, momentum=0.9)
        elif nn_type == "batch_norm__l1_regularizer":
            he_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
            scale = 0.001  # l1 regularization hyperparameter

            # Layer (1) : full connected, batch norm, ELU (exponential linear unit)
            hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1",
                                      kernel_initializer=he_init,
                                      kernel_regularizer=tf.contrib.layers.l1_regularizer(scale),
                                      )
            bn1 = tf.layers.batch_normalization(hidden1, training=training, momentum=0.9)
            bn1_act = tf.nn.elu(bn1)

            # Layer (2)
            hidden2 = tf.layers.dense(bn1_act, n_hidden2, name="hidden2",
                                      kernel_initializer=he_init,
                                      kernel_regularizer=tf.contrib.layers.l1_regularizer(scale),
                                      )
            bn2 = tf.layers.batch_normalization(hidden2, training=training, momentum=0.9)
            bn2_act = tf.nn.elu(bn2)

            # outputs
            logits_before_bn = tf.layers.dense(bn2_act, n_outputs, name="outputs")
            logits = tf.layers.batch_normalization(logits_before_bn, training=training, momentum=0.9)
        elif nn_type == "l1_regularizer":
            scale = 0.001  # l1 regularization hyperparameter

            # He initialization considers only the fan-in,
            # Use Xavier initialization to consider average between fan-in, fan-out (model="FAN_AVG")
            he_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
            hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", activation=tf.nn.elu,
                                      kernel_initializer=he_init,
                                      kernel_regularizer=tf.contrib.layers.l1_regularizer(scale),
                                      )
            hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2",
                                      activation=tf.nn.elu,
                                      kernel_initializer=he_init,
                                      kernel_regularizer=tf.contrib.layers.l1_regularizer(scale))
            logits = tf.layers.dense(hidden2, n_outputs, name="outputs")
        elif nn_type == "dropout":
            dropout_rate = 0.5
            X_drop = tf.layers.dropout(X, dropout_rate, training=training)

            # layer 1 + dropout
            hidden1 = tf.layers.dense(X_drop, n_hidden1, name="hidden1", activation=tf.nn.elu)
            hidden1_drop = tf.layers.dropout(hidden1, dropout_rate, training=training)

            # layer 2 + drop
            hidden2 = tf.layers.dense(hidden1_drop, n_hidden2, name="hidden2", activation=tf.nn.elu)
            hidden2_drop = tf.layers.dropout(hidden2, dropout_rate, training=training)

            # output logits (binary classification)
            logits = tf.layers.dense(hidden2_drop, n_outputs, name="outputs")
        else:
            # He initialization considers only the fan-in,
            # Use Xavier initialization to consider average between fan-in, fan-out (model="FAN_AVG")
            he_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
            hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", activation=tf.nn.elu,
                                      kernel_initializer=he_init
                                      )
            hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.elu,
                                      kernel_initializer=he_init)
            logits = tf.layers.dense(hidden2, n_outputs, name="outputs")

    with tf.name_scope("loss"):
        # Cost function for the NN: Cross Entropy : will penalize models that estimate a low probability for the
        # target class
        # sparse_softmax_cross_entropy_with_logits: computes crosss entropy based on the "logits" (output before going
        # through the softmax function). Expects labels as integers ranging from 0 to the number of classes minus 1

        if nn_type in ("l1_regularizer", "batch_norm__l1_regularizer"):
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                                      logits=logits)
            base_loss = tf.reduce_mean(xentropy, name="avg_xentropy")
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss = tf.add_n([base_loss] + reg_losses, name="loss")
        elif nn_type in ('weighted_cross_entropy_with_logits',):
            sum_wpos = np.sum(y_train == 1.0)
            sum_wneg = np.sum(y_train == 0.0)
            pos_weight_val = int(sum_wneg / sum_wpos)
            model_config['weighted_cross_entropy_with_logits__pos_weight'] = pos_weight_val
            xentropy = tf.nn.weighted_cross_entropy_with_logits(targets=tf.cast(y, tf.float32),
                                                                pos_weight=pos_weight,
                                                                logits=logits)
            loss = tf.reduce_mean(xentropy, name="loss")
        else:
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                                      logits=logits)
            loss = tf.reduce_mean(xentropy, name="loss")

    with tf.name_scope("train"):
        # Now that we have a cost function, we define a GD optimizer that will tween model parameters
        # to minimize the cost function.

        if do_gradient_clipping:
            threshold = 1.0
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            grads_and_vars = optimizer.compute_gradients(loss)
            capped_gvs = [(tf.clip_by_value(grad, -threshold, threshold), var)
                          for grad, var in grads_and_vars]
            training_op = optimizer.apply_gradients(capped_gvs)
        else:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            training_op = optimizer.minimize(loss)

    with tf.name_scope("eval"):
        # construction phase to evaluate the model

        if do_metric_type == "precision":
            # Define the metric and update operations
            tf_metric, tf_metric_update_op = tf.metrics.precision_at_k(labels=y, predictions=logits,
                                                                       k=1, class_id=1)
        else:
            # Define the metric and update operations
            tf_metric, tf_metric_update_op = tf.metrics.recall_at_k(labels=y, predictions=logits,
                                                                    k=1, class_id=1)

        # Isolate the variables stored behind the scenes by the metric operation
        running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES)
        running_vars_initializer = tf.variables_initializer(var_list=running_vars)

    print(tf_metric)

    # initialize all variables. create Saver object to save our trained model to disk
    init = tf.global_variables_initializer()
    filename = os.path.join(model_dir, "credit_risk.ckpt")
    saver = tf.train.Saver()

    train_num_examples = X_train.shape[0]

    # update ops for metrics
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    # now we train the model
    with tf.Session() as sess:
        # init global vars
        init.run()

        # init local vars (eval)
        sess.run(running_vars_initializer)

        for epoch in range(n_epochs):
            X_batch = None
            y_batch = None
            for offset in range(0, train_num_examples, batch_size):
                end = offset + batch_size
                X_batch = X_train[offset:end]
                y_batch = y_train[offset:end]

                sess.run([training_op, extra_update_ops],
                         feed_dict={training: True,
                                    pos_weight: [pos_weight_val],
                                    X: X_batch, y: y_batch})

            metric_test = sess.run(tf_metric_update_op, feed_dict={X: X_test, y: y_test})
            print(epoch, nn_type, do_metric_type, "EvalSet", metric_test)

        save_path = saver.save(sess=sess, save_path=filename)

    # Using the model
    with tf.Session() as sess:
        saver.restore(sess, save_path)
        Z = logits.eval(feed_dict={X: X_test})
        print("logits", Z[:10])

        # np.argmax() return the class {0,1} that has the highest value
        y_pred = np.argmax(Z, axis=1)
        print("argmax", y_pred[:10])

        # Any tensor returned by Session.run or eval is a NumPy array.
        # https://stackoverflow.com/questions/34097281/how-can-i-convert-a-tensor-into-a-numpy-array-in-tensorflow
        y_pred_softmax = sess.run(tf.nn.softmax(logits=Z))
        print("softmax", y_pred_softmax[:10])

        save_predictions_npz(fn=os.path.join(npzdir, "logits.npz"), predictions=Z)
        save_predictions_npz(fn=os.path.join(npzdir, "argmax.npz"), predictions=y_pred)
        save_predictions_npz(fn=os.path.join(npzdir, "softmax.npz"), predictions=y_pred_softmax)

    # kaggle predictions
    with tf.Session() as sess:
        saver.restore(sess, save_path)
        Z = logits.eval(feed_dict={X: X_ktest})
        print("logits", Z[:10])

        submission_df["TARGET"] = np.argmax(Z, axis=1)
        # submission_df["TARGET"] = sess.run(tf.nn.softmax(logits=Z))
        submission_df.to_csv(kaggle_submission_output_csv, index=False)
        print("Wrote to", kaggle_submission_output_csv)
        print("Kaggle Prediction labels:", submission_df["TARGET"].value_counts())

    print(y_pred_softmax.shape), print(y_test.shape)
    print(y_pred_softmax[:, 1].shape)
    y_pred_proba = y_pred_softmax[:, 1]

    print("roc_auc_score", roc_auc_score(y_test, y_pred_proba))
    print("pauc(.50)", roc_auc_score(y_test, y_pred_proba, max_fpr=0.5))
    print("pauc(.05)", roc_auc_score(y_test, y_pred_proba, max_fpr=0.05))
    print("pauc(.005)", roc_auc_score(y_test, y_pred_proba, max_fpr=0.005))
    print("recall_score", recall_score(y_test, y_pred))
    print("precision_score", precision_score(y_test, y_pred))

    stats_json = {
        "roc_auc_score": roc_auc_score(y_test, y_pred_proba),
        "pauc@0.50": roc_auc_score(y_test, y_pred_proba, max_fpr=0.5),
        "pauc@0.05": roc_auc_score(y_test, y_pred_proba, max_fpr=0.05),
        "pauc@0.005": roc_auc_score(y_test, y_pred_proba, max_fpr=0.005),
        "recall_score": recall_score(y_test, y_pred),
        "precision_score": precision_score(y_test, y_pred),
        "meta": {
            "train_fn": train_fn,
            "kaggle_test_fn": test_fn,
            "kaggle_submission_output_csv": kaggle_submission_output_csv,
            "results_output_json_fn": results_output_json_fn,
            "model_dir" : model_dir,
            "b_show_plot" : b_show_plot,
            "output_png": output_png,
        },
        "dnn": model_config
    }

    # preds = y_pred_proba[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plot_roc_curve_interp(fpr=fpr, tpr=tpr, label='default', output_png=output_png)

    if b_show_plot:
        plt.show()

    tf.reset_default_graph()

    with open(results_output_json_fn, 'w') as fp:
        json.dump(stats_json, fp, indent=2)
        print("Wrote to", results_output_json_fn)

    return stats_json


def save_predictions_npz(fn, predictions):
    """
    Save the numpy array to compressed numpy array file

    :param fn: path to output file
    :param predictions: numpy array containing model predictions
    :return: None
    """
    np.savez_compressed(fn, predictions=predictions)
    print("[save_predictions_npz] Wrote to", fn)



