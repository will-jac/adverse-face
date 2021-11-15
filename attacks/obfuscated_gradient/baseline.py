### from https://github.com/anishathalye/obfuscated-gradients/baseline/baseline.ipynb

import tensorflow as tf
import numpy as np


class Attack:
    def __init__(self, model, tol, num_steps, step_size, random_start):
        self.model = model

        self.tol = tol
        self.num_steps = num_steps
        self.step_size = step_size
        self.rand = random_start

        # self.xs = tf.Variable(np.zeros((1, 32, 32, 3), dtype=np.float32), name='modifier')
        # self.orig_xs = tf.placeholder(tf.float32, [None, 32, 32, 3])

        # self.ys = tf.placeholder(tf.int32, [None])

        self.epsilon = 8

        self.optimizer = tf.keras.optimizers.Adam(self.step_size*1)


    def _perterb(self, X, y):
        delta = tf.clip_by_value(X, 0, 255) - X
        delta = tf.clip_by_value(delta, -self.epsilon, self.epsilon)

        # self.do_clip_xs = tf.assign(self.xs, self.orig_xs+delta)
        X = X + delta

        self.logits = logits = self.model(X)

        # for cifar-10
        label_mask = tf.one_hot(y, 10)

        correct_logit = tf.reduce_sum(label_mask * logits, axis=1)
        wrong_logit = tf.reduce_max((1-label_mask) * logits - 1e4*label_mask, axis=1)

        self.loss = (correct_logit - wrong_logit)

        # start_vars = set(x.name for x in tf.global_variables())
 
        # grad,var = self.optimizer.compute_gradients(self.loss, [X])[0]
        # self.train = self.optimizer.apply_gradients([(tf.sign(grad),var)])
        self.train = self.optimizer.minimize(self.loss, [X])[0]

        # end_vars = tf.global_variables()
        # self.new_vars = [x for x in end_vars if x.name not in start_vars]

        return X

    def perturb(self, x, y):
        
        # sess.run(tf.variables_initializer(self.new_vars))
        # sess.run(self.xs.initializer)
        # sess.run(self.do_clip_xs,
        #          {self.orig_xs: x})

        for i in range(self.num_steps):
            x = self._perterb(x,y)
            print(self.train)
            self.train(y)
            # sess.run(self.do_clip_xs,
            #          {self.orig_xs: x})

        # return sess.run(self.xs)
        return x

