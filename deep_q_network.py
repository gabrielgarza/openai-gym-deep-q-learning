"""
Deep Q Network Reinforcement Learning
Uses two neural networks with 3 layers each

"""

import tensorflow as tf
import numpy as np


class DeepQNetwork:
    def __init__(
        self,
        n_y,
        n_x,
        learning_rate=0.01,
        replace_target_iter=100,
        memory_size=1000,
        epsilon_max=0.9,
        epsilon_greedy_increment=0.001,
        batch_size=32,
        reward_decay=0.9,
        load_path=None,
        save_path=None
    ):

        self.n_y = n_y
        self.n_x = n_x
        self.lr = learning_rate
        self.epsilon_max = epsilon_max
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.epsilon_greedy_increment = epsilon_greedy_increment
        self.batch_size = batch_size
        self.reward_decay = reward_decay # this is gamma

        if save_path is not None:
            self.save_path = save_path

        self.memory_counter = 0
        self.learn_step_counter = 0

        if epsilon_greedy_increment is not None:
            self.epsilon = 0
        else:
            self.epsilon = self.epsilon_max

        # Initialize memory
        self.memory_s = np.zeros((n_x,self.memory_size))
        self.memory_a = np.zeros((self.memory_size))
        self.memory_r = np.zeros((self.memory_size))
        self.memory_s_ = np.zeros((n_x,self.memory_size))

        # Config for networks
        n_l1 = 10
        n_l2 = 10
        W_init = tf.contrib.layers.xavier_initializer(seed=1)
        b_init = tf.contrib.layers.xavier_initializer(seed=1)
        self.build_eval_network(n_l1, n_l2, W_init, b_init)
        self.build_target_network(n_l1, n_l2, W_init, b_init)

        self.sess = tf.Session()

        self.cost_history = []

        # $ tensorboard --logdir=logs
        # http://0.0.0.0:6006/
        tf.summary.FileWriter("logs/", self.sess.graph)

        init = tf.global_variables_initializer()
        self.sess.run(init)

        # 'Saver' op to save and restore all the variables
        self.saver = tf.train.Saver()

        # Restore model
        if load_path is not None:
            self.load_path = load_path
            self.saver.restore(self.sess, self.load_path)

    def store_transition(self, s, a, r, s_):
        # Replace old memory with new memory
        index = self.memory_counter % self.memory_size

        self.memory_s[:,index] = s
        self.memory_a[index] = a
        self.memory_r[index] = r
        self.memory_s_[:,index] = s_

        self.memory_counter += 1

    def choose_action(self, observation):
        # Reshape to (num_features, 1)
        observation = observation[ :,np.newaxis ]

        # If random sample from uniform distribution is less than the epsilon parameter then predict action, else take a random action
        if np.random.uniform() < self.epsilon:
            # Forward propagate to get q values of outputs
            actions_q_value = self.sess.run(self.q_eval_outputs, feed_dict={self.X: observation})

            # Get index of maximum q value
            action = np.argmax(actions_q_value)
        else:
            # Random action
            action = np.random.randint(0, self.n_y)

        return action

    def replace_target_net_parameters(self):
        print("target parameters replaced")
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')

        # Assign the parameters trained in the eval net to the target net
        self.sess.run( [ tf.assign(t,e) for t, e in zip(t_params, e_params) ] )

    def learn(self):
        # Replace target params
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.replace_target_net_parameters()

        # Save checkpoint
        if self.learn_step_counter % (self.replace_target_iter * 10) == 0:
            if self.save_path is not None:
                save_path = self.saver.save(self.sess, self.save_path)
                print("Model saved in file: %s" % save_path)

        # Get a memory sample
        index_range = min(self.memory_counter, self.memory_size)
        sample_index = np.random.choice(index_range, size=self.batch_size)

        batch_memory_s = self.memory_s[ :,sample_index ]
        batch_memory_a = self.memory_a[ sample_index ]
        batch_memory_r = self.memory_r[ sample_index ]
        batch_memory_s_ = self.memory_s_[ :,sample_index ]

        # Forward propagate eval and target nets to get q values of actions
        q_next_outputs, q_eval_outputs = self.sess.run([self.q_next_outputs, self.q_eval_outputs], feed_dict={
            self.X_: batch_memory_s_,
            self.X: batch_memory_s
        })

        # Create copy of eval net outputs that we just forward propagated
        q_target_outputs = q_eval_outputs.copy()

        # Setup array of index for batch e.g. for batch size 32 it will be [0, 1, 2, ...31]
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # Get memory actions
        actions_index = batch_memory_a.astype(int)

        # Generate Q target values with Bellman equation
        q_target_outputs[ actions_index, batch_index ] = batch_memory_r + self.reward_decay * np.max(q_next_outputs, axis=0)

        # Train eval network
        _, self.cost = self.sess.run([self.train_op, self.loss], feed_dict={ self.X: batch_memory_s, self.Y: q_target_outputs } )

        # Save cost
        self.cost_history.append(self.cost)

        # Increase epsilon to make it more likely over time to get actions from predictions instead of from random sample
        self.epsilon = min(self.epsilon_max, self.epsilon + self.epsilon_greedy_increment)
        self.learn_step_counter += 1

    def build_eval_network(self, n_l1, n_l2, W_init, b_init):
        ###########
        # EVAL NET
        ###########
        self.X = tf.placeholder(tf.float32, [self.n_x, None], name='s')
        self.Y = tf.placeholder(tf.float32, [self.n_y, None ], name='Q_target')

        with tf.variable_scope('eval_net'):
            # Store variables in collection
            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            with tf.variable_scope('parameters'):
                W1 = tf.get_variable('W1', [n_l1, self.n_x], initializer=W_init, collections=c_names)
                b1 = tf.get_variable('b1', [n_l1, 1], initializer=b_init, collections=c_names)
                W2 = tf.get_variable('W2', [n_l2, n_l1], initializer=W_init, collections=c_names)
                b2 = tf.get_variable('b2', [n_l2, 1], initializer=b_init, collections=c_names)
                W3 = tf.get_variable('W3', [self.n_y, n_l2], initializer=W_init, collections=c_names)
                b3 = tf.get_variable('b3', [self.n_y, 1], initializer=b_init, collections=c_names)

            # First layer
            with tf.variable_scope('layer_1'):
                Z1 = tf.matmul(W1, self.X) + b1
                A1 = tf.nn.relu( Z1 )
            # Second layer
            with tf.variable_scope('layer_2'):
                Z2 = tf.matmul(W2, A1) + b2
                A2 = tf.nn.relu( Z2 )
            # Output layer
            with tf.variable_scope('layer_3'):
                Z3 = tf.matmul(W3, A2) + b3
                self.q_eval_outputs = Z3

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.Y, self.q_eval_outputs))
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def build_target_network(self, n_l1, n_l2, W_init, b_init):
        ############
        # TARGET NET
        ############
        self.X_ = tf.placeholder(tf.float32, [self.n_x, None], name="s_")

        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            with tf.variable_scope('parameters'):
                W1 = tf.get_variable('W1', [n_l1, self.n_x], initializer=W_init, collections=c_names)
                b1 = tf.get_variable('b1', [n_l1, 1], initializer=b_init, collections=c_names)
                W2 = tf.get_variable('W2', [n_l2, n_l1], initializer=W_init, collections=c_names)
                b2 = tf.get_variable('b2', [n_l2, 1], initializer=b_init, collections=c_names)
                W3 = tf.get_variable('W3', [self.n_y, n_l2], initializer=W_init, collections=c_names)
                b3 = tf.get_variable('b3', [self.n_y, 1], initializer=b_init, collections=c_names)

            # First layer
            with tf.variable_scope('layer_1'):
                Z1 = tf.matmul(W1, self.X_) + b1
                A1 = tf.nn.relu( Z1 )
            # Second layer
            with tf.variable_scope('layer_2'):
                Z2 = tf.matmul(W2, A1) + b2
                A2 = tf.nn.relu( Z2 )
            # Output layer
            with tf.variable_scope('layer_3'):
                Z3 = tf.matmul(W3, A2) + b3
                self.q_next_outputs = Z3

    def plot_cost(self):
        import matplotlib
        matplotlib.use("MacOSX")
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_history)), self.cost_history)
        plt.ylabel('Cost')
        plt.xlabel('Training Steps')
        plt.show()
