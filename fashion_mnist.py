# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.import tensorflow as tf

import tensorflow as tf
import argparse
import os
import numpy as np
import json


def model(x_train, y_train, x_test, y_test, epochs=1):
    """Generate a simple model"""
    # Hands-On Machine Learning book
    # nono.ma/book/hands-on-ml
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=[28,28]),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])    

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs)
    model.evaluate(x_test, y_test)

    return model


def _load_data(base_dir):
    """Load MNIST training and testing data"""
    data = np.load(os.path.join(base_dir, 'fashion_mnist.npy'), allow_pickle=True)
    return data

def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
    parser.add_argument('--epochs', type=int, default=1)

    return parser.parse_known_args()

# List devices available to TensorFlow
from tensorflow.python.client import device_lib

if __name__ == "__main__":
    args, unknown = _parse_args()

    print(args)
    #print('SN_MODEL_DIR: {}\n\n'.format(args.SM_MODEL_DIR))
    
    print('\n\nDEVICES\n\n')    
    print(device_lib.list_local_devices())
    print('\n\n')
    
    print('Loading Fashion MNIST data..\n')
    (train_data, train_labels), (eval_data, eval_labels) = _load_data(args.train)

    print('Training model for {} epochs..\n\n'.format(args.epochs))
    mnist_classifier = model(train_data, train_labels, eval_data, eval_labels, epochs=args.epochs)

    if args.current_host == args.hosts[0]:
        # save model to an S3 directory with version number '00000001'
        mnist_classifier.save(os.path.join(args.sm_model_dir, '000000001'), 'my_model.h5')