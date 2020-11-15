"""Config file for GPT2 training.
"""
## fits to x4|x1

tfrecord_data_dir = "data/ROC_comet_clf/" 
arc_data = "data/"

lr = 0.00001

w_fine = 1.0

# rl weights
w_fine_rl = 1.0
w_rl = 0.97


tau = 1

name = ""

max_seq_length = 128 #200 #128
max_decoding_length = max_seq_length
emotion_length= 15
length_penalty=0.7

np = 1
train_batch_size = 2 #2#4 #2 * np #2 #8 #32
max_train_epoch = 4 #100
display_steps = 500 #20 # Print training loss every display_steps; -1 to disable
eval_steps = 1500  #300  # Eval on the dev set every eval_steps; -1 to disable
sample_steps = -1
checkpoint_steps = -1 #2000 # Checkpoint model parameters every checkpoint_steps;
                      # -1 to disable

eval_batch_size = 16 #4 #8 #4 #8
test_batch_size = 16 #6 #4 #8 #4 #8

## Optimization configs

opt = {
    'optimizer': {
        'type': 'AdamOptimizer',
        'kwargs': {
            'learning_rate': lr
        }
    }
}

## Data configs

feature_original_types = {
    # Reading features from TFRecord data file.
    # E.g., Reading feature "text_ids" as dtype `tf.int64`;
    # "FixedLenFeature" indicates its length is fixed for all data instances;
    # and the sequence length is limited by `max_seq_length`.
    "x1_ids": ["tf.int64", "FixedLenFeature", max_seq_length],
    "x1_len": ["tf.int64", "FixedLenFeature"],
    "x1x4_ids": ["tf.int64", "FixedLenFeature", max_seq_length],
    "x1x4_len": ["tf.int64", "FixedLenFeature"],
#    "x4_emo": ["tf.float32", "FixedLenFeature", emotion_length],
    "arc_label": ["tf.string", "FixedLenFeature"],
    "unique_id": ["tf.int64", "FixedLenFeature"]
}
feature_convert_types = {
    # Converting feature dtype after reading. E.g.,
    # Converting the dtype of feature "text_ids" from `tf.int64` (as above)
    # to `tf.int32`
    "x1_ids": "tf.int32",
    "x1_len": "tf.int32",
    "x1x4_ids": "tf.int32",
    "x1x4_len": "tf.int32",
#    "x4_emo": "tf.float32",
    "arc_label": "tf.string",
    "unique_id": "tf.int32"
}

train_hparam = {
    "allow_smaller_final_batch": False,
    "batch_size": train_batch_size,
    "dataset": {
        "data_name": "data",
        "feature_original_types": feature_original_types,
        "feature_convert_types": feature_convert_types,
        "files": "{}/train.tf_record".format(tfrecord_data_dir)
    },
    "shuffle": True,
    "shuffle_buffer_size": 1000
}

dev_hparam = {
    "allow_smaller_final_batch": True,
    "batch_size": eval_batch_size,
    "dataset": {
        "data_name": "data",
        "feature_original_types": feature_original_types,
        "feature_convert_types": feature_convert_types,
        "files": "{}/dev.tf_record".format(tfrecord_data_dir)
    },
    "shuffle": False
}

# Set to `test_hparam` to `None` if generating from scratch
# (instead of generating continuation) at test time
test_hparam = {
    "allow_smaller_final_batch": True,
    "batch_size": test_batch_size,
    "dataset": {
        "data_name": "data",
        "feature_original_types": feature_original_types,
        "feature_convert_types": feature_convert_types,
        "files": "{}/test.tf_record".format(tfrecord_data_dir)
    },
    "shuffle": False
}
