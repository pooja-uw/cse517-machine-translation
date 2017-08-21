import math
import re
from random import shuffle

import numpy as np
from nltk.translate.chrf_score import sentence_chrf
from nltk.translate.gleu_score import sentence_gleu
from nltk.tree import Tree
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from smart_open import smart_open as so

# Initialize data structures and configuration.
training_data = {}
dev_data = {}
testing_data = {}
do_shuffle = False
dev_percentage = 10

# Constants
labeled_data_path = "A5.train.labeled"
unlabeled_data_path = "A5.test.unlabeled"
output_data_path = "A5.test.labeled"
labeled_candidate_trees_path = "candidate.trees.labeled"
labeled_reference_trees_path = "reference.trees.labeled"
unlabeled_candidate_trees_path = "candidate.trees.unlabeled"
unlabeled_reference_trees_path = "reference.trees.unlabeled"

source_lines = "source_lines"
reference_lines = "reference_lines"
candidate_lines = "candidate_lines"
bleu_score_lines = "bleu_score_lines"
bleu_scores_lines = "bleu_scores_lines"
chrf_scores = "chrf_scores"
gleu_scores = "gleu_scores"
provided_labels = "provided_labels"
labels = "labels"
predicted_labels = "predicted_labels"
bleu_scores = "bleu_scores"
candidate_tree_heights = "candidate_tree_heights"
reference_tree_heights = "reference_tree_heights"

fields = [source_lines,
          reference_lines,
          candidate_lines,
          bleu_scores_lines,
          provided_labels,
          reference_tree_heights,
          candidate_tree_heights]

ratio_num_char_source_candidate = "ratio_num_char_source_candidate"
ratio_num_token_candidate_reference = "ratio_num_token_candidate_reference"
ratio_num_tokens_source_candidate = "ratio_num_tokens_source_candidate"
ratio_mean_token_length_source_candidate = "ratio_mean_token_length_source_candidate"
ratio_common_bigrams_candidate_reference = "ratio_common_bigrams_candidate_reference"
ratio_tree_height_candidate_reference = "ratio_tree_height_candidate_reference"
feature_vector = "feature_vector"

# Features used for Classifier
features = [gleu_scores,
            ratio_num_tokens_source_candidate,
            ratio_mean_token_length_source_candidate,
            ratio_common_bigrams_candidate_reference]


# Load the data from the input files
def load_data():
    line_ptr = 0
    for line in so(labeled_data_path):
        if line_ptr % 6 == 5:
            line_ptr += 1
            continue
        line_str = str(line, 'utf-8').strip()
        training_data[fields[line_ptr % 6]] = [] if fields[line_ptr % 6] not in training_data else training_data[
            fields[line_ptr % 6]]
        training_data[fields[line_ptr % 6]].append(line_str)
        line_ptr += 1

    training_data[candidate_tree_heights] = []
    for line in so(labeled_candidate_trees_path):
        line_str = str(line, 'utf-8').strip()
        training_data[candidate_tree_heights].append(Tree.fromstring(line_str).height())
    training_data[reference_tree_heights] = []
    for line in so(labeled_reference_trees_path):
        line_str = str(line, 'utf-8').strip()
        training_data[reference_tree_heights].append(Tree.fromstring(line_str).height())

    if do_shuffle:
        shuffled_idx = list(range(len(training_data[candidate_lines])))
        shuffle(shuffled_idx)
        shuffled_data = {}
        for line_idx in shuffled_idx:
            for field in fields:
                shuffled_data[field] = [] if field not in shuffled_data else shuffled_data[field]
                shuffled_data[field].append(training_data[field][line_idx])
        for field in fields:
            training_data[field] = shuffled_data[field]

    dev_line_ptr = math.ceil(len(training_data[source_lines]) * ((100 - dev_percentage) / 100))

    for line_idx in range(dev_line_ptr, len(training_data[source_lines])):
        for field in fields:
            dev_data[field] = [] if field not in dev_data else dev_data[field]
            dev_data[field].append(training_data[field][line_idx])

    for field in fields:
        training_data[field] = training_data[field][0:dev_line_ptr]

    line_ptr = 0
    for line in so(unlabeled_data_path):
        if line_ptr % 6 == 5:
            line_ptr += 1
            continue
        line_str = str(line, 'utf-8').strip()
        testing_data[fields[line_ptr % 6]] = [] if fields[line_ptr % 6] not in testing_data else testing_data[
            fields[line_ptr % 6]]
        testing_data[fields[line_ptr % 6]].append(line_str)
        line_ptr += 1

    testing_data[candidate_tree_heights] = []
    for line in so(unlabeled_candidate_trees_path):
        line_str = str(line, 'utf-8').strip()
        testing_data[candidate_tree_heights].append(Tree.fromstring(line_str).height())
    testing_data[reference_tree_heights] = []
    for line in so(unlabeled_reference_trees_path):
        line_str = str(line, 'utf-8').strip()
        testing_data[reference_tree_heights].append(Tree.fromstring(line_str).height())


# Add features to the in-memory data structures
def compute_features(data):
    # Initialize all feature placeholders
    data[ratio_num_char_source_candidate] = []
    data[ratio_num_tokens_source_candidate] = []
    data[ratio_mean_token_length_source_candidate] = []
    data[ratio_common_bigrams_candidate_reference] = []
    data[ratio_num_token_candidate_reference] = []
    data[gleu_scores] = []
    data[bleu_scores] = []
    data[chrf_scores] = []
    data[labels] = []
    data[ratio_tree_height_candidate_reference] = []

    for line_idx in range(0, len(data[source_lines])):
        # Feature: gleu_scores
        data[gleu_scores].append(sentence_gleu(data[reference_lines][line_idx], data[candidate_lines][line_idx]))

        # Feature: chrf_scores
        data[chrf_scores].append(sentence_chrf(data[reference_lines][line_idx], data[candidate_lines][line_idx]))

        # Feature: bleu_scores
        data[bleu_scores].append(float(data[bleu_scores_lines][line_idx]))

        # Feature: ratio_num_char_source_candidate
        data[ratio_num_char_source_candidate].append(
            len(re.sub('[\s+]', '', data[source_lines][line_idx]))
            / len(re.sub('[\s+]', '', data[candidate_lines][line_idx])))

        # Feature: ratio_num_tokens_source_candidate
        data[ratio_num_tokens_source_candidate].append(
            len(re.compile('\S+').findall(data[source_lines][line_idx]))
            / len(re.compile('\S+').findall(data[candidate_lines][line_idx])))

        # Feature: ratio_num_token_candidate_reference
        data[ratio_num_token_candidate_reference].append(
            len(re.sub('[\S+]', '', data[candidate_lines][line_idx]))
            / len(re.sub('[\S+]', '', data[reference_lines][line_idx])))

        # Feature: ratio_mean_token_length_source_candidate
        data[ratio_mean_token_length_source_candidate].append(
            np.mean(list(map(len, re.compile('\S+').findall(data[source_lines][line_idx]))))
            / np.mean(list(map(len, re.compile('\S+').findall(data[candidate_lines][line_idx])))))

        # Feature: ratio_common_bigrams_candidate_reference
        data[ratio_common_bigrams_candidate_reference].append(
            len(
                set([b for b in zip(re.compile('\S+').findall(data[reference_lines][line_idx])[:-1],
                                    re.compile('\S+').findall(data[reference_lines][line_idx])[1:])])
                &
                set([b for b in zip(re.compile('\S+').findall(data[candidate_lines][line_idx])[:-1],
                                    re.compile('\S+').findall(data[candidate_lines][line_idx])[1:])])
            )
            /
            len([b for b in zip(re.compile('\S+').findall(data[reference_lines][line_idx])[:-1],
                                re.compile('\S+').findall(data[reference_lines][line_idx])[1:])]))

        # Feature: ratio_tree_height_candidate_reference
        data[ratio_tree_height_candidate_reference].append(
            data[candidate_tree_heights][line_idx] / data[reference_tree_heights][line_idx]
        )

        # Feature: labels
        data[labels].append(1 if data[provided_labels][line_idx] == "H" else 0)


# Computes the feature vector
def compute_feature_vector(data, feature_set):
    data[feature_vector] = []
    for line_idx in range(0, len(data[source_lines])):
        # Feature: feature_vector
        data[feature_vector].append([])
        for feature in feature_set:
            data[feature_vector][line_idx].append(data[feature][line_idx])
        data[feature_vector][line_idx] = np.array(data[feature_vector][line_idx])


# Load all the data
load_data()

# Compute features and feature vector for training data
compute_features(training_data)
compute_feature_vector(training_data, features)

# Compute features and feature vector for dev data
compute_features(dev_data)
compute_feature_vector(dev_data, features)

# Compute features and feature vector for testing data
compute_features(testing_data)
compute_feature_vector(testing_data, features)

# Initialize and train classifier
svc_model = SVC()
svc_model.fit(np.array(training_data[feature_vector]), np.array(training_data[labels]))

# Predict for dev data
svc_predict_dev = svc_model.predict(np.array(dev_data[feature_vector]))
print("Statistics for Dev Testing\n==========================")
print("Accuracy: {0:.4f}".format(accuracy_score(np.array(dev_data[labels]), svc_predict_dev)))
print("F1 Score: {0:.4f}".format(f1_score(np.array(dev_data[labels]), svc_predict_dev)))
print("Confusion Matrix")
print(confusion_matrix(np.array(dev_data[labels]), svc_predict_dev))

# Predict for unlabeled data
svc_predict_test = svc_model.predict(np.array(testing_data[feature_vector]))
testing_data[predicted_labels] = list(map(lambda label: "M" if label == 0 else "H", svc_predict_test))

# Output for unlabeled data
output = open(output_data_path, "w")
for line_idx in range(len(testing_data[source_lines])):
    for field in [source_lines, reference_lines, candidate_lines, bleu_scores_lines, predicted_labels]:
        output.write(testing_data[field][line_idx] + "\n")
    output.write("\n")
output.close()
print("\nSaved labels for unlabelled data to: " + output_data_path)
