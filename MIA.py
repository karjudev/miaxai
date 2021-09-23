import pandas as pd
import pickle
import math

from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim.adam import Adam
from classifier_wrapper import pytorch_classifier_wrapper, sklearn_classifier_wrapper
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import multiprocessing as ml
from imblearn.under_sampling import RandomUnderSampler
from torch import nn, softmax, tanh
import dask.array as da
from dask_ml.wrappers import ParallelPostFit


"""Class MIA (Membership Inference Attack)
3 main objects:
1 black-box model (its train and test data)
n shadow models which mimic the black-box (each with train and test dataset, with overlaps)
|labels| attack model 
(train and test different wrt the other models: here predict proba and in and out labels)"""


class MIA(object):
    def __init__(
        self,
        path,
        target_model=None,
        black_box=None,
        train_s_shadows=None,
        train_l_shadows=None,
        test_s_shadows=None,
        test_l_shadows=None,
        shadows=None,
        n_shadows=1,
    ):
        super().__init__()
        self.black_box = black_box
        self.train_set_shadows = list() if train_s_shadows is None else train_s_shadows
        self.train_label_shadows = (
            list() if train_l_shadows is None else train_l_shadows
        )
        self.test_set_shadows = list() if test_s_shadows is None else test_s_shadows
        self.test_label_shadows = list() if test_l_shadows is None else test_l_shadows
        self.shadows = list() if shadows is None else shadows
        self.path = path
        self.n_shadows = n_shadows
        self.target_model = target_model
        self.processes = list()

    """train della black box
    input: train set, train set label, type of black box
    """

    def train_black_box(self, trs, trl, type):
        self.type = type
        name = "black_box"
        if self.type == "RF":
            self.black_box = self.rfc(trs, trl, name)

    """test della black box
    input test set, test set label
    output a report with prediction performance"""

    def test_black_box(self, ts, tsl):
        with open("report_black_box.txt", "w") as reports:
            ts_pred_norm = self.black_box.predict(ts)
            reports.writelines(classification_report(tsl, ts_pred_norm))
            reports.close()

    """creation of the dataset for the attack model, from the dataset of the black box"""

    def create_attack_dataset_from_bb(self, trs, trl, ts, tsl, labels):
        trs_pred = self.black_box.predict_proba(trs)
        ts_pred = self.black_box.predict_proba(ts)
        inout = list()
        for j in range(0, len(trl.values)):
            temp = list()
            temp.append(trl.iloc[j])
            for el in trs_pred[j]:
                temp.append(el)
            temp.append("in")
            inout.append(temp)
        for j in range(0, len(tsl.values)):
            temp = list()
            temp.append(tsl.iloc[j])
            for el in ts_pred[j]:
                temp.append(el)
            temp.append("out")
            inout.append(temp)
        del trs
        del ts
        del trl
        del tsl
        col = list()
        col.append("true")
        for l in labels:
            col.append(str(l))
        col.append("inout")
        inout_df = pd.DataFrame(inout, columns=col)
        pickle.dump(inout_df, open(self.path + "inout_black_box", "wb"))
        print(inout_df)
        del inout
        del inout_df

    def create_shadow_dataset(self, path, n_chunks, n_samples):
        X = da.concatenate(
            [da.from_zarr(f"{path}/chunk_{i}.zarr") for i in range(n_chunks)]
        )
        parallel_model = ParallelPostFit(self.black_box, scoring="f1")
        y_pred = parallel_model.predict(X).rechunk({0: "auto"})
        y_pred
        y_prob = parallel_model.predict_proba(X).rechunk({0: "auto", 1: "auto"})
        X_sample, _, y_sample, _, y_prob_sample, _ = train_test_split(
            X, y_pred, y_prob, train_size=n_samples
        )

    """Master procedure for the train of the shadow models
    input: dataset (the one only for the shadow models, the method splits them into overlapping)
    labels of the dataset
    model type of the shadow models
    test size to split the dataset
    workers number of threads to parallelize"""

    def train_shadows(self, dataset, labels, type, test_size, workers):

        self.type = type

        # for parallelization
        items_for_worker = math.ceil(self.n_shadows / float(workers))
        start = 0
        print(items_for_worker)
        end = int(items_for_worker)

        # create workers
        print("Dispatching jobs to workers...\n")
        for i in range(0, workers):
            process = ml.Process(
                target=self.train_shadows_workers,
                args=(i, start, end, dataset, labels, test_size,),
            )
            self.processes.append(process)
            process.start()
            start = end
            end += int(items_for_worker)
            if end > self.n_shadows:
                workers = workers - 1
                break

        # join workers
        for i in range(0, workers):
            self.processes[i].join()
        print("All workers joint.\n")

    """Procedure to train the shadow model
    Procedure for the worker
    id of the worker
    start index of the worker
    end index of the worker 
    dataset to split
    labels of the dataset
    test size to split the dataset"""

    def train_shadows_workers(self, id, start, end, dataset, labels, test_size):
        print("Start process id ", id, start, end)
        # creare train, test per ogni shadow
        for i in range(start, end):
            # split the orginal dataset, the split is stratified
            # it is fine to have overlaps here, random state not fixed to select different splits
            trs, ts, trl, tsl = train_test_split(
                dataset, labels, test_size=test_size, stratify=labels
            )
            # dump the datasets
            pickle.dump(trs, open(self.path + "trs_" + str(i), "wb"))
            pickle.dump(ts, open(self.path + "ts_" + str(i), "wb"))
            pickle.dump(trl, open(self.path + "trl_" + str(i), "wb"))
            pickle.dump(tsl, open(self.path + "tsl_" + str(i), "wb"))
            del trs
            del ts
            del trl
            del tsl
        for i in range(start, end):
            trs = pickle.load(open(self.path + "trs_" + str(i), "rb"))
            trl = pickle.load(open(self.path + "trl_" + str(i), "rb"))
            if self.type == "RF":
                name = str(i) + "_shadow"
                shadow = self.rfc(trs, trl, name)
            else:
                print("unknown model for the shadow models %s" % type)
                raise Exception
            self.shadows.append(shadow)
            del trs
            del trl
            del shadow
        print("End process id ", id)

    """Random forest classifier implementation
    input: train set, train set label, index+name"""

    def rfc(self, trs, trl, i):
        # qui i tuoi parametri
        mod = RandomForestClassifier(n_estimators=100)
        mod.fit(trs, trl)
        mod = sklearn_classifier_wrapper(mod)
        pickle.dump(mod, open(self.path + str(self.type) + str(i), "wb"))
        return mod

    """Creation of the attack dataset from the shadow model
    One file for each shadow model (done for space and time constraints)
    One file for each label (the attack has a model for each label)
    type: of the shadow model
    labels to apply predict proba
    report Boolean if you want prediction performance report or not"""

    def shadow_pred_partial_attack_dataset(self, type, labels, report):
        # cycle on all the shadow models and their train/test
        # predict the prob vector for train and test
        # label train as: true label, predicted label, in
        # label test as: true label, predicted label, out
        # store the partial dataset to use during the training of the attack
        pickle_file = self.path
        for i in range(0, self.n_shadows):
            shadow = pickle.load(open(self.path + str(type) + str(i) + "_shadow", "rb"))
            trs = pickle.load(open(self.path + "trs_" + str(i), "rb"))
            ts = pickle.load(open(self.path + "ts_" + str(i), "rb"))
            trl = pickle.load(open(self.path + "trl_" + str(i), "rb"))
            tsl = pickle.load(open(self.path + "tsl_" + str(i), "rb"))
            trs_pred = shadow.predict_proba(trs)
            ts_pred = shadow.predict_proba(ts)
            # ts_pred = shadow.predict_proba(ts)
            if report is True:
                reports = open("report_" + str(i) + ".txt", "w")
                ts_pred_norm = shadow.predict(ts)
                reports.writelines(classification_report(tsl.values, ts_pred_norm))
                reports.close()
            inout = list()
            for j in range(0, len(trl.values)):
                temp = list()
                temp.append(trl.iloc[j])
                for el in trs_pred[j]:
                    temp.append(el)
                temp.append("in")
                inout.append(temp)
            for j in range(0, len(tsl.values)):
                temp = list()
                temp.append(tsl.iloc[j])
                for el in ts_pred[j]:
                    temp.append(el)
                temp.append("out")
                inout.append(temp)
            del trs
            del ts
            del trl
            del tsl
            col = list()
            col.append("true")
            for l in labels:
                col.append(str(l))
            col.append("inout")
            inout_df = pd.DataFrame(inout, columns=col)
            pickle.dump(inout_df, open(self.path + "inout_" + str(i), "wb"))
            print(inout_df)
            del inout
            del inout_df

    """Merge of the attack dataset from the shadow model and creation of train test data
    One file for each label"""

    def merge_partial_attack_dataset(self, test_size):
        temp = pickle.load(open(self.path + "inout_" + str(0), "rb"))
        attack_dataset = temp
        for i in range(1, self.n_shadows):
            temp = pickle.load(open(self.path + "inout_" + str(i), "rb"))
            attack_dataset = pd.concat([attack_dataset, temp], ignore_index=True)
        for l in attack_dataset["true"].unique():
            temp = attack_dataset[attack_dataset["true"] == l]
            label = temp.pop("inout")
            train_attack, test_attack, train_l_attack, test_l_attack = train_test_split(
                temp, label, test_size=test_size, stratify=label
            )
            pickle.dump(
                train_attack, open(self.path + "train_attack_label_" + str(l), "wb")
            )
            pickle.dump(
                train_l_attack, open(self.path + "train_l_attack_label_" + str(l), "wb")
            )
            pickle.dump(
                test_attack, open(self.path + "test_attack_label_" + str(l), "wb")
            )
            pickle.dump(
                test_l_attack, open(self.path + "test_l_attack_label_" + str(l), "wb")
            )
            del train_attack
            del train_l_attack
            del test_attack
            del test_l_attack
        del attack_dataset

    """Train of the attack model
    One attack model for each label
    labels to load the train set
    type of the attack model"""

    def train_attack_model(self, labels, type, under_val):
        # train a model for each output class
        self.type = type
        for l in labels:
            print(self.path + "train_attack_label_" + str(l))
            # train an attack model
            # rf con i tuoi parametri
            # nn con i tuoi parametri
            train_attack = pickle.load(
                open(self.path + "train_attack_label_" + str(l), "rb")
            )
            train_l_attack = pickle.load(
                open(self.path + "train_l_attack_label_" + str(l), "rb")
            )
            if under_val != -1:
                sampler = RandomUnderSampler(
                    sampling_strategy=under_val, random_state=42
                )
                train_attack, train_l_attack = sampler.fit_sample(
                    train_attack, train_l_attack
                )
            if type == "RF":
                i = "attack_" + str(l)
                self.rfc(train_attack, train_l_attack, i)
            del train_attack
            del train_l_attack

    """Test of the attack model, considered as an ensemble method
    dataset: the dataset to test. it can be the dataset from the shadows or from the black-box"""

    def test_attack(self, dataset, labels, type):
        # load of all the models
        attacks = list()
        for l in labels:
            print(self.path + "attack_" + str(l))
            attacks.append(pickle.load(open(self.path + "attack_" + str(l), "rb")))
        if type == "all":
            predictions = list()
            for a in attacks:
                predictions.append(a.predict(dataset))
            pred_row_wise = zip(*predictions)
            max_predictions = list()
            for p in pred_row_wise:
                try:
                    max_predictions.append(max(p, key=p.count))
                except:
                    print("max does not return a single value")
                    raise Exception
            report = open("report_attack_test_all.txt", "w")
            report.writelines(classification_report(dataset.values, max_predictions))
            report.close()

    """test set attack are the data to test
    test_l_attack are the labels of the test set
    labels the unique labels
    type of the attack model"""

    def test_attack_model(self, test_set_attack, test_l_attack, labels, type):
        # load of all the models
        attacks = list()
        for l in labels:
            print(self.path + self.type + "attack_" + str(l))
            attacks.append(
                pickle.load(open(self.path + self.type + "attack_" + str(l), "rb"))
            )
        if type == "all":
            predictions = list()
            for a in attacks:
                predictions.append(a.predict(test_set_attack))
            pred_row_wise = zip(*predictions)
            max_predictions = list()
            for p in pred_row_wise:
                try:
                    max_predictions.append(max(p, key=p.count))
                except:
                    print("max does not return a single value")
                    raise Exception
            report = open("report_attack_test_all.txt", "w")
            report.writelines(
                classification_report(test_l_attack.values, max_predictions)
            )
            report.close()
        # non funziona
        elif type == "single":
            attacks = list()
            for l in labels:
                attacks.append(pickle.load(open(self.path + "attack_" + str(l), "rb")))
            tests_attack = list()
            tests_l_attack = list()
            for l in range(0, len(labels)):
                tests_attack.append(
                    pickle.load(open(self.path + "test_attack_label_" + str(l), "rb"))
                )
                tests_l_attack.append(
                    pickle.load(open(self.path + "test_l_attack_label_" + str(l), "rb"))
                )
            # in questo caso la lista risulta essere una per label, se binario ho solo due liste
            predictions = list()
            for i in range(0, len(tests_attack)):
                predictions.append(attacks[i].predict(tests_attack[i]))
                # report di predizioni singole
                report = open("report_attack_test_single_" + str(i) + ".txt", "w")
                report.writelines(
                    classification_report(tests_l_attack[i].values, predictions[i])
                )
                report.close()
            tests_l = tests_l_attack[0]
            for i in range(1, len(tests_l_attack)):
                tests_l = pd.concat([tests_l, tests_l_attack[i]], ignore_index=True)
            preds = list()
            for p in predictions:
                for t in p:
                    preds.append(t)
            print(len(tests_l.values))
            print(len(preds))
            report = open("report_attack_test_single_complete.txt", "w")
            report.writelines(classification_report(tests_l.values, preds))
            report.close()

    def test_auditing(self, train_set_bb, train_label_bb, attacks, nome):
        predictions = list()
        for a in attacks:
            predictions.append(a.predict(train_set_bb))
        pred_row_wise = zip(*predictions)
        max_predictions = list()
        for p in pred_row_wise:
            try:
                max_predictions.append(max(p, key=p.count))
            except:
                print("max does not return a single value")
                raise Exception
        report = open("report_attack_test_auditing_" + nome + ".txt", "w")
        report.writelines(classification_report(train_label_bb.values, max_predictions))
        report.close()

