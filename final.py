import os
import glob
import json
import pprint
import glob
import numpy as np
from tqdm import tqdm
from lightgbm import LGBMClassifier

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE


SEED = 41


def read_label_csv(path):  # 정답 파일에 대한 경로를 path로 주면 된다.
    label_table = dict()
    with open(path, "r") as f:
        for line in f.readlines()[1:]:
            fname, label = line.strip().split(",")
            label_table[fname] = int(label)
    return label_table


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_model(**kwargs):
    if kwargs["model"] == "rf":
        return RandomForestClassifier(random_state=kwargs["random_state"], n_jobs=4)
    elif kwargs["model"] == "dt":
        return DecisionTreeClassifier(random_state=kwargs["random_state"])
    elif kwargs["model"] == "lgb":
        return LGBMClassifier(random_state=kwargs["random_state"])
    elif kwargs["model"] == "svm":
        return SVC(random_state=kwargs["random_state"])
    elif kwargs["model"] == "lr":
        return LogisticRegression(random_state=kwargs["random_state"], n_jobs=-1)
    elif kwargs["model"] == "knn":
        return KNeighborsClassifier(n_jobs=-1)
    elif kwargs["model"] == "adaboost":
        return AdaBoostClassifier(random_state=kwargs["random_state"])
    elif kwargs["model"] == "mlp":
        return MLPClassifier(random_state=kwargs["random_state"])
    else:
        print("Unsupported Algorithm")
        return None


def train(X_train, y_train, model):
    '''
        머신러닝 모델을 선택하여 학습을 진행하는 함수

        :param X_train: 학습할 2차원 리스트 특징벡터
        :param y_train: 학습할 1차원 리스트 레이블 벡터
        :param model: 문자열, 선택할 머신러닝 알고리즘
        :return: 학습된 머신러닝 모델 객체
    '''
    clf = load_model(model=model, random_state=SEED)
    clf.fit(X_train, y_train)
    return clf


def evaluate(X_test, y_test, model):
    '''
        학습된 머신러닝 모델로 검증 데이터를 검증하는 함수

        :param X_test: 검증할 2차원 리스트 특징 벡터
        :param y_test: 검증할 1차원 리스트 레이블 벡터
        :param model: 학습된 머신러닝 모델 객체
    '''
    predict = model.predict(X_test)
    print("정확도", model.score(X_test, y_test))


class PeminerParser:
    def __init__(self, path):
        self.report = read_json(path)
        self.vector = []

    def process_report(self):
        self.vector = [value for _, value in sorted(self.report.items(), key=lambda x: x[0])]
        return self.vector


class EmberParser:
    def __init__(self, path):
        self.report = read_json(path)
        self.vector = []

    def get_histogram_info(self):
        histogram = np.array(self.report["histogram"])
        total = histogram.sum()
        vector = histogram / total
        return vector.tolist()

    def get_byte_entropy_info(self):
        histogram = np.array(self.report["byteentropy"])
        total = histogram.sum()
        vector = histogram / total
        return vector.tolist()

    def get_string_info(self):
        strings = self.report["strings"]

        hist_divisor = float(strings['printables']) if strings['printables'] > 0 else 1.0
        vector = [
            strings['numstrings'],
            strings['avlength'],
            strings['printables'],
            strings['entropy'],
            strings['paths'],
            strings['urls'],
            strings['registry'],
            strings['MZ']
        ]

        vector += (np.asarray(strings['printabledist']) / hist_divisor).tolist()
        return vector

    def get_general_file_info(self):
        general = self.report["general"]
        vector = [
            general['size'], general['vsize'], general['has_debug'], general['exports'], general['imports'],
            general['has_relocations'], general['has_resources'], general['has_signature'], general['has_tls'],
            general['symbols']
        ]
        return vector

    def get_data_directory_info(self):
        data_directory = self.report['datadirectories']
        if len(data_directory) != 15:
            vector = [0] * 15
        else:
            vector = []
            for data in data_directory:
                vector.append(data['size'])
        return vector

    def process_report(self):
        vector = []
        vector += self.get_general_file_info()
        vector += self.get_histogram_info()
        vector += self.get_string_info()
        vector += self.get_byte_entropy_info()
        vector += self.get_data_directory_info()
        return vector


class PestudioParser:
    def __init__(self, path):
        self.report = read_json(path)['image']
        self.vector = []

    def hex_to_int(self, hex):
        if hex is None:
            return 0
        return int(hex, 16)

    def tf_to_int(self, tf):
        return int(tf == 'true')

    def check_x(self, string):
        if string == 'x':
            return 1
        else:
            return 0

    def get_overview_info(self):
        overview = self.report.get('overview', None)
        if overview is None:
            return [0.0]

        try:
            vector = [float(overview['entropy'])]
        except:
            vector = [0.0]
        return vector

    def get_file_header_info(self):
        file_header = self.report.get('file-header', None)
        if file_header is None:
            vector = [0] * 13
        else:
            vector = [
                self.tf_to_int(file_header['relocation-stripped']),
                self.tf_to_int(file_header['executable']),
                self.tf_to_int(file_header['large-address-aware']),
                self.tf_to_int(file_header['processor-32bit']),
                self.tf_to_int(file_header['uniprocessor']),
                self.tf_to_int(file_header['system-image']),
                self.tf_to_int(file_header['dynamic-link-library']),
                self.tf_to_int(file_header['debug-stripped']),
                self.tf_to_int(file_header['media-run-from-swap']),
                self.tf_to_int(file_header['network-run-from-swap']),
                int(file_header['sections']),
                int(file_header['number-of-symbols']),
                int(file_header['size-of-optional-header'])
            ]

        return vector

    def get_optional_header_info(self):
        optional_header = self.report.get('optional-header', None)
        if optional_header is None:
            vector = [0] * 15
        else:
            vector = [
                int(optional_header['size-of-code']),
                int(optional_header['size-of-initialized-data']),
                int(optional_header['size-of-uninitialized-data']),
                self.hex_to_int(optional_header['entry-point']),
                self.hex_to_int(optional_header['base-of-code']),
                self.hex_to_int(optional_header['base-of-data']),
                self.hex_to_int(optional_header['image-base']),
                self.hex_to_int(optional_header['section-alignment']),
                self.hex_to_int(optional_header['file-alignment']),
                int(optional_header['size-of-image']),
                int(optional_header['size-of-headers']),
                self.hex_to_int(optional_header['file-checksum']),
                self.tf_to_int(optional_header['code-integrity']),
                self.tf_to_int(optional_header['structured-exception-handling']),
                self.tf_to_int(optional_header['data-execution-prevention'])
            ]
        return vector

    def get_sections_info(self):
        sections = self.report.get('sections', None)
        if sections is None:
            return [0] * 13

        sections = self.report['sections']['section']
        if type(sections) == list:
            vector = [len(sections)]
            vector.append(max([int(data['@virtual-size']) for data in sections]))
            vector.append(max([int(data['@raw-size']) for data in sections]))
            vector.append(len([self.check_x(data['@self-modifying']) for data in sections]))
            vector.append(len([self.check_x(data['@blacklisted']) for data in sections]))
            vector.append(len([self.check_x(data['@initialized-data']) for data in sections]))
            vector.append(len([self.check_x(data['@uninitialized-data']) for data in sections]))
            vector.append(len([self.check_x(data['@discardable']) for data in sections]))
            vector.append(len([self.check_x(data['@shareable']) for data in sections]))
            vector.append(len([self.check_x(data['@executable']) for data in sections]))
            vector.append(len([self.check_x(data['@readable']) for data in sections]))
            vector.append(len([self.check_x(data['@writable']) for data in sections]))

            try:
                entropy = max([float(data['@entropy']) for data in sections])
            except:
                entropy = 0
            vector.append(entropy)

        else:
            vector = [1]
            vector.append(int(sections['@virtual-size']))
            vector.append(int(sections['@raw-size']))
            vector.append(self.check_x(sections['@self-modifying']))
            vector.append(self.check_x(sections['@blacklisted']))
            vector.append(self.check_x(sections['@initialized-data']))
            vector.append(self.check_x(sections['@uninitialized-data']))
            vector.append(self.check_x(sections['@discardable']))
            vector.append(self.check_x(sections['@shareable']))
            vector.append(self.check_x(sections['@executable']))
            vector.append(self.check_x(sections['@readable']))
            vector.append(self.check_x(sections['@writable']))

            try:
                entropy = float(sections['@entropy'])
            except:
                entropy = 0
            vector.append(entropy)

        return vector

    def get_resources_info(self):
        if self.report.get('resources', None) is None:
            return [0, 0, 0.0, 0.0]
        # resources 필드가 n/a 일 경우
        if self.report['resources'] == 'n/a':
            return [0, 0, 0.0, 0.0]
        else:
            # resources 필드는 n/a가 아니지만 instance 필드가 없는 경우
            if self.report['resources'].get('instance', None) is None:
                return [0, 0, 0.0, 0.0]
            else:
                resources = self.report['resources']['instance']
                vector = []
                if type(resources) == list:
                    vector.append(len(resources))
                    vector.append(max([int(data['@size']) for data in resources if data != None], default=0))
                    vector.append(max([float(data['@entropy']) for data in resources if data != None and data['@entropy'] != '-'], default=0.0))
                    vector.append(max([float(data['@file-ratio'][:-1]) for data in resources if data != None and data['@file-ratio'] != 'n/a'], default=0.0))
                else:
                    vector.append(1)
                    vector.append(int(resources['@size']))
                    vector.append(float(resources['@entropy'] if resources['@entropy'] != '-' else 0.0))
                    vector.append(float(resources['@file-ratio'][:-1] if resources['@file-ratio'] != 'n/a' else 0.0))

                return vector

    def get_string_info(self):
        strings = self.report['strings']
        vector = [
            int(strings['@count']),
            int(strings['@bl']),
            int(strings['ascii']['@count']),
        ]

        # 'unicode'에 'null'이 나올 수 있음.
        try:
            uni_count = int(strings['unicode']['@count'])
        except:
            vector.append(0)
        else:
            vector.append(uni_count)
        return vector

    def etc_info(self):
        manifest = self.report.get('manifest', None)
        debug = self.report.get('debug', None)
        version = self.report.get('version', None)
        tls_callback = self.report.get('tls-callbacks', None)

        mani_vector = [1 if manifest != 'n/a' and manifest != None else 0]
        debug_vector = [1 if debug != 'n/a' and debug != None else 0]
        version_vector = [1 if version != 'n/a' and version != None else 0]
        tls_vector = [1 if tls_callback != 'n/a' and tls_callback != None else 0]

        vector = []
        vector += mani_vector
        vector += debug_vector
        vector += version_vector
        vector += tls_vector

        return vector

    def get_certificate_info(self):
        if self.report.get('certificate', None) is None:
            return [0]

        if self.report['certificate'] == 'n/a':
            return [0]
        else:
            return [1]

    def get_overlay_info(self):
        if self.report.get('overlay', None) is None:
            return [0, 0, 0, 0, 0]

        overlay = self.report['overlay']
        if overlay == 'n/a':
            return [0, 0, 0, 0, 0]
        else:
            vector = [1]
            vector.append(self.hex_to_int(overlay['file-offset']))
            vector.append(int(overlay['size']))
            vector.append(float(overlay['entropy']))
            vector.append(float(overlay['file-ratio'][:-1]))

            return vector

    def process_report(self):
        vector = []
        vector += self.get_overview_info()
        vector += self.get_file_header_info()
        vector += self.get_optional_header_info()
        vector += self.get_sections_info()
        vector += self.get_resources_info()
        vector += self.get_string_info()
        vector += self.get_certificate_info()
        vector += self.get_overlay_info()
        vector += self.etc_info()

        return vector


def ensemble_result(peminer_X, ember_X, pestudio_X, peminer_y, ember_y, pestudio_y, models):
    '''
        학습된 모델들의 결과를 앙상블하는 함수

        :param X: 검증할 2차원 리스트 특징 벡터
        :param y: 검증할 1차원 리스트 레이블 벡터
        :param models: 1개 이상의 학습된 머신러닝 모델 객체를 가지는 1차원 리스트
    '''

    # Soft Voting
    # https://devkor.tistory.com/entry/Soft-Voting-%EA%B3%BC-Hard-Voting
    predicts = []

    prob = [pr for pr in models[0].predict_proba(peminer_X)]
    predicts.append(prob)

    prob = [pr for pr in models[1].predict_proba(ember_X)]
    predicts.append(prob)

    prob = [pr for pr in models[2].predict_proba(pestudio_X)]
    predicts.append(prob)

    predict = np.mean(predicts, axis=0)
    predict = [1 if x >= 0.5 else 0 for x in predict]

    print("정확도", accuracy_score(peminer_y, predict))

def select_feature(X, y, model):
    '''
        주어진 특징 벡터에서 특정 알고리즘 기반 특징 선택

        본 예제에서는 RFE 알고리즘 사용
        https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html#sklearn.feature_selection.RFE.fit_transform

        :param X: 검증할 2차원 리스트 특징 벡터
        :param y: 검증할 1차원 리스트 레이블 벡터
        :param model: 문자열, 특징 선택에 사용할 머신러닝 알고리즘
    '''

    model = load_model(model=model, random_state=SEED)
    rfe = RFE(estimator=model)
    return rfe.fit_transform(X, y)


label_table = read_label_csv("./data/학습데이터_정답.csv")
test_table = read_label_csv("./data/검증데이터_정답.csv")

def peminer_dataset(manual_test=True):
    peminer = "./data/PEMINER/학습데이터/*"
    peminer_valid = "./data/PEMINER/검증데이터/*"
    peminer_test = "./data/PEMINER/테스트데이터/*"

    peminer_file_list = glob.glob(peminer)
    peminer_valid_file_list = glob.glob(peminer_valid)
    peminer_fname_list = [file.split("\\")[1].split(".")[0] for file in peminer_file_list]
    peminer_valid_fname_list = [file.split("\\")[1].split(".")[0] for file in peminer_valid_file_list]

    peminer_X_train, peminer_y_train = [], []
    peminer_X_valid, peminer_y_valid = [], []

    for fname in tqdm(peminer_fname_list):
        feature_vector = []
        label = label_table[fname]
        path = f"./data/PEMINER/학습데이터/{fname}.json"
        feature_vector += PeminerParser(path).process_report()
        peminer_X_train.append(feature_vector)
        peminer_y_train.append(label)

    for fname in tqdm(peminer_valid_fname_list):
        feature_vector = []
        label = test_table[fname]
        path = f"./data/PEMINER/검증데이터/{fname}.json"
        feature_vector += PeminerParser(path).process_report()
        peminer_X_valid.append(feature_vector)
        peminer_y_valid.append(label)

    if manual_test:
        models = []
        for model in ["rf"]:
            clf = train(peminer_X_train, peminer_y_train, model)
            models.append(clf)

        for model in models:
            evaluate(peminer_X_valid, peminer_y_valid, model)

    return [peminer_X_train, peminer_y_train, peminer_X_valid, peminer_y_valid]

def ember_dataset(manual_test=True):
    ember = "./data/EMBER/학습데이터/*"
    ember_valid = "./data/EMBER/검증데이터/*"
    ember_test = "./data/EMBER/테스트데이터/*"

    ember_file_list = glob.glob(ember)
    ember_valid_file_list = glob.glob(ember_valid)
    ember_fname_list = [file.split("\\")[1].split(".")[0] for file in ember_file_list]
    ember_valid_fname_list = [file.split("\\")[1].split(".")[0] for file in ember_valid_file_list]

    ember_X_train, ember_y_train = [], []
    ember_X_valid, ember_y_valid = [], []

    for fname in tqdm(ember_fname_list):
        feature_vector = []
        label = label_table[fname]
        path = f"./data/EMBER/학습데이터/{fname}.json"
        feature_vector += EmberParser(path).process_report()
        ember_X_train.append(feature_vector)
        ember_y_train.append(label)

    for fname in tqdm(ember_valid_fname_list):
        feature_vector = []
        label = test_table[fname]
        path = f"./data/EMBER/검증데이터/{fname}.json"
        feature_vector += EmberParser(path).process_report()
        ember_X_valid.append(feature_vector)
        ember_y_valid.append(label)

    if manual_test:
        models = []
        for model in ["rf"]:
            clf = train(ember_X_train, ember_y_train, model)
            models.append(clf)

        for model in models:
            evaluate(ember_X_valid, ember_y_valid, model)

    return [ember_X_train, ember_y_train, ember_X_valid, ember_y_valid]

def pestudio_dataset(manual_test=True):
    pestudio = "./data/PESTUDIO/학습데이터/*"
    pestudio_valid = "./data/PESTUDIO/검증데이터/*"
    pestudio_test = "./data/PESTUDIO/테스트데이터/*"

    pestudio_file_list = glob.glob(pestudio)
    pestudio_valid_file_list = glob.glob(pestudio_valid)
    pestudio_fname_list = [file.split("\\")[1].split(".")[0] for file in pestudio_file_list]
    pestudio_valid_fname_list = [file.split("\\")[1].split(".")[0] for file in pestudio_valid_file_list]

    pestudio_X_train, pestudio_y_train = [], []
    pestudio_X_valid, pestudio_y_valid = [], []

    for fname in tqdm(pestudio_fname_list):
        feature_vector = []
        label = label_table[fname]
        path = f"./data/PESTUDIO/학습데이터/{fname}.json"
        feature_vector += PestudioParser(path).process_report()
        pestudio_X_train.append(feature_vector)
        pestudio_y_train.append(label)

    for fname in tqdm(pestudio_valid_fname_list):
        feature_vector = []
        label = test_table[fname]
        path = f"./data/PESTUDIO/검증데이터/{fname}.json"
        feature_vector += PestudioParser(path).process_report()
        pestudio_X_valid.append(feature_vector)
        pestudio_y_valid.append(label)

    if manual_test:
        models = []
        for model in ["rf"]:
            clf = train(pestudio_X_train, pestudio_y_train, model)
            models.append(clf)

        for model in models:
            evaluate(pestudio_X_valid, pestudio_y_valid, model)

    return [pestudio_X_train, pestudio_y_train, pestudio_X_valid, pestudio_y_valid]

peminer = peminer_dataset()
ember = ember_dataset()
pestudio = pestudio_dataset()
