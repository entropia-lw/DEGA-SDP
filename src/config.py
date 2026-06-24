"""Experiment constants for DEGA-SDP."""

KAMEI_SYSTEMS = ["bugzilla", "columba", "jdt", "mozilla", "platform", "postgres"]

PROMISE6_PAIRS = [
    ("xerces-1.3.csv", "ivy-2.0.csv"),
    ("ant-1.6.csv", "camel-1.4.csv"),
    ("jedit-4.1.csv", "camel-1.4.csv"),
    ("xalan-2.5.csv", "lucene-2.2.csv"),
    ("xalan-2.5.csv", "xerces-1.3.csv"),
    ("ivy-2.0.csv", "xerces-1.3.csv"),
]

WPDP_KAMEI_NESTED_F1 = {
    "bugzilla": 0.7481,
    "columba": 0.7754,
    "jdt": 0.8268,
    "mozilla": 0.9304,
    "platform": 0.8410,
    "postgres": 0.7986,
}
WPDP_KAMEI_BEST_F1 = {
    "bugzilla": 0.7554,
    "columba": 0.7690,
    "jdt": 0.8245,
    "mozilla": 0.9344,
    "platform": 0.8372,
    "postgres": 0.7976,
}

WPDP_PROMISE_NESTED_F1 = {
    "ant": 0.7890,
    "camel": 0.8072,
    "jedit": 0.8393,
    "log4j": 0.8245,
    "lucene": 0.6314,
    "poi": 0.8038,
    "synapse": 0.8121,
    "velocity": 0.8116,
    "xalan": 0.8414,
    "xerces": 0.8274,
}
WPDP_PROMISE_BEST_F1 = {
    "ant": 0.7662,
    "camel": 0.7943,
    "jedit": 0.8362,
    "log4j": 0.8308,
    "lucene": 0.7133,
    "poi": 0.8068,
    "synapse": 0.7987,
    "velocity": 0.8367,
    "xalan": 0.8316,
    "xerces": 0.8376,
}

CPDP_KAMEI_NESTED_F1 = 0.7073
CPDP_KAMEI_BEST_F1 = 0.6940

CPDP_PROMISE6_NESTED_F1 = {
    ("xerces-1.3.csv", "ivy-2.0.csv"): 0.8910,
    ("ant-1.6.csv", "camel-1.4.csv"): 0.7410,
    ("jedit-4.1.csv", "camel-1.4.csv"): 0.7345,
    ("xalan-2.5.csv", "lucene-2.2.csv"): 0.5256,
    ("xalan-2.5.csv", "xerces-1.3.csv"): 0.6645,
    ("ivy-2.0.csv", "xerces-1.3.csv"): 0.8565,
}
CPDP_PROMISE6_BEST_F1 = {
    ("xerces-1.3.csv", "ivy-2.0.csv"): 0.4729,
    ("ant-1.6.csv", "camel-1.4.csv"): 0.3954,
    ("jedit-4.1.csv", "camel-1.4.csv"): 0.6934,
    ("xalan-2.5.csv", "lucene-2.2.csv"): 0.7832,
    ("xalan-2.5.csv", "xerces-1.3.csv"): 0.4005,
    ("ivy-2.0.csv", "xerces-1.3.csv"): 0.4207,
}

K_VALUES = [2, 3, 4, 5, 6]
BETA_VALUES = [0.25, 0.5, 0.75, 1.0]
