from enum import Enum


class EnumHeadTypes(str, Enum):
    RegressionHead = "RegressionHead"
    SimpleRegressionHead = "SimpleRegressionHead"  # previously named SimpleLinearClassifier


class EnumBackboneType(str, Enum):
    default = 'default'
    student = 'student'
    teacher = 'teacher'


class EnumMode(str, Enum):
    dino = 'dino'
    supervised = 'supervised'
    head = 'head'
    test = 'test'
    none = 'none'


class EnumDatasets(str, Enum):
    drive360 = "drive360"
    driving_dataset = "driving_dataset"
    age = "age"


class EnumUncertaintyTypes(str, Enum):
    none = "none"
    bayes_by_backprop = "bayes_by_backprop"
    mcbn = "mcbn"
    aleatoric = "aleatoric"
    aleatoric_bbb = "aleatoric_bbb"
    aleatoric_mcbn = "aleatoric_mcbn"


class EnumMetricParams(str, Enum):
    mode = "mode"
    use_for_is_best = "use_for_is_best"
    use_for_plateau_detection = "use_for_plateau_detection"
    patience = "patience"
    min_delta = "min_delta"
    percentage = "percentage"
    values = "values"
    best = "best"
    unchanged_for_epochs = "unchanged_for_epochs"


class EnumSamplerTypes(str, Enum):
    SequentialSampler = "SequentialSampler"
    DistributedSampler = "DistributedSampler"
    UpscaleDistributedSampler = "UpscaleDistributedSampler"
    DynamicDistributedSampler = "DynamicDistributedSampler"


class EnumSchedulerTypes(str, Enum):
    none = 'none'
    linear = 'linear'
    cosine = 'cosine'
    exponential = 'exponential'


class EnumBBBCostFunction(str, Enum):
    samples_per_epoch = 'samples_per_epoch'
    graves = 'graves'
    paper_proposal = 'paper_proposal'
    paper_proposal_all_epochs = 'paper_proposal_all_epochs'

