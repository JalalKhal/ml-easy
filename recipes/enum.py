from enum import Enum


class MLFlowErrorCode(Enum):
    INTERNAL_ERROR = 1
    INVALID_PARAMETER_VALUE = 2


class StepStatus(Enum):
    """
    Represents the execution status of a step.
    """

    # Indicates that no execution status information is available for the step,
    # which may occur if the step has never been run or its outputs have been cleared
    UNKNOWN = 'UNKNOWN'
    # Indicates that the step is currently running
    RUNNING = 'RUNNING'
    # Indicates that the step completed successfully
    SUCCEEDED = 'SUCCEEDED'
    # Indicates that the step completed with one or more failures
    FAILED = 'FAILED'


class StepClass(Enum):
    """
    Represents the class of a step.
    """

    # Indicates that the step class is unknown.
    UNKNOWN = 'UNKNOWN'
    # Indicates that the step runs at training time.
    TRAINING = 'TRAINING'
    # Indicates that the step runs at inference time.
    PREDICTION = 'PREDICTION'


class StepExecutionStateKeys(Enum):
    KEY_STATUS = 'recipe_step_execution_status'
    KEY_LAST_UPDATED_TIMESTAMP = 'recipe_step_execution_last_updated_timestamp'
    KEY_STACK_TRACE = 'recipe_step_stack_trace'


class EncodingType(Enum):
    UTF8 = 'utf-8'
    ASCII = 'ascii'
    LATIN1 = 'latin-1'
    UTF16 = 'utf-16'


class SeparatorType(Enum):
    COMMA = ','
    SEMICOLON = ';'


class ScoreType(Enum):
    AccuracyScore = 'accuracy_score'
    F1Score = 'f1_score'
    AUCScore = 'auc_score'
    MAEScore = 'mean_absolute_error_score'
    MSEScore = 'mean_squared_error_score'
    R2Score = 'r2_score'


class FilterType(Enum):
    EQUAL = 'EqualFilter'
    IN = 'InFilter'


class SourceType(Enum):
    SQL_ALCHEMY_BASED = 'sql_alchemy_based'
