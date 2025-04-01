import os
import boto3
import sagemaker
import sagemaker.session
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model
from sagemaker.transformer import Transformer
from sagemaker.model_metrics import MetricsSource, ModelMetrics, FileSource
from sagemaker.drift_check_baselines import DriftCheckBaselines
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet, Join
from sagemaker.workflow.parameters import ParameterBoolean, ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CreateModelStep, TransformStep
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.clarify_check_step import (
    DataBiasCheckConfig,
    ClarifyCheckStep,
    ModelBiasCheckConfig,
    ModelPredictedLabelConfig,
    ModelExplainabilityCheckConfig,
    SHAPConfig
)
from sagemaker.workflow.quality_check_step import (
    DataQualityCheckConfig,
    ModelQualityCheckConfig,
    QualityCheckStep,
)
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.model_monitor import DatasetFormat
from sagemaker.clarify import BiasConfig, DataConfig, ModelConfig
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.pipeline_context import PipelineSession

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_session(region, default_bucket):
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline_session(region, default_bucket):
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

def get_pipeline(
    region,
    role=None,
    default_bucket=None,
    model_package_group_name="FraudDetectionPackageGroup",
    pipeline_name="FraudDetectionPipeline",
    base_job_prefix="FraudDetection",
    processing_instance_type="ml.m5.xlarge",
    training_instance_type="ml.m5.xlarge",
    sagemaker_project_name=None,
):
    sagemaker_session = get_session(region, default_bucket)
    default_bucket = sagemaker_session.default_bucket()
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    pipeline_session = get_pipeline_session(region, default_bucket)

    # Pipeline parameters
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="PendingManualApproval")
    input_data = ParameterString(
        name="InputDataUrl",
        default_value=f"s3://akshronix-frauddata/Chandan's Playground MLOPS/Test 1/Whole Fraud Dataframe.csv",  # Replace with your fraud dataset S3 path
    )

    # Quality and bias check parameters
    skip_check_data_quality = ParameterBoolean(name="SkipDataQualityCheck", default_value=False)
    register_new_baseline_data_quality = ParameterBoolean(name="RegisterNewDataQualityBaseline", default_value=False)
    supplied_baseline_statistics_data_quality = ParameterString(name="DataQualitySuppliedStatistics", default_value='')
    supplied_baseline_constraints_data_quality = ParameterString(name="DataQualitySuppliedConstraints", default_value='')

    skip_check_data_bias = ParameterBoolean(name="SkipDataBiasCheck", default_value=False)
    register_new_baseline_data_bias = ParameterBoolean(name="RegisterNewDataBiasBaseline", default_value=False)
    supplied_baseline_constraints_data_bias = ParameterString(name="DataBiasSuppliedBaselineConstraints", default_value='')

    skip_check_model_quality = ParameterBoolean(name="SkipModelQualityCheck", default_value=False)
    register_new_baseline_model_quality = ParameterBoolean(name="RegisterNewModelQualityBaseline", default_value=False)
    supplied_baseline_statistics_model_quality = ParameterString(name="ModelQualitySuppliedStatistics", default_value='')
    supplied_baseline_constraints_model_quality = ParameterString(name="ModelQualitySuppliedConstraints", default_value='')

    skip_check_model_bias = ParameterBoolean(name="SkipModelBiasCheck", default_value=False)
    register_new_baseline_model_bias = ParameterBoolean(name="RegisterNewModelBiasBaseline", default_value=False)
    supplied_baseline_constraints_model_bias = ParameterString(name="ModelBiasSuppliedBaselineConstraints", default_value='')

    skip_check_model_explainability = ParameterBoolean(name="SkipModelExplainabilityCheck", default_value=False)
    register_new_baseline_model_explainability = ParameterBoolean(name="RegisterNewModelExplainabilityBaseline", default_value=False)
    supplied_baseline_constraints_model_explainability = ParameterString(name="ModelExplainabilitySuppliedBaselineConstraints", default_value='')

    # Preprocessing step
    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/sklearn-fraud-preprocess",
        sagemaker_session=pipeline_session,
        role=role,
    )

    step_args = sklearn_processor.run(
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
            ProcessingOutput(output_name="preprocessor", source="/opt/ml/processing/artifacts", destination=f"s3://{default_bucket}/{base_job_prefix}/preprocessor"),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
        arguments=["--input-data", input_data],
    )

    step_process = ProcessingStep(
        name="PreprocessFraudData",
        step_args=step_args,
    )

    # Data Quality Check
    check_job_config = CheckJobConfig(
        role=role,
        instance_count=1,
        instance_type="ml.c5.xlarge",
        volume_size_in_gb=120,
        sagemaker_session=pipeline_session,
    )

    data_quality_check_config = DataQualityCheckConfig(
        baseline_dataset=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
        dataset_format=DatasetFormat.csv(header=False, output_columns_position="START"),
        output_s3_uri=Join(on='/', values=['s3:/', default_bucket, base_job_prefix, ExecutionVariables.PIPELINE_EXECUTION_ID, 'dataqualitycheckstep']),
    )

    data_quality_check_step = QualityCheckStep(
        name="DataQualityCheckStep",
        skip_check=skip_check_data_quality,
        register_new_baseline=register_new_baseline_data_quality,
        quality_check_config=data_quality_check_config,
        check_job_config=check_job_config,
        supplied_baseline_statistics=supplied_baseline_statistics_data_quality,
        supplied_baseline_constraints=supplied_baseline_constraints_data_quality,
        model_package_group_name=model_package_group_name,
    )

    # Data Bias Check
    data_bias_data_config = DataConfig(
        s3_data_input_path=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
        s3_output_path=Join(on='/', values=['s3:/', default_bucket, base_job_prefix, ExecutionVariables.PIPELINE_EXECUTION_ID, 'databiascheckstep']),
        label=-1,  # Fraud label is the last column after preprocessing
        dataset_type="text/csv",
    )

    data_bias_config = BiasConfig(
        label_values_or_threshold=[1],  # Fraud = 1
        facet_name=[0],  # Example: Check bias on the first feature (adjust based on your data)
        facet_values_or_threshold=[[0.5]],  # Example threshold
    )

    data_bias_check_config = DataBiasCheckConfig(
        data_config=data_bias_data_config,
        data_bias_config=data_bias_config,
    )

    data_bias_check_step = ClarifyCheckStep(
        name="DataBiasCheckStep",
        clarify_check_config=data_bias_check_config,
        check_job_config=check_job_config,
        skip_check=skip_check_data_bias,
        register_new_baseline=register_new_baseline_data_bias,
        model_package_group_name=model_package_group_name,
    )

    # Training Step
    model_path = f"s3://{default_bucket}/{base_job_prefix}/FraudTrain"
    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.0-1",
        py_version="py3",
        instance_type=training_instance_type,
    )

    xgb_train = Estimator(
        image_uri=image_uri,
        instance_type=training_instance_type,
        instance_count=1,
        output_path=model_path,
        base_job_name=f"{base_job_prefix}/fraud-train",
        sagemaker_session=pipeline_session,
        role=role,
    )

    xgb_train.set_hyperparameters(
        objective="binary:logistic",  # Binary classification
        num_round=50,
        max_depth=5,
        eta=0.2,
        gamma=4,
        min_child_weight=6,
        subsample=0.7,
        eval_metric="auc",  # Use AUC for evaluation
    )

    step_args = xgb_train.fit(
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
    )

    step_train = TrainingStep(
        name="TrainFraudModel",
        step_args=step_args,
        depends_on=["DataQualityCheckStep", "DataBiasCheckStep"],
    )

    # Create Model
    model = Model(
        image_uri=image_uri,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=pipeline_session,
        role=role,
    )

    step_args = model.create(
        instance_type="ml.m5.large",
        accelerator_type="ml.eia1.medium",
    )

    step_create_model = ModelStep(
        name="FraudCreateModel",
        step_args=step_args,
    )

    # Batch Transform
    transformer = Transformer(
        model_name=step_create_model.properties.ModelName,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        accept="text/csv",
        assemble_with="Line",
        output_path=f"s3://{default_bucket}/FraudTransform",
        sagemaker_session=pipeline_session,
    )

    transform_inputs = TransformInput(
        data=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
    )

    step_args = transformer.transform(
        data=transform_inputs.data,
        input_filter="$[0:-1]",  # Exclude the label column
        join_source="Input",
        output_filter="$[0,-1]",  # Prediction and original label
        content_type="text/csv",
        split_type="Line",
    )

    step_transform = TransformStep(
        name="FraudTransform",
        step_args=step_args,
    )

    # Model Quality Check
    model_quality_check_config = ModelQualityCheckConfig(
        baseline_dataset=step_transform.properties.TransformOutput.S3OutputPath,
        dataset_format=DatasetFormat.csv(header=False),
        output_s3_uri=Join(on='/', values=['s3:/', default_bucket, base_job_prefix, ExecutionVariables.PIPELINE_EXECUTION_ID, 'modelqualitycheckstep']),
        problem_type="BinaryClassification",  # Changed to classification
        inference_attribute="_c0",  # Predicted probability/class
        ground_truth_attribute="_c1",  # True label
    )

    model_quality_check_step = QualityCheckStep(
        name="ModelQualityCheckStep",
        skip_check=skip_check_model_quality,
        register_new_baseline=register_new_baseline_model_quality,
        quality_check_config=model_quality_check_config,
        check_job_config=check_job_config,
        supplied_baseline_statistics=supplied_baseline_statistics_model_quality,
        supplied_baseline_constraints=supplied_baseline_constraints_model_quality,
        model_package_group_name=model_package_group_name,
    )

    # Model Bias Check
    model_bias_data_config = DataConfig(
        s3_data_input_path=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
        s3_output_path=Join(on='/', values=['s3:/', default_bucket, base_job_prefix, ExecutionVariables.PIPELINE_EXECUTION_ID, 'modelbiascheckstep']),
        label=-1,  # Fraud label is the last column
        dataset_type="text/csv",
    )

    model_config = ModelConfig(
        model_name=step_create_model.properties.ModelName,
        instance_count=1,
        instance_type='ml.m5.large',
    )

    model_bias_config = BiasConfig(
        label_values_or_threshold=[1],  # Fraud = 1
        facet_name=[0],  # Example: First feature (adjust as needed)
        facet_values_or_threshold=[[0.5]],  # Example threshold
    )

    model_bias_check_config = ModelBiasCheckConfig(
        data_config=model_bias_data_config,
        data_bias_config=model_bias_config,
        model_config=model_config,
        model_predicted_label_config=ModelPredictedLabelConfig(),
    )

    model_bias_check_step = ClarifyCheckStep(
        name="ModelBiasCheckStep",
        clarify_check_config=model_bias_check_config,
        check_job_config=check_job_config,
        skip_check=skip_check_model_bias,
        register_new_baseline=register_new_baseline_model_bias,
        supplied_baseline_constraints=supplied_baseline_constraints_model_bias,
        model_package_group_name=model_package_group_name,
    )

    # Model Explainability Check
    model_explainability_data_config = DataConfig(
        s3_data_input_path=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
        s3_output_path=Join(on='/', values=['s3:/', default_bucket, base_job_prefix, ExecutionVariables.PIPELINE_EXECUTION_ID, 'modelexplainabilitycheckstep']),
        label=-1,  # Fraud label is the last column
        dataset_type="text/csv",
    )

    shap_config = SHAPConfig(
        seed=123,
        num_samples=10,
    )

    model_explainability_check_config = ModelExplainabilityCheckConfig(
        data_config=model_explainability_data_config,
        model_config=model_config,
        explainability_config=shap_config,
    )

    model_explainability_check_step = ClarifyCheckStep(
        name="ModelExplainabilityCheckStep",
        clarify_check_config=model_explainability_check_config,
        check_job_config=check_job_config,
        skip_check=skip_check_model_explainability,
        register_new_baseline=register_new_baseline_model_explainability,
        supplied_baseline_constraints=supplied_baseline_constraints_model_explainability,
        model_package_group_name=model_package_group_name,
    )

    # Evaluation Step
    script_eval = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/script-fraud-eval",
        sagemaker_session=pipeline_session,
        role=role,
    )

    step_args = script_eval.run(
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
        ],
        code=os.path.join(BASE_DIR, "evaluate.py"),
    )

    evaluation_report = PropertyFile(
        name="FraudEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    step_eval = ProcessingStep(
        name="EvaluateFraudModel",
        step_args=step_args,
        property_files=[evaluation_report],
    )

    # Model Metrics and Drift Check Baselines
    model_metrics = ModelMetrics(
        model_data_statistics=MetricsSource(s3_uri=data_quality_check_step.properties.CalculatedBaselineStatistics, content_type="application/json"),
        model_data_constraints=MetricsSource(s3_uri=data_quality_check_step.properties.CalculatedBaselineConstraints, content_type="application/json"),
        bias_pre_training=MetricsSource(s3_uri=data_bias_check_step.properties.CalculatedBaselineConstraints, content_type="application/json"),
        model_statistics=MetricsSource(s3_uri=model_quality_check_step.properties.CalculatedBaselineStatistics, content_type="application/json"),
        model_constraints=MetricsSource(s3_uri=model_quality_check_step.properties.CalculatedBaselineConstraints, content_type="application/json"),
        bias_post_training=MetricsSource(s3_uri=model_bias_check_step.properties.CalculatedBaselineConstraints, content_type="application/json"),
        bias=MetricsSource(s3_uri=model_bias_check_step.properties.CalculatedBaselineConstraints, content_type="application/json"),
        explainability=MetricsSource(s3_uri=model_explainability_check_step.properties.CalculatedBaselineConstraints, content_type="application/json"),
    )

    drift_check_baselines = DriftCheckBaselines(
        model_data_statistics=MetricsSource(s3_uri=data_quality_check_step.properties.BaselineUsedForDriftCheckStatistics, content_type="application/json"),
        model_data_constraints=MetricsSource(s3_uri=data_quality_check_step.properties.BaselineUsedForDriftCheckConstraints, content_type="application/json"),
        bias_pre_training_constraints=MetricsSource(s3_uri=data_bias_check_step.properties.BaselineUsedForDriftCheckConstraints, content_type="application/json"),
        bias_config_file=FileSource(s3_uri=model_bias_check_config.monitoring_analysis_config_uri, content_type="application/json"),
        model_statistics=MetricsSource(s3_uri=model_quality_check_step.properties.BaselineUsedForDriftCheckStatistics, content_type="application/json"),
        model_constraints=MetricsSource(s3_uri=model_quality_check_step.properties.BaselineUsedForDriftCheckConstraints, content_type="application/json"),
        bias_post_training_constraints=MetricsSource(s3_uri=model_bias_check_step.properties.BaselineUsedForDriftCheckConstraints, content_type="application/json"),
        explainability_constraints=MetricsSource(s3_uri=model_explainability_check_step.properties.BaselineUsedForDriftCheckConstraints, content_type="application/json"),
        explainability_config_file=FileSource(s3_uri=model_explainability_check_config.monitoring_analysis_config_uri, content_type="application/json"),
    )

    # Register Model
    model = Model(
        image_uri=image_uri,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=pipeline_session,
        role=role,
    )

    step_args = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
        drift_check_baselines=drift_check_baselines,
    )

    step_register = ModelStep(
        name="RegisterFraudModel",
        step_args=step_args,
    )

    # Condition Step (F1-score ≥ 0.8)
    cond_gte = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="classification_metrics.f1.value",  # Updated for classification
        ),
        right=0.8,  # Threshold for F1-score
    )

    step_cond = ConditionStep(
        name="CheckF1FraudEvaluation",
        conditions=[cond_gte],
        if_steps=[step_register],
        else_steps=[],
    )

    # Pipeline Definition
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            model_approval_status,
            input_data,
            skip_check_data_quality,
            register_new_baseline_data_quality,
            supplied_baseline_statistics_data_quality,
            supplied_baseline_constraints_data_quality,
            skip_check_data_bias,
            register_new_baseline_data_bias,
            supplied_baseline_constraints_data_bias,
            skip_check_model_quality,
            register_new_baseline_model_quality,
            supplied_baseline_statistics_model_quality,
            supplied_baseline_constraints_model_quality,
            skip_check_model_bias,
            register_new_baseline_model_bias,
            supplied_baseline_constraints_model_bias,
            skip_check_model_explainability,
            register_new_baseline_model_explainability,
            supplied_baseline_constraints_model_explainability,
        ],
        steps=[
            step_process,
            data_quality_check_step,
            data_bias_check_step,
            step_train,
            step_create_model,
            step_transform,
            model_quality_check_step,
            model_bias_check_step,
            model_explainability_check_step,
            step_eval,
            step_cond,
        ],
        sagemaker_session=pipeline_session,
    )
    return pipeline