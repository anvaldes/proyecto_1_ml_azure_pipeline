from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, Output, command, dsl
from azure.ai.ml.entities import Environment, BuildContext, AmlCompute
from azure.ai.ml.constants import AssetTypes

# 1. Conexión al workspace
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="88e891ec-0f1c-4cba-9ed0-2dafc3c69861",
    resource_group_name="rg-proyecto1",
    workspace_name="ws-proyecto1-v2"
)
print("✅ 1. Conectado al Workspace")

# 2. Crear cluster si no existe
cluster_name = "cpu-cluster"
try:
    ml_client.compute.get(cluster_name)
    print(f"✅ 2. Cluster '{cluster_name}' ya existe")
except:
    ml_client.compute.begin_create_or_update(
        AmlCompute(
            name=cluster_name,
            size="Standard_DS3_v2",
            min_instances=0,
            max_instances=2,
            idle_time_before_scale_down=120
        )
    ).result()
    print(f"✅ 2. Cluster '{cluster_name}' creado")

# 3. Crear environment desde Dockerfile
env = Environment(
    name="ml-custom-env",
    build=BuildContext(path="."),
)
env = ml_client.environments.create_or_update(env)
print(f"✅ 3. Environment '{env.name}' creado con versión: {env.version}")

# 4. Definir componentes como funciones (no ejecutar command directamente)
def create_train_component(env, compute_target):
    return command(
        name="train_component",
        code=".",
        command=(
            "python train.py "
            "--data_dir ${{inputs.data_dir}} "
            "--model_output ${{outputs.model_output}} "
            "--metrics_output ${{outputs.metrics_output}}"
        ),
        inputs={
            "data_dir": Input(type=AssetTypes.URI_FOLDER)
        },
        outputs={
            "model_output": Output(type=AssetTypes.URI_FOLDER),
            "metrics_output": Output(type=AssetTypes.URI_FOLDER)
        },
        environment=env.id,
        compute=compute_target,
    )

def create_eval_component(env, compute_target):
    return command(
        name="eval_component",
        code=".",
        command="python evaluate.py --model_dir ${{inputs.model_dir}} --data_dir ${{inputs.data_dir}}",
        inputs={
            "model_dir": Input(type=AssetTypes.URI_FOLDER),
            "data_dir": Input(type=AssetTypes.URI_FOLDER)
        },
        environment=env.id,
        compute=compute_target,
    )

# 5. Definir pipeline DSL
@dsl.pipeline(
    compute=cluster_name,
    description="Training + Evaluation pipeline"
)
def training_pipeline():
    train_step = create_train_component(env, cluster_name)(
        data_dir=Input(
            type=AssetTypes.URI_FOLDER,
            path="azureml://datastores/pipelineblobstore/paths/datasets/2025_06"
        )
    )

    eval_step = create_eval_component(env, cluster_name)(
        model_dir=train_step.outputs.model_output,
        data_dir=Input(
            type=AssetTypes.URI_FOLDER,
            path="azureml://datastores/pipelineblobstore/paths/datasets/2025_06"
        )
    )

    return {
        "trained_model": train_step.outputs.model_output,
        "metrics": train_step.outputs.metrics_output
    }

# 6. Ejecutar pipeline
pipeline_job = ml_client.jobs.create_or_update(training_pipeline())
print(f"✅ 6. Pipeline lanzado con ID: {pipeline_job.name}")
print(f"📊 Monitorea en:\nhttps://ml.azure.com/experiments/{pipeline_job.display_name or 'pipeline'}/runs/{pipeline_job.name}")