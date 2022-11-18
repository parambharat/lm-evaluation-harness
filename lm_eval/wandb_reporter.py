from collections import defaultdict

import textwrap
import wandb
from pytablewriter import MarkdownTableWriter

wandb.require("report-editing:v0")
import wandb.apis.reports as wr
from lm_eval import utils, tasks
import json
from datetime import datetime

import inspect


def get_task_description(task):
    clazz = tasks.TASK_REGISTRY[task]
    task_doc = inspect.cleandoc(inspect.getdoc(inspect.getmodule(clazz)))
    task_citation = getattr(inspect.getmodule(clazz), "_CITATION")
    return task_doc, task_citation


def log_results(results, name):
    if wandb.run is not None:
        _run = wandb.run
        result_dict = results.copy()
        wandb.config.update(result_dict["config"])
        fewshot = result_dict["config"]["num_fewshot"]
        task_results = defaultdict(lambda: defaultdict(list))
        columns = ["Task", "Version", "num_fewshot", "Metric", "Value", "Stderr"]
        values = []
        for k, dic in result_dict["results"].items():
            version = result_dict["versions"][k]
            for m, v in dic.items():
                if m.endswith("_stderr"):
                    continue
                if m + "_stderr" in dic:
                    se = dic[m + "_stderr"]
                    values.append([k, version, fewshot, m, v, se])
                    task_results[k][m].extend(["%.4f" % v, "%.4f" % se])
                else:
                    values.append([k, version, fewshot, m, v, None])
                    task_results[k][m].extend(["%.4f" % v, ""])
        results_table = wandb.Table(columns=columns, data=values)
        _run.log({name: results_table})
        return task_results


def write_metric_table(metrics):
    md_writer = MarkdownTableWriter()
    md_writer.headers = ["Metric", "Value", "± Stderr"]
    md_values = []
    for k, v in metrics.items():
        md_values.append([f"**{k}**"] + v)
    md_writer.value_matrix = md_values
    return md_writer.dumps()


def create_report_by_task(task_results):
    tasks_report = []
    for task, metrics in task_results.items():
        tasks_report.append(wr.H3(task.title()))
        task_description, task_citation = get_task_description(task)
        if task_description:
            tasks_report.append(wr.P(task_description))
        tasks_report.append(
            wr.MarkdownBlock(
                "**Metrics**" + "\n\n" + write_metric_table(metrics) + "\n\n"
            )
        )
    return tasks_report


def get_model_name_from_args(args):
    model_args = utils.simple_parse_args_string(args.model_args)

    model_name = args.model
    if args.model == "gpt2":
        model_type = model_args.get("pretrained", "")
        if model_type:
            model_name = model_type
    elif args.model == "gpt3":
        model_type = model_args.get("engine", "")
        if model_type:
            model_name = model_name + ":" + model_type
    return model_name


def report_evaluation(
    args, results, results_md,
):
    run = wandb.init(
        project=args.wandb_project, entity=wandb.apis.PublicApi().default_entity
    )

    task_results = log_results(results, run.name)
    tasks_report = create_report_by_task(task_results)
    model_name = get_model_name_from_args(args)
    report = wr.Report(
        project=args.wandb_project,
        entity=wandb.apis.PublicApi().default_entity,
        title=f"Evaluation Report for {model_name}",
        description=f"Evaluation run on : {datetime.utcnow()}",
    )

    report.blocks = (
        [
            wr.TableOfContents(),
            wr.H1("Complete Evaluation Results"),
            wr.MarkdownBlock(results_md),
            wr.H2("Evaluation Results By Task"),
        ]
        + tasks_report
        + [
            wr.H1("Evaluation Runs"),
            wr.WeaveTableBlock(
                project=args.wandb_project,
                entity=wandb.apis.PublicApi().default_entity,
                table_name=f"{run.name}",
            ),
            wr.PanelGrid(
                runsets=[
                    wr.Runset(
                        project=args.wandb_project, entity=run.entity,
                    ).set_filters_with_python_expr(f'Name == "{str(run.name)}"'),
                ]
            ),
            wr.H1("Evaluation Config"),
            wr.CodeBlock(
                json.dumps(results["config"], indent=5).split("\n"), language="json"
            ),
        ]
    )
    report.save()
