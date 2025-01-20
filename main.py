#! /usr/bin/env python3
import json
import yaml
from pathlib import Path
import importlib
import os
import pipes
import sys
import datetime
import shutil
import glob 
import logging

from dommel_library.datastructs.dict import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


"""
Main.py
-------
Entry point for training and evaluating experiments with scripts. This script
will check for the DISPLAY variable and switch between matplotlib backends
if necessary. It will also parse command line arguments of the type
`<key>=<value>` to a config dictionary ready to be merged in a config object.
Missing arguments will be filled in with defaults.
Note if the `config` argument is provided all defaults will be read from
a config yaml file.
"""
def get_model_parameters(log_dir, epoch=None):
    #check if we inputed a param file
    dir, ext = os.path.splitext(log_dir)
    if ext == '.pt':
        return log_dir

    model_dir = os.path.join(log_dir, "models")
    if epoch is None:
        model_files = glob.glob(os.path.join(model_dir, "*.pt"))
        model = max(model_files, key=os.path.getctime)
    else:
        model = os.path.join(model_dir, "model-{:04d}.pt".format(epoch))
    return model


def run_experiment(mode, experiment_file, args):
    logger.info("Run experiment %s in %s mode", experiment_file, mode)
    experiment_module, ext = os.path.splitext(experiment_file)
    experiment_name = os.path.basename(experiment_module)
    if ext == ".py":
        # path to .py file given
        config_file = experiment_file[:-3] + ".yml"
    elif ext == "":
        if experiment_name == "":
            # path to directory with trailing /
            experiment_name = os.path.basename(experiment_module[0:-1])
            experiment_module = os.path.join(experiment_module,
                                             experiment_name)
        else:
            # path to directory without trailing /
            experiment_module = os.path.join(experiment_module,
                                             experiment_name)
        experiment_file = experiment_module + ".py"
        config_file = experiment_module + ".yml"
    else:
        print("Invalid mode (", mode, ") or experiment (" + experiment_file + ")")
        print('Usage: ./main.py [run|evaluate] [experiment_file] [args]')
        exit(-1)

    # parse args that are key-value pairs
    overrides = Dict({})
    if args is not None:
        for arg in args:
            k, v = arg.split("=")

            # override config file
            if k == "config":
                if os.path.exists(v):
                    config_file = v
                else:
                    _, ext = os.path.splitext(v)
                    if ext != ".yml":
                        v = v + ".yml"
                    f = os.path.join(os.path.dirname(experiment_file), v)
                    if os.path.exists(f):
                        config_file = f
                    else:
                        raise Exception("Invalid configuration file: " + v)

                overrides["config"] = config_file
                continue
  

            # keys might be hierarchical with . notation
            overrides_dict = overrides
            keys = k.split(".")
            while len(keys) > 1:
                key = keys.pop(0)
                if key not in overrides_dict.keys():
                    overrides_dict[key] = {}
                overrides_dict = overrides_dict[key]
            k = keys[0]

            # check for json
            if v.startswith("{") or v.startswith("["):
                v = json.loads(v)

            # check for booleans
            elif v in ["True", "true", "False", "false"]:
                v = v in ["True", "true"]

            # check for None
            elif v == "None":
                v = None

            # check for numbers
            else:
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        pass

            overrides_dict[k] = v
    
    
    
    # set the log_dir for this experiment
    run_id = overrides.get("id", datetime.datetime.now().isoformat())
    run_id = run_id.replace(":", "")
    config_file_name = os.path.basename(config_file)
    config_id, _ = os.path.splitext(config_file_name)
    experiment_dir = overrides.get("experiment_dir",
                                os.path.join('runs', experiment_name))
    overrides["experiment_dir"] = experiment_dir
    log_dir = os.path.join(experiment_dir, config_id, run_id)
    overrides["log_dir"] = log_dir

    # create default paths
    if mode == 'run':
        # check if we are resuming, if not make sure we have a new log_dir
        resume = overrides.get(
            "resume", ("start_epoch" in overrides.keys()))
        if resume:
            # load the config file from the resuming experiment
            config_file = os.path.join(log_dir, experiment_name + ".yml")

            # if start_epoch not set, continue from last checkpoint
            if "start_epoch" not in overrides.keys():
                last_model = get_model_parameters(log_dir)
                last_epoch = int(last_model.split(".")[-2].split("-")[-1])
                overrides["start_epoch"] = last_epoch
        else:
            # don't overwrite existing experiment logs
            if os.path.exists(log_dir):
                count = 1
                log_dir = os.path.join(experiment_dir, config_id,
                                       run_id + '_' + str(count))

                while os.path.exists(log_dir):
                    count += 1
                    log_dir = os.path.join(experiment_dir, config_id,
                                           run_id + '_' + str(count))

                run_id + '_' + str(count)
                overrides["log_dir"] = log_dir

            overrides["start_epoch"] = 0
            os.makedirs(experiment_dir, exist_ok=True)
            os.makedirs(log_dir)

            src_destination_path = os.path.join(log_dir, 'src')
            src_source_path = os.path.dirname(os.path.abspath(__file__))

            def ignore_files(d, files):
                return set(f for f in files
                           if f in ["runs", ".git"] or  # ignore these dirs
                           (not f.endswith(".py") and  # and only include .py
                            not os.path.isdir(os.path.join(d, f))))
            try:
                shutil.copytree(
                    src_source_path,
                    src_destination_path,
                    ignore=ignore_files
                )  # ignore runs and git files
            except OSError:
                logger.error(
                    "Warning: failed to copy source files into the log path")

            # record the executed command
            arg_string = ' '.join(pipes.quote(a) for a in args)
            cmd = './main.py run' + experiment_file + ' ' + arg_string
            try:
                print(cmd, file=open(os.path.join(log_dir, 'cmd'), 'w'))
            except OSError:
                logger.error(
                    "Warning: failed to log the command into the log path")

    else:
        # use the config file from the previous experiment if id specified
        if "id" in overrides.keys():
            config_file = os.path.join(log_dir, experiment_name + ".yml")
    # read config_file
    try:
        logger.info("Read config %s", config_file)
        config = Dict(yaml.load(open(config_file, "r"),
                                Loader=yaml.FullLoader))
        
    except OSError as e:
        logger.error(
            "Error: configuration file %s not available", config_file)
        config = Dict({})
        
    if 'log_dir' in config:
        overrides["log_dir"] = config['log_dir']
    if 'experiment_dir' in config:
        overrides["experiment_dir"] = config['experiment_dir']
   
    # override with CLI args
    config.update(overrides)

    logger.info("Experiment ID " + overrides["experiment_dir"])
    logger.info("Log directory " + overrides["log_dir"])

    # dump the overriden config yaml
    if mode != "evaluate":
        
        try:
            config_destination_path = os.path.join(log_dir,
                                                   experiment_name + ".yml")
            count = 1
            while os.path.exists(config_destination_path):
                config_destination_path = os.path.join(log_dir,
                                                       experiment_name + "_" + str(count) + ".yml")
                count += 1

            yaml.dump(config.dict(), open(config_destination_path, "w"),
                      default_flow_style=False)
        except OSError:
            logger.error(
                "Warning: failed to write the config into the log path")
    # load module and run
    experiment_module = experiment_module.replace('/', '.')
    experiment_module = experiment_module.replace("\\", ".")
    if experiment_module[0:1] == "..":
        del experiment_module[0]
    experiment = importlib.import_module(experiment_module)
    
    func = None
    try:
        func = getattr(experiment, mode)
    except AttributeError:
        logger.error("Unsupported mode: %s", mode)

    if func is not None:
        func(config)


def main(args):
    if len(args) < 3:
        print('Usage: ./main.py [run|evaluate] [experiment_file] [args]')
        exit(-1)
    else:
        run_experiment(args[1], args[2], args[3:])


if __name__ == "__main__":
    try:
        main(sys.argv)
    except:
        logger.info("Exiting experiment due to exception", exc_info=True)
        raise
    finally:
        logger.info("Experiment done.")

        # flush the logs now, just to be sure all output is seen
        logging.shutdown()

        # Flush stdout and stderr
        sys.stdout.flush()
        sys.stderr.flush()
