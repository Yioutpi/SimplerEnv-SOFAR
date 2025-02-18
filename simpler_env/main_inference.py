import os
import numpy as np
import tensorflow as tf

from simpler_env.evaluation.argparse import get_args
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator
# from simpler_env.policies.octo.octo_server_model import OctoServerInference
from simpler_env.policies.rt1.rt1_model import RT1Inference
from simpler_env.policies.openvla.openvla_model import OpenVLAInference

# try:
#     from simpler_env.policies.octo.octo_model import OctoInference
# except ImportError as e:
#     print("Octo is not correctly imported.")
#     print(e)


if __name__ == "__main__":
    args = get_args()

    os.environ["DISPLAY"] = ""
    # prevent a single jax process from taking up all the GPU memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        # prevent a single tf process from taking up all the GPU memory
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=args.tf_memory_limit)],
        )
    print(f"**** {args.policy_model} ****")
    # policy model creation; update this if you are using a new policy model
    if args.policy_model == "rt1":
        assert args.ckpt_path is not None
        model = RT1Inference(
            saved_model_path=args.ckpt_path,
            policy_setup=args.policy_setup,
            action_scale=args.action_scale,
        )
    elif args.policy_model == "openvla":
        assert args.ckpt_path is not None
        model = OpenVLAInference(
            saved_model_path=args.ckpt_path,
            policy_setup=args.policy_setup,
            action_scale=args.action_scale,
        )
    elif args.policy_model == "sofar":
        model = "sofar"
    elif args.policy_model == "sofar_widowx":
        model = "sofar_widowx"
    else:
        raise NotImplementedError()
    
    if args.policy_model == "sofar_widowx":
        from simpler_env.evaluation.maniskill2_evaluator_sofar_widowx import maniskill2_evaluator_sofar_widowx
        success_arr = maniskill2_evaluator_sofar_widowx(model, args)
    elif args.policy_model == "sofar":
        from simpler_env.evaluation.maniskill2_evaluator_sofar import maniskill2_evaluator_sofar
        success_arr = maniskill2_evaluator_sofar(model, args)
    else:
        success_arr = maniskill2_evaluator(model, args)
    # run real-to-sim evaluation
    print(" " * 10, "Average success", np.mean(success_arr))
