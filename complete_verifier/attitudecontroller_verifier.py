import os
import socket
import random
import time
import gc

import numpy as np
import pandas as pd

import torch
import arguments

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from bab_verification_general import mip, incomplete_verifier, bab


from model_defs import AttitudeController
model_ori = AttitudeController()

def config_args():
    # Add arguments specific for this front-end.
    h = ["general"]
    arguments.Config.add_argument("--mode", type=str, default="verified-acc", choices=["verified-acc", "runnerup", "clean-acc", "specify-target"],
            help='Verify against all labels ("verified-acc" mode), or just the runnerup labels ("runnerup" mode), or using a specified label in dataset ("speicify-target" mode, only used for oval20).', hierarchy=h + ["mode"])
    arguments.Config.add_argument('--complete_verifier', choices=["bab", "mip", "bab-refine", "skip"], default="bab",
            help='Complete verification verifier. "bab": branch and bound with beta-CROWN; "mip": mixed integer programming (MIP) formulation; "bab-refine": branch and bound with intermediate layer bounds computed by MIP.', hierarchy=h + ["complete_verifier"])
    arguments.Config.add_argument('--no_incomplete', action='store_false', dest='incomplete',
            help='Enable/Disable initial alpha-CROWN incomplete verification (this can save GPU memory when disabled).', hierarchy=h + ["enable_incomplete_verification"])
    arguments.Config.add_argument("--crown", action='store_true', help='Compute CROWN verified accuracy before verification (not used).', hierarchy=h + ["get_crown_verified_acc"])
 
    h = ["init"]
    arguments.Config.add_argument("--min", nargs='+', type=float, default=[-0.45, -0.55, 0.65, -0.75, 0.85, -0.65], help='Min initial input vector.', hierarchy=h + ["min"])
    arguments.Config.add_argument("--max", nargs='+', type=float, default=[-0.44, -0.54, 0.66, -0.74, 0.86, -0.64], help='Max initial input vector.', hierarchy=h + ["max"])
    
    h = ["debug"]
    arguments.Config.add_argument("--lp_test", type=str, default=None, choices=["MIP", "LP", "LP_intermediate_refine", "MIP_intermediate_refine", None], help='Debugging option, do not use.', hierarchy=h + ['lp_test'])

    arguments.Config.parse_config()


def main():
    print(f'Experiments at {time.ctime()} on {socket.gethostname()}')
    torch.manual_seed(arguments.Config["general"]["seed"])
    random.seed(arguments.Config["general"]["seed"])
    np.random.seed(arguments.Config["general"]["seed"])
    if arguments.Config["general"]["device"] != 'cpu':
        torch.cuda.manual_seed_all(arguments.Config["general"]["seed"])
        # Always disable TF32 (precision is too low for verification).
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    if arguments.Config["general"]["deterministic"]:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)

    if arguments.Config["general"]["double_fp"]:
        torch.set_default_dtype(torch.float64)

    if arguments.Config["specification"]["norm"] != np.inf and arguments.Config["attack"]["pgd_order"] != "skip":
        print('Only Linf-norm attack is supported, the pgd_order will be changed to skip')
        arguments.Config["attack"]["pgd_order"] = "skip"

    with torch.no_grad():
        x_min = torch.tensor(arguments.Config["init"]["min"]).unsqueeze(0)
        x_max = torch.tensor(arguments.Config["init"]["max"]).unsqueeze(0)
        model_ori, data_max, data_min = model_ori.to(arguments.Config["general"]["device"]), data_max.to(arguments.Config["general"]["device"]), data_min.to(arguments.Config["general"]["device"])

        x = (x_min + x_max)/2.
        perturb_eps = x - x_min
        u_pred = model_ori(x)
        print("Given medium input {}".format(u_pred))
        print("Attitude controller's output {}".format(u_pred))
    init_global_lb = saved_bounds = saved_slopes = None

    if arguments.Config["general"]["enable_incomplete_verification"] or arguments.Config["general"]["complete_verifier"] == "bab-refine":
        print(">>>>>>>>>>>>>>>Incomplete verification is enabled by default. The intermediate lower and upper bounds will be reused in bab and mip.")
        start_incomplete = time.time()
        data = x
        data_max = data_ub = x_max
        data_min = data_lb = x_min

        ############ incomplete_verification execution
        verified_status, init_global_lb, saved_bounds, saved_slopes = incomplete_verifier(
            model_ori = model_ori, data = data, 
            norm = arguments.Config["specification"]["norm"], \
            y = None, data_ub=data_ub, data_lb=data_lb, eps=0.)
        ############
        print(saved_bounds)

        
   


if __name__ == "__main__":
    config_args()
    main()