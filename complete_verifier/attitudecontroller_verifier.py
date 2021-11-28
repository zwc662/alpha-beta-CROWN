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
    
    h = ["model"]
    arguments.Config.add_argument("--model", type=str, default="please_specify_model_name", help='Name of model. Model must be defined in the load_verification_dataset() function in utils.py.', hierarchy=h + ["name"])

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


    save_path = 'Verified_ret_[{}]_start={}_end={}_iter={}_b={}_timeout={}_branching={}-{}-{}_lra-init={}_lra={}_lrb={}_PGD={}.npy'. \
        format(arguments.Config['model']['name'], arguments.Config["data"]["start"],  arguments.Config["data"]["end"], arguments.Config["solver"]["beta-crown"]["iteration"], arguments.Config["solver"]["beta-crown"]["batch_size"],
               arguments.Config["bab"]["timeout"], arguments.Config["bab"]["branching"]["method"], arguments.Config["bab"]["branching"]["reduceop"],
               arguments.Config["bab"]["branching"]["candidates"], arguments.Config["solver"]["alpha-crown"]["lr_alpha"], arguments.Config["solver"]["beta-crown"]["lr_alpha"], arguments.Config["solver"]["beta-crown"]["lr_beta"], arguments.Config["attack"]["pgd_order"])
    print(f'saving results to {save_path}')
 
    bnb_ids = range(arguments.Config["data"]["start"],  arguments.Config["data"]["end"])
    ret, lb_record, attack_success = [], [], []
    mip_unsafe, mip_safe, mip_unknown = [], [], []
    verified_acc = len(bnb_ids)
    verified_failed = []
    nat_acc = len(bnb_ids)
    cnt = 0
    orig_timeout = arguments.Config["bab"]["timeout"]

    init_global_lb = saved_bounds = saved_slopes = None

    # Load model
    model_ori = AttitudeController().to(arguments.Config["general"]["device"])

    # The initial state min, max as a single range are loaded into a list
    x_min = torch.tensor(arguments.Config["init"]["min"]).unsqueeze(0).to(arguments.Config["general"]["device"])
    x_max = torch.tensor(arguments.Config["init"]["max"]).unsqueeze(0).to(arguments.Config["general"]["device"])
    x = (x_max + x_min)/2.
    perturb_eps = x - x_min

    # Initialize lists of current and next state ranges and control output ranges 
    X_min, X_max, U_min, U_max, X_nxt, X_min_nxt, X_max_nxt = [x_min], [x_max], [], [], [], [], []

    # Test run the initial control output given a medium state
    with torch.no_grad():
        u_pred = model_ori(x)
        print("Given medium input {}".format(x))
        print("Attitude controller's output {}".format(u_pred))

    # Run step by step
    for step in bnb_ids:
        # Extract each range from the range list
        for idx in range(len(X_max)):
            x_max = X_max[idx]
            x_min = X_min[idx]
            x = (x_max + x_min)/2.

            if arguments.Config["general"]["enable_incomplete_verification"] or arguments.Config["general"]["complete_verifier"] == "bab-refine":
                print(">>>>>>>>>>>>>>>Incomplete verification is enabled by default. The intermediate lower and upper bounds will be reused in bab and mip.")
                start_incomplete = time.time()
                
                data = x
                data_ub = x_max
                data_lb = x_min

                ############ incomplete_verification execution
                verified_status, init_global_lb, saved_bounds, saved_slopes = incomplete_verifier(
                    model_ori = model_ori, data = data, 
                    norm = arguments.Config["specification"]["norm"], \
                    y = None, data_ub=data_ub, data_lb=data_lb, eps=0.)
                ############
                print(verified_status, init_global_lb, saved_bounds)
                lower_bounds, upper_bounds = saved_bounds[1], saved_bounds[2]
                arguments.Config["bab"]["timeout"] -= (time.time()-start_incomplete)
                ret.append([idx, 0, 0, time.time()-start_incomplete, idx, -1, np.inf, np.inf])

            if arguments.Config["general"]["mode"] == "verified-acc":
                if arguments.Config["general"]["enable_incomplete_verification"]:
                    # We have initial incomplete bounds.
                    labels_to_verify = init_global_lb.argsort().squeeze().tolist()
                else:
                    labels_to_verify = list(range(arguments.Config["data"]["num_classes"]))
            elif arguments.Config["general"]["mode"] == "runnerup":
                labels_to_verify = [u_pred.argsort(descending=True)[1]]
            else:
                raise ValueError("unknown verification mode")
            
            pidx_all_verified = True
            print("Summary: ", verified_status, init_global_lb, saved_bounds, labels_to_verify)
            exit(0)

if __name__ == "__main__":
    config_args()
    main()