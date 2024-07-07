import torch


def wdl_to_cp(wdl_eval, scaling_factor=410):
    """
    Convert WDL space evaluation to centipawn space evaluation.
    WDL space evaluation is a value range [0, 1] where 0 means loss, 0.5 means draw, and 1 means win.
    Centipawn space evaluation is a value range where 100 is roughly equivalent to the value of a pawn.
    """
    # Ensure wdl_space_eval is within the valid range for logit function
    eps = 1e-7
    wdl_eval = torch.clamp(torch.tensor(wdl_eval), torch.tensor(eps), torch.tensor(1 - eps))

    # Calculate the logit
    logit_value = torch.log(wdl_eval / (1 - wdl_eval))

    # Calculate cp_space_eval
    cp_eval = scaling_factor * logit_value

    return cp_eval.item()


def cp_to_wdl(cp_eval, scaling_factor=410):
    """
    Convert centipawn space evaluation to WDL space evaluation.
    Centipawn space evaluation is a value range where 100 is roughly equivalent to the value of a pawn.
    WDL space evaluation is a value range [0, 1] where 0 means loss, 0.5 means draw, and 1 means win.
    """
    # Calculate the logit
    logit_value = int(cp_eval) / scaling_factor

    # Calculate wdl_space_eval
    wdl_eval = torch.sigmoid(torch.tensor(logit_value))

    return wdl_eval.item()
