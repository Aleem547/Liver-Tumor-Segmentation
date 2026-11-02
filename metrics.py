"""Custom segmentation metrics."""


def foreground_acc(inp, targ, bkg_idx=0, axis=1):
    "Non-background accuracy for multiclass segmentation"
    targ = targ.squeeze(1)
    mask = targ != bkg_idx
    return (inp.argmax(dim=axis)[mask] == targ[mask]).float().mean()


def cust_foreground_acc(inp, targ):
    "Accuracy including the background (3 is a dummy index so class 0 is kept)"
    return foreground_acc(inp=inp, targ=targ, bkg_idx=3, axis=1)
