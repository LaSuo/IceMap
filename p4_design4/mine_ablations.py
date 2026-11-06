"""Explicit parameter-matched controls; widths are implementation choices."""

from network import SPAIIceInversionNet
from models.plain_phase_cnn import PlainPhaseCNN
from models.plain_density_cnn import PlainDensityCNN
from models.plain_delay_cnn import PlainDelayCNN
from models.direct_fusion_cnn import DirectFusionCNN
from models.plain_physical_cnn import PlainPhysicalCNN
from torch import nn

CASES = ("full", "plain_phase", "plain_density", "plain_delay", "direct_fusion")


def build_ablation(config, case="full"):
    if case not in CASES:
        raise ValueError(f"Unknown parameter-matched ablation: {case}")
    model = SPAIIceInversionNet(config)
    count = lambda m: sum(p.numel() for p in m.parameters())
    reference = count(model)
    metadata = {"case": case, "reference_total_parameters": reference}
    if case == "direct_fusion":
        model = DirectFusionCNN(config, reference)
        metadata["module_match"] = model.parameter_match
    elif case == "plain_delay":
        physical_budget = (
            count(model.tof_encoder) + count(model.g_cal) + count(model.g_ref)
        )
        head = PlainPhysicalCNN(config.delay_channels)
        model.tof_encoder = PlainDelayCNN(config, physical_budget - count(head))
        model.g_cal = nn.Identity()
        model.g_ref = nn.Identity()
        model.direct_physical_head = head
        metadata["module_match"] = model.tof_encoder.parameter_match
        metadata["physical_head_parameters"] = count(head)
        metadata["analytic_h_constraint"] = False
    elif case != "full":
        fields = {
            "plain_phase": ("skeleton", PlainPhaseCNN),
            "plain_density": ("density_unet", PlainDensityCNN),
        }
        name, constructor = fields[case]
        replacement = constructor(config, count(getattr(model, name)))
        setattr(model, name, replacement)
        metadata["module_match"] = replacement.parameter_match
    metadata["actual_total_parameters"] = count(model)
    metadata["total_relative_difference"] = abs(count(model) - reference) / reference
    model.ablation_metadata = metadata
    return model
