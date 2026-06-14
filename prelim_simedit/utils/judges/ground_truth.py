"""Ground-truth 'oracle' judge — provides the ceiling for editing performance.

Always returns the GT label as its diagnosis. No API, no image needed.
"""

from .base import Judge


class GroundTruthJudge(Judge):
    name = "ground_truth"

    def load(self):
        pass

    def judge(self, image, dx, differential="top_1", gt_y16=None):
        gt = str(gt_y16).strip() if gt_y16 else ""

        if differential == "top_1":
            return {
                "diagnosis": gt,
                "reasoning": "Ground truth diagnosis.",
            }
        else:
            return {
                "corrected_differential": f"1. {gt}: Ground truth diagnosis",
            }
