from .base import Evaluator
from rquge_score.scorer import RQUGE


class RQUGEEvaluator(Evaluator):
    def __init__(self, args):
        super().__init__(args)
        self.rquge_model = RQUGE(sp_scorer_path="alirezamsh/quip-512-mocha", qa_model_path="allenai/unifiedqa-v2-t5-large-1363200", device='cuda')

    def get_name(self):
        return 'rquge'

    def get_score(self, context, response, ground_truth, context2=None):
        score = self.rquge_model.scorer(context, response, context2)

        # print(results)
        return round(score * 20, 1)
