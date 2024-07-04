from datasets import load_dataset
from .dialog_eval import BaseTaskForResponseEvaluation


class DailyDialogForResponseEvaluation(BaseTaskForResponseEvaluation):
    def __init__(self, args, evaluators):
        super().__init__(args, evaluators)
        raw_dataset = load_dataset('hlt-lab/daily_dialog_safe', split='test' if args.use_train_data else 'validation',
                                   cache_dir='./cache')

        self.dataset = self.preprocess(raw_dataset)
        if self.args.demo_size != -1:
            self.dataset = self.dataset[self.args.demo_start: self.args.demo_start + self.args.demo_size]

    def get_name(self):
        return "dailydialog" + ("-train" if self.args.use_train_data else "") + f"-{self.args.demo_start}-{self.args.demo_start + self.args.demo_size}"
