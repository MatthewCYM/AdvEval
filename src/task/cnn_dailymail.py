from datasets import load_dataset


class CNNDailyMailForSummarizationEvaluation:
    def __init__(self, args, evaluators):
        self.args = args
        raw_dataset = load_dataset('hlt-lab/cnn_dailymail_safe', split='test', cache_dir='./cache')
        # raw_dataset = load_dataset('cnn_dailymail', '3.0.0', split='test', cache_dir='./cache')

        self.dataset = raw_dataset.rename_column('article', 'context').rename_column('highlights', 'response')
        self.dataset = [[item['context'], item['response']] for item in self.dataset]
        if self.args.demo_size != -1:
            self.dataset = self.dataset[self.args.demo_start: self.args.demo_start + self.args.demo_size]

    def get_name(self):
        return "cnndailymail" + ("-train" if self.args.use_train_data else "")

    @staticmethod
    def add_args(parser):
        return parser
