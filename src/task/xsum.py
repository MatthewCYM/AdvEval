from datasets import load_dataset



class XSumForSummarizationEvaluation:
    def __init__(self, args, evaluators):
        self.args = args
        raw_dataset = load_dataset('mattymchen/xsum-safe', split='test',
                                   cache_dir='./cache')

        self.dataset = raw_dataset.rename_column('document', 'context').rename_column('summary', 'response')
        self.dataset = [[item['context'], item['response']] for item in self.dataset]
        if self.args.demo_size != -1:
            self.dataset = self.dataset[self.args.demo_start: self.args.demo_start + self.args.demo_size]

    def get_name(self):
        return "xsum" + ("-train" if self.args.use_train_data else "")

    @staticmethod
    def add_args(parser):
        return parser